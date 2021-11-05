import copy
import time

import gym
from replay_buffer import TransitionBuffer, Batch
from executor import Executor
from trainer import Trainer
from gym import Env
import numpy as np
import sonnet as snt
from datetime import datetime
from logger import MyLogger
import tensorflow as tf

# PARAMS
N_EPISODES = 10000
TRAIN_EVERY_N_TIME_STEPS = 4
EPSILON_DECAY = 0.99999 # exponential decay
EVAL_EPISODES = 5
EVAL_EVERY = 1000

env: Env = gym.make('ma_gym:Lumberjacks-v0')

# Get env specs
observation_dim = env.observation_space.sample()[0].shape[0]
num_agents: int = env.n_agents
num_actions = env.action_space._agents_action_space[0].n

# Logging
logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
logger = MyLogger(logdir=logdir)

# Create networks
q_network = snt.nets.MLP(output_sizes=(128, num_actions))
target_q_network = copy.deepcopy(q_network)

# Replay buffer
buffer = TransitionBuffer(
    observation_dim=observation_dim,
    num_agents=num_agents,
)


# Executor
executor = Executor(
    q_network=target_q_network, # uses the target network
    num_agents=num_agents,
    num_actions=num_actions,
    replay_buffer=buffer,
    epsilon_decay=EPSILON_DECAY,
)

# Trainer
trainer = Trainer(
    q_network=q_network,
    target_q_network=target_q_network,
    num_agents=num_agents,
    num_actions=num_actions,
    replay_buffer=buffer,
)

t = 0 # timestep counter
for episode in range(N_EPISODES):
    observations = env.reset()
    ep_returns = 0
    dones = [False for _ in range(num_agents)]
    logs = {}
    # Training loop
    while not all(dones):
        # Select action
        actions = executor.select_actions(observations)

        # Step environment
        next_observations, rewards, dones, info = env.step(actions)

        if t % TRAIN_EVERY_N_TIME_STEPS == 0 and buffer.can_sample_batch():
            # Periodically train
            loss = trainer.step()
            logs["total loss"] = tf.reduce_sum(loss).numpy()

        # Create a batch from transition
        batch = Batch(
            observations=observations,
            next_observations=next_observations,
            rewards=rewards,
            actions=actions,
            dones=dones,
        )

        # Store transition in replay buffer
        executor.observe(batch)

        # Add rewards to episode return
        ep_returns += sum(rewards)

        # Critical!! 
        observations = next_observations

        # Increment timestep counter
        t += 1

    # Log to tensorboard
    logs["episode_return"] = ep_returns
    logs["epsilon"] = executor.epsilon
    logger.write(logs)
    
    if episode % 100 == 0:
        # Periodically log to terminal
        print(f"Episode {episode}: {logs}")

    if episode % EVAL_EVERY == 0:
        returns = []
        # Eval loop
        for eval_episode in range(EVAL_EPISODES):
            observations = env.reset()
            
            ep_return = 0
            dones = [False for _ in range(num_agents)]
            while not all(dones):
                env.render() # Render
                time.sleep(0.08) # sleep to slow down annimation

                # Select greedy action
                actions = executor.select_greedy_actions(observations)

                # Step environment
                next_observations, rewards, dones, info = env.step(actions)

                # Add rewards to the return
                ep_return += sum(rewards)

                # Critical!!
                observations = next_observations

            returns.append(ep_return)

        print(f"Evaluation average return: {np.mean(returns)}")

env.close()