import sonnet as snt
import tensorflow as tf
import trfl

class Trainer:

    def __init__(
        self,
        q_network: snt.Module,
        target_q_network,
        num_agents,
        num_actions,
        replay_buffer,
        target_update_period=100,
        learning_rate=5e-4,
        discount=0.99,
    ):
        self.q_network = q_network
        self.target_q_network = target_q_network

        self.num_agents = num_agents
        self.num_actions = num_actions
        self.replay_buffer = replay_buffer

        self.target_update_period = target_update_period
        self.optimiser = snt.optimizers.Adam(learning_rate=learning_rate)
        self.discount = discount

        self.trainer_steps = 0

    @tf.function
    def _step(self, batch):
        # Get the relevant quantities
        # Make agent dim the leading dim for easier access. (N is the number of agents)
        rewards = tf.transpose(batch.rewards, perm=[1, 0]) # [B, N] -> [N, B]
        actions = tf.transpose(batch.actions, perm=[1, 0]) # [B, N] -> [N, B]
        dones = tf.transpose(batch.dones, perm=[1, 0]) # [B, N] -> [N, B]
        observations = tf.transpose(batch.observations, perm=[1, 0, 2]) # [B, N, Obs] -> [N, B, Obs]
        next_observations = tf.transpose(batch.next_observations, perm=[1, 0, 2]) # [B, N, Obs] -> [N, B, Obs]

        with tf.GradientTape(persistent=True) as tape:
            loss = []

            for agent in range(self.num_agents):

                q_values = self.q_network(observations[agent])  # [B, A]
                selected_q_value = trfl.batched_index(q_values, actions[agent])

                # Standard Q-learning
                next_q_values = self.target_q_network(next_observations[agent])
                max_next_q_value = tf.reduce_max(next_q_values, axis=-1)

                # Bellman target
                target = rewards[agent] + self.discount * (1 - dones[agent]) * max_next_q_value

                # Temporal difference
                td_error = selected_q_value - target

                # Mean-squared error
                agent_loss = tf.reduce_mean(td_error ** 2)

                loss.append(agent_loss)

        # Backward pass over agents
        for agent in range(self.num_agents):
            # Get trainable variables
            trainable_variables = self.q_network.trainable_variables

            # Get gradients
            gradients = tape.gradient(target=loss[agent], sources=trainable_variables)

            # Apply gradients to the variables
            self.optimiser.apply(updates=gradients, parameters=trainable_variables)

        del tape

        return loss

    def step(self):

        # Sample replay 
        batch = self.replay_buffer.sample()

        # Training
        loss = self._step(batch)

        # Periodically update target network
        self._update_target_network()

        return loss

    def _update_target_network(self):
        """Update target network."""
        if self.trainer_steps % self.target_update_period == 0:
            online_variables = (*self.q_network.variables,)
            target_variables = (*self.target_q_network.variables,)

            for src, dest in zip(online_variables, target_variables):
                dest.assign(src)

        self.trainer_steps += 1
