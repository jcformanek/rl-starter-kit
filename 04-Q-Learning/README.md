# DQN
In the notebook `dqn.ipynb` I have done a minimal implementation of [DQN](https://arxiv.org/abs/1312.5602). The implementation successfully learns on `LunarLander` gym environment. The primary shortfall of this implementation is that the replay buffer gets really slow to sample as it grows in size. So the training time dramatically slows down as training progresses. If you want to improve this implementation, consider building a faster replay buffer.  

<a href="https://colab.research.google.com/github/jcformanek/rl-starter-kit/blob/main/05-Q-Learning/dqn.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
