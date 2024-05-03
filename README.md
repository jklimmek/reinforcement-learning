# Reinforcement Learning Zoo


Welcome to the Reinforcement Learning Zoo! This repository is a collection of diverse reinforcement learning algorithms designed to beat OpenAI's gym environments. The goal is to continuously expand the repo over time. For simplicity most of the algorithms are trained on LunarLander-v2.


## Implemented Algorithms

1. Deep Q Network (DQN) - Double Q-learning with soft update of target network, trained on LunarLander-v2.
2. Policy Gradients (PG) - Vanilla REINFORCE on LunarLander-v2.
3. Actor-Critic (AC) - Temporal Difference Actor-Critic trained on LunarLander-v2.
4. Proximal Policy Optimization (PPO) - Convolutional neural net inspired by DeepMind's DQN paper trained to play Atari Breakout. After training for 5,000,000 steps the model achieves score of 400. 
5. Rainbow - Convolutional DQN with six improvements from DeepMind's Rainbow paper trained to beat Atari Pacman.