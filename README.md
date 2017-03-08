# openai_playground
This collection of Python modules implements some Reinforcement Learning algorithms, most notably **Deep Q Networks (DQN)** and **Prioritized Experience Replay (PER)**, where the proportional prioritization variant has been implemented.. It has been built to solve [OpenAI Gym environments]((https://gym.openai.com/), although it has only been tested on classic control environments with discrete action sets.

The code supports a variety of hyper parameters, that are usually tuned to particular environments. Bayesian optimization with [Scikit-Optimize](https://scikit-optimize.github.io/) is a simple way of tuning those hyper parameters.

The code uses [Tensorflow](https://www.tensorflow.org/) to model a value function for a Reinforcement Learning agent.
I've run it with Tensorflow 1.0 on Python 3.5 under Windows 7.

## References
1. [Deep Learning tutorial](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Resources_files/deep_rl.pdf), David Silver, Google DeepMind.
2. [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952), T. Schaul., J. Quan and D. Silver. Feb 2016.
3. [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461), Hado van Hasselt, Arthur Guez, David Silver. Dec 2015.
