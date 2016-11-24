# Deep Reinforcement Learning

This is a project designed to make the development, training and testing of DRL algorithms easy. Any new algorithms can be implemented as an Agent class using tensorflow. Then, they can be compared against other agents under the exact same setting.

The currently implemented agents are:
* DQN  ```agents/deepmind_dqn.py```
* PDQN

Use ```tensorboard --logdir=log``` to view all the runs.

The arguments of each "run" can be found in ```log/[run_name]/config.txt```
