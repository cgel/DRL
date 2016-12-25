# Deep Reinforcement Learning

This is a project designed to make the development, training and testing of DRL algorithms easy. Any new algorithms can be implemented as an Agent class using tensorflow. Then, they can be compared against other agents under the exact same setting.

The currently implemented agents are:
* **DQN**
* **PDQN** _Research project. Work in progress_


## Easy to understand
Use ```tensorboard --logdir=log``` to view all the runs. They will be all organized by a number, the name of the agent and the name of the environment.
![tb_breakout-v0](./assets/tb_runs.jpg){:style="float: right;margin-right: 15px;margin-left: 15px;margin-bottom: 7px;"}

You can also see the arguments of each run in ```log/[run_name]/config.txt```

## Performance
#### DQN on breakout-v0 (unfinished run)
![tb_breakout-v0](./assets/tb_breakout-v0.jpg)
