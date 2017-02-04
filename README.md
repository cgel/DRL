# Deep Reinforcement Learning

This is a project designed to make the development, training and testing of DRL algorithms easy. Any new algorithms can be implemented as an Agent class using tensorflow. Then, they can be compared against each other under the exact same setting.

Agents are derived from the ```BaseAgent``` class, so that a particular algorithms can be implemented in very few lines. And if a new agent is very similar to an old one, you can just derive from the old one and re-implement only the new part.

The currently implemented agents are:
* **DQN**
* **DoubleDQN**
* **DuelingDoubleDQN**

Example usage:
```python train.py -agent DQN -device 1 -env_name Breakout-v0```

## Understand the runs
<img align="right" src="./assets/tb_runs.jpg">

When you are testing many different algorithms in many different settings things can get confusing. DRL tries to organize everything you might need in the simplest manner.

Use ```tensorboard --logdir=log``` to view all the runs. They will be all organized by a number, the name of the agent and the name of the environment.

You can also see the arguments of each run in ```log/[run_name]/config.txt```

## Performance
**I have very limited compute resources and they are being uses for my research. I would really appreciate if somebody wanted to help by running DQN in different environments.**
#### DQN on breakout-v0 for 50,000,000 steps
![tb_breakout-v0](./assets/tb_breakout-v0.jpg)
_Note that Breakout-v0 != Breakout from DQN paper. It has 6 actions while the one from the paper had 4. This run clearly was still learning and needed to continue. When I have time I will test on the original Breakout._
