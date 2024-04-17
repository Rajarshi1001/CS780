## CS780 Assignment 4

This assignment consists of the implementation of some state of the art deep reinforcement learning models for continuous action spaces mainly:

- `Deep Deterministic Policy Gradient (DDPG)`
- `Twin Delayed Deep Deterministic Policy Gradient (TD3)`
- `Proximal Policy Optimization (PPO)`

on three different OpenAI gym environments like `Pendulum-v1`, `Hopper-v4` and `HalfCheetah-v1` respectively.

Each of the implementations have their separate classes

1. The replay buffer implementation is provided in replaybuffer.py file.

2. The helper functions including the Value network, Policy Network, Value network for TD3, Policy Network for PPO.

All the plots are available under the   `./images` folder.

```
200762_Assignment_4
|
|----images/
|----CS780-RAJARSHI-DUTTA-200762-ASSIGN-4.zip
|----assignment_4.ipynb
|----helper.py
|----replay_buffer.py
|----ddpg.py
|----td3.py
|----ppo.py
```