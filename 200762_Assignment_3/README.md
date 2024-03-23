## CS780 Assignment 3

This assignment primarily includes the implementation of 5 *Value Based Deep RL models* namely:
- `Neural Fitted Q Iteration (NFQ)`
- `Deep Q Network (DQN)`
- `Double Deep Q Network (DDQN)` 
- `Dueling Double Deep Q Network (D3QN)`
- `Dueling Double Deep Q Network with Prioritized Experience Replay (D3QN-PER)`

and 2 *Policy Based Deep RL models* namely:
- `REINFORCE`
- `Vanilla Policy Gradient (VPG)`

on two different OpenAI gym environments like __Cartpole-v0__ and __MountainCar-v1__ respectively.


1. Each of the implementations have their separate classes

2. The replay buffer implementation is provided in  `replaybuffer.py` file.

2. The helper functions including the Value network, Policy Network, Duelling Network, agent exploration strategies, hyperparemeters are provided in `helper.py` file

3. The value plotting funtion and policy plotting functions are provided in `run_value_plots.py` and `run_policy_plots.py` respectively.

4. All the plots are available under the `images` folder.

Directory Structure

200762_Assignment_3
|
|----images/
|----assignment_3.ipynb



