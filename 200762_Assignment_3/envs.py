import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.distributions import Categorical
import time
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as a



# Create CartPole environment
#https://gymnasium.farama.org/environments/classic_control/cart_pole/

env1 = gym.make('CartPole-v0')
# env.seed(34)
s = env1.reset()
# print(s)
print("Observation Space = ")
print(env1.observation_space)
print("Action Space = ")
print(env1.action_space)
done = False
for episode in range(20):
    print("In episode {}".format(episode))
    for i in range(100):
        env1.render()
        print(s)
        a = env1.action_space.sample()
        step_result = env1.step(a)
        s, r, terminated, truncated, _ = env1.step(a)
        if terminated or truncated:
            print("Finished after {} timestep".format(i+1))
env1.close()

# Create MountainCar environment:
# https://gymnasium.farama.org/environments/classic_control/mountain_car/

env2 = gym.make('MountainCar-v0')
# env.seed(45)
s = env2.reset()
print("Observation Space = ")
print(env2.observation_space)
print("Action Space = ")
print(env2.action_space)
done = False
for episode in range(20):
    print("In episode {}".format(episode))
    for i in range(100):
        env2.render()
        print(s)
        a = env2.action_space.sample()
        s, r, terminated, truncated, _ = env2.step(a)
        if terminated or truncated:
            print("Finished after {} timestep".format(i+1))
env2.close()

