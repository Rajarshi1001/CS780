import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.distributions import Categorical
import time
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from helper import *

'''
This class creates a buffer for storing and retrieving experiences. This is a generic class and can be used
for different agents like NFQ, DQN, DDQN, PER_DDQN, etc.
Following are the methods for this class which are implemented in subsequent cells

```
class ReplayBuffer():
    def __init__(self, bufferSize, **kwargs)
    def store(self, experience)
    def update(self, indices, priorities)
    def collectExperiences(env, state, explorationStrategy, net = None)
    def sample(self, batchSize, **kwargs)
    def splitExperiences(self, experiences)
    def length(self)
```   
'''

class ReplayBuffer():
    def __init__(self, bufferSize, bufferType = 'DQN', **kwargs):
        # this function creates the relevant data-structures, and intializes all relevant variables
        # it can take variable number of parameters like alpha, beta, beta_rate (required for PER)
        # here the bufferType variable can be used to maintain one class for all types of agents
        # using the bufferType parameter in the methods below, you can implement all possible functionalities
        # that could be used for different types of agents
        # permissible values for bufferType = NFQ, DQN, DDQN, D3QN and PER-D3QN

        self.bufferSize = bufferSize
        self.bufferType = bufferType
        self.position = 0
        self.buffer = [] # stores the experiences for sampling later

        if self.bufferType == 'PER-D3QN':
          self.max_priority = 1.0
          self.priorities = []
          self.alpha = kwargs.get('alpha', 1.0)
          self.beta = kwargs.get('beta', 0.4)
          self.beta_rate = kwargs.get('beta_rate', 0.2)


class ReplayBuffer(ReplayBuffer):
    def store(self, experience):
        #stores the experiences, based on parameters in init it can assign priorities, etc.
        #
        #this function does not return anything
        if len(self.buffer) < self.bufferSize:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        if self.bufferType == 'PER-D3QN':
            if len(self.priorities) < self.bufferSize:
                self.priorities.append(self.max_priority)
            else:
                self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.bufferSize


        
class ReplayBuffer(ReplayBuffer):
    def update(self, indices, priorities):
        #this is mainly used for PER-DDQN
        #otherwise just have a pass in this method
        #
        #this function does not return anything
        #
        temp = 1e-5
        if self.bufferType == 'PER-D3QN':
          for index, priority in zip(indices, priorities):
            self.priorities[index] = np.power(priority + temp, self.alpha)
            self.max_priority = max(priority, self.max_priority)
        else:
          pass
    
class ReplayBuffer(ReplayBuffer):
    def collectExperiences(self, env, state, explorationStrategy, countExperiences, net = None):
        #this method allows the agent to interact with the environment starting from a state and it collects
        #experiences during the interaction, it uses network to get the value function and uses exploration strategy
        #to select action. It collects countExperiences and in case the environment terminates before that it returns
        #the function calling this method needs to handle early termination accordingly.
        experiences = []
        initial_temp = 1.0
        min_temp = 0.01
        initial_epsilon = 1.0
        min_epsilon = 0.2
        decay_rate = 0.2

        for idx in range(countExperiences):
          if explorationStrategy == 'greedy':
            action = selectGreedyAction(net, state)
          elif explorationStrategy == 'softmax':
            temp = decayTemperature(initial_temp, min_temp, decay_rate, idx)
            action = selectSoftMaxAction(net, state, temp)
          elif explorationStrategy == 'epsilon greedy':
            epsilon = decayEpsilon(initial_epsilon, min_epsilon, decay_rate, idx)
            action = selectEpsilonGreedyAction(net, state, epsilon)

          next_state, reward, terminated, truncated, _ = env.step(action)
          # Store the experience (state, action, reward, next_state, done)
          experience = (state, action, reward, next_state, terminated)
          # self.store(experiences)
          experiences.append(experience)

          if terminated or truncated:
              state = env.reset()
          else:
              state = next_state
          if (terminated or truncated) and len(experiences) < countExperiences:
              break

        return

class ReplayBuffer(ReplayBuffer):
    def sample(self, batchSize, **kwargs):
        # this method returns batchSize number of experiences
        # based on extra arguments, it could do sampling or it could return the latest batchSize experiences or
        # via some other strategy
        #
        # in the case of Prioritized Experience Replay (PER) the sampling needs to take into account the priorities
        #
        # this function returns experiences samples
        #
        if self.bufferType == 'PER-D3QN':
          total_priority = sum(self.priorities)
          probs = [priority / total_priority for priority in self.priorities]
          sampled_indices = np.random.choice(len(self.buffer), batchSize, p = probs) # sampling a minibatch from the replay buffer
          samples = [self.buffer[idx] for idx in sampled_indices]
          weights = [(len(self.buffer) * probs[idx]) ** -self.beta for idx in sampled_indices]
          self.beta = min(1.0, self.beta + self.beta_rate)
          return samples, weights, sampled_indices
        else:
          samples = random.sample(self.buffer, min(len(self.buffer), batchSize))
          return samples
      
class ReplayBuffer(ReplayBuffer):
    def splitExperiences(self, experiences):
        #it takes in experiences and gives the following:
        #states, actions, rewards, nextStates, dones
        #
        states, actions, rewards, nextStates, dones = list(map(list, zip(*experiences)))
        return states, actions, rewards, nextStates, dones
    
class ReplayBuffer(ReplayBuffer):
    def length(self):
        #tells the number of experiences stored in the internal buffer

        return len(self.buffer)