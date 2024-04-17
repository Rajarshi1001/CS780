# all imports go in here
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pygame
import swig
import time
from tqdm import tqdm
import random
from itertools import count, cycle


"""
    This class creates a buffer for storing and retrieving experiences. This is a generic class and can be used
for different agents like NFQ, DQN, DDQN, PER_DDQN, etc.
Following are the methods for this class which are implemented in subsequent cells

```
class ReplayBuffer():
    def __init__(self, bufferSize, batch_size, seed)
    def store(self, state, action, reward, next_state, done)
    def sample(self, batchSize)
    def length(self)
"""

class ReplayBuffer():
    def __init__(self, bufferSize, bufferType = 'DQN', **kwargs):
        # this function creates the relevant data-structures, and intializes all relevant variables
        # it can take variable number of parameters like alpha, beta, beta_rate (required for PER)
        # here the bufferType variable can be used to maintain one class for all types of agents
        # using the bufferType parameter in the methods below, you can implement all possible functionalities
        # that could be used for different types of agents
        # permissible values for bufferType = NFQ, DQN, DDQN, D3QN and PER-D3QN

        #Your code goes in here
        self.ss_mem = np.empty(shape=(bufferSize), dtype=np.ndarray)
        self.as_mem = np.empty(shape=(bufferSize), dtype=np.ndarray)
        self.rs_mem = np.empty(shape=(bufferSize), dtype=np.ndarray)
        self.ps_mem = np.empty(shape=(bufferSize), dtype=np.ndarray)
        self.ds_mem = np.empty(shape=(bufferSize), dtype=np.ndarray)

        self.max_size = bufferSize
        self.bufferType = bufferType
        self._idx = 0
        self.size = 0
        # Assume default batch size for sampling if not provided
        self.default_batch_size = 64

        return
    
class ReplayBuffer(ReplayBuffer):
    def store(self, experience):
        #stores the experiences, based on parameters in init it can assign priorities, etc.
        #
        #this functacion does not return anything
        #
        #Your code goes in here
        if(self.bufferType == "D3QN-PER"):
            priority = 1.0
            if self.n_entries > 0:
                priority = self.memory[
                    :self.n_entries,
                    self.td_error_index].max()
            self.memory[self.next_index,
                        self.td_error_index] = priority
            self.memory[self.next_index,
                        self.sample_index] = np.array(experience, dtype = 'object')
            self.n_entries = min(self.n_entries + 1, self.max_samples)
            self.next_index += 1
            self.next_index = self.next_index % self.max_samples

        else:

            s, a, r, p, d = experience
            self.ss_mem[self._idx] = s
            self.as_mem[self._idx] = a
            self.rs_mem[self._idx] = r
            self.ps_mem[self._idx] = p
            self.ds_mem[self._idx] = d

            self._idx = (self._idx + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)

        return
    
class ReplayBuffer(ReplayBuffer):
    def collectExperiences(env, state, explorationStrategy, countExperiences, net = None):
        #this method allows the agent to interact with the environment starting from a state and it collects
        #experiences during the interaction, it uses network to get the value function and uses exploration strategy
        #to select action. It collects countExperiences and in case the environment terminates before that it returns
        #the function calling this method needs to handle early termination accordingly.
        #
        #this function does not return anything
        #
        #Your code goes in here
        experiences = []
        for _ in range(countExperiences):
            action = explorationStrategy(state, net)  # Assuming explorationStrategy uses net
            next_state, reward, done, _ = env.step(action)
            experiences.append((state, action, reward, next_state, done))
            if done:
                break
            state = next_state
        return experiences
    
class ReplayBuffer(ReplayBuffer):
    def sample(self, batchSize=None, **kwargs):
        # this method returns batchSize number of experiences
        # based on extra arguments, it could do sampling or it could return the latest batchSize experiences or
        # via some other strategy
        #
        # in the case of Prioritized Experience Replay (PER) the sampling needs to take into account the priorities
        #
        # this function returns experiences samples
        #
        #Your code goes in here
        if batchSize is None:
            batchSize = self.default_batch_size
        idxs = np.random.choice(self.size, batchSize, replace=False)
        experiences = (np.vstack(self.ss_mem[idxs]), \
                        np.vstack(self.as_mem[idxs]), \
                        np.vstack(self.rs_mem[idxs]), \
                        np.vstack(self.ps_mem[idxs]), \
                        np.vstack(self.ds_mem[idxs]))
        return experiences
    
class ReplayBuffer(ReplayBuffer):
    def splitExperiences(self, experiences):
        #it takes in experiences and gives the following:
        #states, actions, rewards, nextStates, dones
        #
        #Your code goes in here
        states, actions, rewards, nextStates, dones = experiences
        return states, actions, rewards, nextStates, dones
    

class ReplayBuffer(ReplayBuffer):
    def length(self):
        #tells the number of experiences stored in the internal buffe
        return self.size