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


#Value Network
def createValueNetwork(inDim, outDim, hDim=[32, 32], activation=F.relu):
    #this creates a Feed Forward Neural Network class and instantiates it and returns the class
    #the class should be derived from torch nn.Module and it should have init and forward method at the very least
    #the forward function should return q-value for each possible action
    class ValueNetwork(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dims, activation_fc):
            super(ValueNetwork, self).__init__()
            self.activation_fc = activation_fc

            self.first_layer = nn.Linear(input_dim, hidden_dims[0])
            self.second_layer = nn.Linear(hidden_dims[0] + output_dim, hidden_dims[1])
            self.hidden_layers = nn.ModuleList()
            for i in range(1, len(hidden_dims) - 1):
                self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
            
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.to(self.device)

        def _format(self, state, action=None):
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32, device=self.device)
            if state.dim() == 1:
                state = state.unsqueeze(0)

            if action is not None:
                if not isinstance(action, torch.Tensor):
                    action = torch.tensor(action, dtype=torch.float32, device=self.device)
                if action.dim() == 1:
                    action = action.unsqueeze(0)
            return state, action

        def forward(self, state, action):
            state, action = self._format(state, action)
            x = self.activation_fc(self.first_layer(state))
            x = torch.cat((x, action), dim=-1)  
            x = self.activation_fc(self.second_layer(x))
            for hidden_layer in self.hidden_layers:
                x = self.activation_fc(hidden_layer(x))
            x = self.output_layer(x)
            return x


        def load(self, experiences):
            states, actions, rewards, new_states, is_terminals = experiences
            states = torch.from_numpy(states).float().to(self.device)
            actions = torch.from_numpy(actions).long().to(self.device)
            new_states = torch.from_numpy(new_states).float().to(self.device)
            rewards = torch.from_numpy(rewards).float().to(self.device)
            is_terminals = torch.from_numpy(is_terminals).float().to(self.device)
            return states, actions, rewards, new_states, is_terminals

    return ValueNetwork(inDim, outDim, hDim, activation)



#Value Network
def createValueNetwork_TD3(inDim, outDim, hDim=[32, 32], activation=F.relu):
    #this creates a Feed Forward Neural Network class and instantiates it and returns the class
    #the class should be derived from torch nn.Module and it should have init and forward method at the very least
    #the forward function should return q-value for each possible action
    class ValueNetwork(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dims, activation_fc):
            super(ValueNetwork, self).__init__()
            self.activation_fc = activation_fc

            # first network
            self.first_layer1 = nn.Linear(input_dim, hidden_dims[0])
            self.second_layer1 = nn.Linear(hidden_dims[0] + output_dim, hidden_dims[1])
            self.hidden_layers1 = nn.ModuleList()
            for i in range(1, len(hidden_dims) - 1):
                self.hidden_layers1.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.output_layer1 = nn.Linear(hidden_dims[-1], output_dim)
            
            # second network
            self.first_layer2 = nn.Linear(input_dim, hidden_dims[0])
            self.second_layer2 = nn.Linear(hidden_dims[0] + output_dim, hidden_dims[1])
            self.hidden_layers2 = nn.ModuleList()
            for i in range(1, len(hidden_dims) - 1):
                self.hidden_layers2.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.output_layer2 = nn.Linear(hidden_dims[-1], output_dim)
            
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.to(self.device)

        def _format(self, state, action=None):
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32, device=self.device)
            if state.dim() == 1:
                state = state.unsqueeze(0)

            if action is not None:
                if not isinstance(action, torch.Tensor):
                    action = torch.tensor(action, dtype=torch.float32, device=self.device)
                if action.dim() == 1:
                    action = action.unsqueeze(0)
            return state, action

        def forward(self, state, action):
            state1, action = self._format(state, action)
            q1 = self.activation_fc(self.first_layer1(state1))
            q1 = torch.cat((q1, action), dim=-1)  
            q1 = self.activation_fc(self.second_layer1(q1))
            for hidden_layer1 in self.hidden_layers1:
                q1 = self.activation_fc(hidden_layer1(q1))
            q1 = self.output_layer1(q1)
            
            # second network pass
            state2, action = self._format(state, action)
            q2 = self.activation_fc(self.first_layer2(state2))
            q2 = torch.cat((q2, action), dim=-1)  
            q2 = self.activation_fc(self.second_layer2(q2))
            for hidden_layer2 in self.hidden_layers1:
                q2 = self.activation_fc(hidden_layer2(q2))
            q2 = self.output_layer2(q2)
            
            return q1, q2


        def load(self, experiences):
            states, actions, rewards, new_states, is_terminals = experiences
            states = torch.from_numpy(states).float().to(self.device)
            actions = torch.from_numpy(actions).long().to(self.device)
            new_states = torch.from_numpy(new_states).float().to(self.device)
            rewards = torch.from_numpy(rewards).float().to(self.device)
            is_terminals = torch.from_numpy(is_terminals).float().to(self.device)
            return states, actions, rewards, new_states, is_terminals

    return ValueNetwork(inDim, outDim, hDim, activation)


#Policy Network
def createPolicyNetwork(inDim, outDim, envActionRange = (-1, 1), hDim = [32,32], activation = F.relu):
    #this creates a Feed Forward Neural Network class and instantiates it and returns the class
    #the class should be derived from torch nn.Module and it should have init and forward method at the very least
    #the forward function should return action logit vector
    #Your code goes in here

    class policyNetwork(nn.Module):
        def __init__(self, input_dim, output_dim, envActionRange, num_outputs, hidden_dims, activation_fn):
            super(policyNetwork, self).__init__()
            self.activation_fn = activation_fn
            self.envActionRange = envActionRange

            self.layers = nn.ModuleList()
            self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
            for i in range(len(hidden_dims)-1):
                self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.layers.append(nn.Linear(hidden_dims[-1], output_dim))
            
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        def _format(self, state):
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32)
            if state.dim() == 1:
                state = state.unsqueeze(0)
            return state

        def forward(self, x):
            min_action, max_action = self.envActionRange
            x = self._format(x)
            for layer in self.layers[:-1]:
                x = self.activation_fn(layer(x))
            x = F.tanh(self.layers[-1](x))

            actions = (x + 1.0) * ((max_action - min_action)/(torch.tensor(2.0))) + torch.tensor(min_action)

            return actions

    return policyNetwork(inDim, outDim, envActionRange, envActionRange[0].shape[0], hDim, activation)


# policy network for PPO
def createPolicyNetworkPPO(inDim, outDim, envActionRange = (-1, 1), hDim = [32,32], activation = F.relu):
    #this creates a Feed Forward Neural Network class and instantiates it and returns the class
    #the class should be derived from torch nn.Module and it should have init and forward method at the very least
    #the forward function should return action logit vector
    #Your code goes in here
    
    class policyNetwork(nn.Module):
        def __init__(self, input_dim, output_dim, envActionRange, num_outputs, hidden_dims, activation_fn):
            super(policyNetwork, self).__init__()
            self.activation_fn = activation_fn
            self.envActionRange = envActionRange

            self.layers = nn.ModuleList()
            self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
            for i in range(len(hidden_dims)-1):
                self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            # self.layers.append(nn.Linear(hidden_dims[-1], output_dim))
            self.mean_layer = nn.Linear(hidden_dims[-1], output_dim)
            self.log_std_layer = nn.Parameter(torch.zeros(output_dim))
            
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
        def _format(self, state):
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32)
            if state.dim() == 1:
                state = state.unsqueeze(0)
            return state
        
        def forward(self, x):
            x = self._format(x)
            for layer in self.layers:
                x = self.activation_fn(layer(x))
            mean = self.mean_layer(x)
            std = torch.exp(self.log_std_layer.clamp(-20, 2))
            normal = torch.distributions.Normal(mean, std)
            actions = normal.sample()
            log_probs = normal.log_prob(actions)
            
            min_action, max_action = self.envActionRange
            min_action = torch.tensor(min_action, device=self.device, dtype=torch.float32)  
            max_action = torch.tensor(max_action, device=self.device, dtype=torch.float32)  
            actions = torch.tanh(actions)
            actions = (actions + 1.0) / 2.0 * (max_action - min_action) + min_action     
            actions = actions.view(-1, 1)
            # log_probs = log_probs.view(-1, 1)
            
            return actions, log_probs, normal.entropy()
            
        
    return policyNetwork(inDim, outDim, envActionRange, envActionRange[0].shape[0], hDim, activation)


def reset_env(env, seed=None, options=None):
    # Reset the environment with seed and options, handling new seeding mechanism
    initial_observation, info = env.reset(seed=seed, options=options)
    return initial_observation, info

