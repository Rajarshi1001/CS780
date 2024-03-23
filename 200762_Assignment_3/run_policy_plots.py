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
from replaybuffer import ReplayBuffer
from helper import *
from reinforce import *
from vpg import *


# Use either Cart pole or Mountain Car Environment
env1 = gym.make('CartPole-v0')
env2 = gym.make('MountainCar-v0')


def runDeepPolicyBasedAgents(question_no):
    # this function will initialize 5 different instances of the env (using different seeds), run all the agents
    # over these different instances. Collects results and generate the plots state above.
    # generate your plots in the cells below
    # write the answers to part 11, 12 and 13 in the cells below the plot-cells.
    if question_no == 1: # train rewards for cartpole
        
        plot_stats_reinforce(num_instances = 5,
               train_episodes = 1000,
                chosen_env = env1, 
                plot_train_rewards = True, 
                plot_train_times = False, 
                plot_wall_clock_times = False,
                plot_total_steps = False, 
                plot_eval_rewards = False)

        plot_stats_vpg(num_instances = 5,
               train_episodes = 1000,
                chosen_env = env1, 
                plot_train_rewards = True, 
                plot_train_times = False, 
                plot_wall_clock_times = False,
                plot_total_steps = False, 
                plot_eval_rewards = False)
    
    elif question_no == 2: # train rewards for mountain car
        
        plot_stats_reinforce(num_instances = 5,
               train_episodes = 1000,
                chosen_env = env2, 
                plot_train_rewards = True, 
                plot_train_times = False, 
                plot_wall_clock_times = False,
                plot_total_steps = False, 
                plot_eval_rewards = False)

        plot_stats_vpg(num_instances = 5,
               train_episodes = 1000,
                chosen_env = env2, 
                plot_train_rewards = True, 
                plot_train_times = False, 
                plot_wall_clock_times = False,
                plot_total_steps = False, 
                plot_eval_rewards = False)
        
    elif question_no == 3: # eval rewards for cartpole
        
        plot_stats_reinforce(num_instances = 5,
               train_episodes = 1000,
                chosen_env = env1, 
                plot_train_rewards = False, 
                plot_train_times = False, 
                plot_wall_clock_times = False,
                plot_total_steps = False, 
                plot_eval_rewards = True)

        plot_stats_vpg(num_instances = 5,
               train_episodes = 1000,
                chosen_env = env1, 
                plot_train_rewards = False, 
                plot_train_times = False, 
                plot_wall_clock_times = False,
                plot_total_steps = False, 
                plot_eval_rewards = True)

    elif question_no == 4: # eval rewards for mountain car
        
        plot_stats_reinforce(num_instances = 5,
               train_episodes = 1000,
                chosen_env = env2, 
                plot_train_rewards = False, 
                plot_train_times = False, 
                plot_wall_clock_times = False,
                plot_total_steps = False, 
                plot_eval_rewards = True)

        plot_stats_vpg(num_instances = 5,
               train_episodes = 1000,
                chosen_env = env2, 
                plot_train_rewards = False, 
                plot_train_times = False, 
                plot_wall_clock_times = False,
                plot_total_steps = False, 
                plot_eval_rewards = True)
    
    elif question_no == 5: # total steps for cartpole
        
        plot_stats_reinforce(num_instances = 5,
               train_episodes = 1000,
                chosen_env = env1, 
                plot_train_rewards = False, 
                plot_train_times = False, 
                plot_wall_clock_times = False,
                plot_total_steps = True, 
                plot_eval_rewards = False)

        plot_stats_vpg(num_instances = 5,
               train_episodes = 1000,
                chosen_env = env1, 
                plot_train_rewards = False, 
                plot_train_times = False, 
                plot_wall_clock_times = False,
                plot_total_steps = True, 
                plot_eval_rewards = False)
    
    elif question_no == 6: # total steps for mountain-car
        
        plot_stats_reinforce(num_instances = 5,
               train_episodes = 1000,
                chosen_env = env2, 
                plot_train_rewards = False, 
                plot_train_times = False, 
                plot_wall_clock_times = False,
                plot_total_steps = True, 
                plot_eval_rewards = False)

        plot_stats_vpg(num_instances = 5,
               train_episodes = 1000,
                chosen_env = env2, 
                plot_train_rewards = False, 
                plot_train_times = False, 
                plot_wall_clock_times = False,
                plot_total_steps = True, 
                plot_eval_rewards = False)
    
    elif question_no == 7: # train times for cartpole
        
        plot_stats_reinforce(num_instances = 5,
               train_episodes = 1000,
                chosen_env = env1, 
                plot_train_rewards = False, 
                plot_train_times = True, 
                plot_wall_clock_times = False,
                plot_total_steps = False, 
                plot_eval_rewards = False)

        plot_stats_vpg(num_instances = 5,
               train_episodes = 1000,
                chosen_env = env1, 
                plot_train_rewards = False, 
                plot_train_times = True, 
                plot_wall_clock_times = False,
                plot_total_steps = False, 
                plot_eval_rewards = False)
        
    elif question_no == 8: # train times for mountain-car
        
        plot_stats_reinforce(num_instances = 5,
               train_episodes = 1000,
                chosen_env = env2, 
                plot_train_rewards = False, 
                plot_train_times = True, 
                plot_wall_clock_times = False,
                plot_total_steps = False, 
                plot_eval_rewards = False)

        plot_stats_vpg(num_instances = 5,
               train_episodes = 1000,
                chosen_env = env2, 
                plot_train_rewards = False, 
                plot_train_times = True, 
                plot_wall_clock_times = False,
                plot_total_steps = False, 
                plot_eval_rewards = False)
    
    elif question_no == 9:
        
        plot_stats_reinforce(num_instances = 5,
               train_episodes = 1000,
                chosen_env = env1, 
                plot_train_rewards = False, 
                plot_train_times = False, 
                plot_wall_clock_times = True,
                plot_total_steps = False, 
                plot_eval_rewards = False)

        plot_stats_vpg(num_instances = 5,
               train_episodes = 1000,
                chosen_env = env1, 
                plot_train_rewards = False, 
                plot_train_times = False, 
                plot_wall_clock_times = True,
                plot_total_steps = False, 
                plot_eval_rewards = False)
    
    elif question_no == 10:
        
        plot_stats_reinforce(num_instances = 5,
               train_episodes = 1000,
                chosen_env = env2, 
                plot_train_rewards = False, 
                plot_train_times = False, 
                plot_wall_clock_times = True,
                plot_total_steps = False, 
                plot_eval_rewards = False)

        plot_stats_vpg(num_instances = 5,
               train_episodes = 1000,
                chosen_env = env2, 
                plot_train_rewards = False, 
                plot_train_times = False, 
                plot_wall_clock_times = True,
                plot_total_steps = False, 
                plot_eval_rewards = False)
    
    
    
## Depending on what you want to plot and for which environment
## put your question_no as an argument

if __name__ == '__main__':
    runDeepPolicyBasedAgents(3)    
