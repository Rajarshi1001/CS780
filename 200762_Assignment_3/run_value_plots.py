
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
from nfq import *
from dqn import *
from ddqn import *
from d3qn import *
from d3qn_per import *


# Use either Cart pole or Mountain Car Environment
env1 = gym.make('CartPole-v0')
env2 = gym.make('MountainCar-v0')


## Plotting functions


def runDeepValueBasedAgents(question_no):
    # this function will initialize 5 different instances of the env (using different seeds), run all the agents
    # over these different instances. Collects results and generate the plots state above.
    # generate your plots in the cells below
    # write the answers to part 11, 12 and 13 in the cells below the plot-cells.
    if question_no == 1: # train rewards for cartpole
        
        plot_stats_nfq(num_instances = 5,
               train_episodes = 2000,
                chosen_env = env1, 
                plot_train_rewards = True, 
                plot_train_times = False, 
                plot_wall_clock_times = False,
                plot_total_steps = False, 
                plot_eval_rewards = False)
        
        plot_stats_dqn(num_instances = 5,
               train_episodes = 2000,
                chosen_env = env1, 
                plot_train_rewards = True, 
                plot_train_times = False, 
                plot_wall_clock_times = False,
                plot_total_steps = False, 
                plot_eval_rewards = False)
        
        plot_stats_ddqn(num_instances = 5,
               train_episodes = 2000,
                chosen_env = env1, 
                plot_train_rewards = True, 
                plot_train_times = False, 
                plot_wall_clock_times = False,
                plot_total_steps = False, 
                plot_eval_rewards = False)
        
        plot_stats_d3qn(num_instances = 5,
               train_episodes = 2000,
                chosen_env = env1, 
                plot_train_rewards = True, 
                plot_train_times = False, 
                plot_wall_clock_times = False,
                plot_total_steps = False, 
                plot_eval_rewards = False)
        
        plot_stats_d3qn_per(num_instances = 5,
               train_episodes = 10000,
                chosen_env = env1, 
                plot_train_rewards = True, 
                plot_train_times = False, 
                plot_wall_clock_times = False,
                plot_total_steps = False, 
                plot_eval_rewards = False)

    elif question_no == 2: # train rewards for mountain car
        
       plot_stats_nfq(num_instances = 5,
              train_episodes = 2000,
              chosen_env = env2, 
              plot_train_rewards = True, 
              plot_train_times = False, 
              plot_wall_clock_times = False,
              plot_total_steps = False, 
              plot_eval_rewards = False)
       
       plot_stats_dqn(num_instances = 5,
              train_episodes = 2000,
              chosen_env = env2, 
              plot_train_rewards = True, 
              plot_train_times = False, 
              plot_wall_clock_times = False,
              plot_total_steps = False, 
              plot_eval_rewards = False)
       
       plot_stats_ddqn(num_instances = 5,
              train_episodes = 2000,
              chosen_env = env2, 
              plot_train_rewards = True, 
              plot_train_times = False, 
              plot_wall_clock_times = False,
              plot_total_steps = False, 
              plot_eval_rewards = False)
       
       plot_stats_d3qn(num_instances = 5,
              train_episodes = 2000,
              chosen_env = env2, 
              plot_train_rewards = True, 
              plot_train_times = False, 
              plot_wall_clock_times = False,
              plot_total_steps = False, 
              plot_eval_rewards = False)
       
       plot_stats_d3qn_per(num_instances = 5,
              train_episodes = 3000,
              chosen_env = env2, 
              plot_train_rewards = True, 
              plot_train_times = False, 
              plot_wall_clock_times = False,
              plot_total_steps = False, 
              plot_eval_rewards = False)

    elif question_no == 3: # eval rewards for cartpole
           
       plot_stats_nfq(num_instances = 5,
              train_episodes = 2000,
              chosen_env = env1, 
              plot_train_rewards = False, 
              plot_train_times = False, 
              plot_wall_clock_times = False,
              plot_total_steps = False, 
              plot_eval_rewards = True)
       
       plot_stats_dqn(num_instances = 5,
              train_episodes = 2000,
              chosen_env = env1, 
              plot_train_rewards = False, 
              plot_train_times = False, 
              plot_wall_clock_times = False,
              plot_total_steps = False, 
              plot_eval_rewards = True)
       
       plot_stats_ddqn(num_instances = 5,
              train_episodes = 2000,
              chosen_env = env1, 
              plot_train_rewards = False, 
              plot_train_times = False, 
              plot_wall_clock_times = False,
              plot_total_steps = False, 
              plot_eval_rewards = True)
       
       plot_stats_d3qn(num_instances = 5,
              train_episodes = 2000,
              chosen_env = env1, 
              plot_train_rewards = False, 
              plot_train_times = False, 
              plot_wall_clock_times = False,
              plot_total_steps = False, 
              plot_eval_rewards = True)
       
       plot_stats_d3qn_per(num_instances = 5,
              train_episodes = 10000,
              chosen_env = env1, 
              plot_train_rewards = False, 
              plot_train_times = False, 
              plot_wall_clock_times = False,
              plot_total_steps = False, 
              plot_eval_rewards = True)
       
    elif question_no == 4: # eval rewards for mountain car
       
       plot_stats_nfq(num_instances = 5,
              train_episodes = 2000,
              chosen_env = env2, 
              plot_train_rewards = False, 
              plot_train_times = False, 
              plot_wall_clock_times = False,
              plot_total_steps = False, 
              plot_eval_rewards = True)
       
       plot_stats_dqn(num_instances = 5,
              train_episodes = 2000,
              chosen_env = env2, 
              plot_train_rewards = False, 
              plot_train_times = False, 
              plot_wall_clock_times = False,
              plot_total_steps = False, 
              plot_eval_rewards = True)
       
       plot_stats_ddqn(num_instances = 5,
              train_episodes = 2000,
              chosen_env = env2, 
              plot_train_rewards = False, 
              plot_train_times = False, 
              plot_wall_clock_times = False,
              plot_total_steps = False, 
              plot_eval_rewards = True)
       
       plot_stats_d3qn(num_instances = 5,
              train_episodes = 2000,
              chosen_env = env2, 
              plot_train_rewards = False, 
              plot_train_times = False, 
              plot_wall_clock_times = False,
              plot_total_steps = False, 
              plot_eval_rewards = True)
       
       plot_stats_d3qn_per(num_instances = 5,
              train_episodes = 3000,
              chosen_env = env2, 
              plot_train_rewards = False, 
              plot_train_times = False, 
              plot_wall_clock_times = False,
              plot_total_steps = False, 
              plot_eval_rewards = True)
       
    elif question_no == 5: # total steps for cartpole
           
       plot_stats_nfq(num_instances = 5,
              train_episodes = 2000,
              chosen_env = env1, 
              plot_train_rewards = False, 
              plot_train_times = False, 
              plot_wall_clock_times = False,
              plot_total_steps = True, 
              plot_eval_rewards = False)
       
       plot_stats_dqn(num_instances = 5,
              train_episodes = 2000,
              chosen_env = env1, 
              plot_train_rewards = False, 
              plot_train_times = False, 
              plot_wall_clock_times = False,
              plot_total_steps = True, 
              plot_eval_rewards = False)
       
       plot_stats_ddqn(num_instances = 5,
              train_episodes = 2000,
              chosen_env = env1, 
              plot_train_rewards = False, 
              plot_train_times = False, 
              plot_wall_clock_times = False,
              plot_total_steps = True, 
              plot_eval_rewards = False)
       
       plot_stats_d3qn(num_instances = 5,
              train_episodes = 2000,
              chosen_env = env1, 
              plot_train_rewards = False, 
              plot_train_times = False, 
              plot_wall_clock_times = False,
              plot_total_steps = True, 
              plot_eval_rewards = False)
       
       plot_stats_d3qn_per(num_instances = 5,
              train_episodes = 10000,
              chosen_env = env1, 
              plot_train_rewards = False, 
              plot_train_times = False, 
              plot_wall_clock_times = False,
              plot_total_steps = True, 
              plot_eval_rewards = False)

    elif question_no == 6: # total steps for mountain car
       
       plot_stats_nfq(num_instances = 5,
              train_episodes = 2000,
              chosen_env = env2, 
              plot_train_rewards = False, 
              plot_train_times = False, 
              plot_wall_clock_times = False,
              plot_total_steps = True, 
              plot_eval_rewards = False)
       
       plot_stats_dqn(num_instances = 5,
              train_episodes = 2000,
              chosen_env = env2, 
              plot_train_rewards = False, 
              plot_train_times = False, 
              plot_wall_clock_times = False,
              plot_total_steps = True, 
              plot_eval_rewards = False)
       
       plot_stats_ddqn(num_instances = 5,
              train_episodes = 2000,
              chosen_env = env2, 
              plot_train_rewards = False, 
              plot_train_times = False, 
              plot_wall_clock_times = False,
              plot_total_steps = True, 
              plot_eval_rewards = False)
       
       plot_stats_d3qn(num_instances = 5,
              train_episodes = 2000,
              chosen_env = env2, 
              plot_train_rewards = False, 
              plot_train_times = False, 
              plot_wall_clock_times = False,
              plot_total_steps = True, 
              plot_eval_rewards = False)
       
       plot_stats_d3qn_per(num_instances = 5,
              train_episodes = 3000,
              chosen_env = env2, 
              plot_train_rewards = False, 
              plot_train_times = False, 
              plot_wall_clock_times = False,
              plot_total_steps = True, 
              plot_eval_rewards = False)

    elif question_no == 7: # train times for cartpole
           
       plot_stats_nfq(num_instances = 5,
              train_episodes = 2000,
              chosen_env = env1, 
              plot_train_rewards = False, 
              plot_train_times = True, 
              plot_wall_clock_times = False,
              plot_total_steps = False, 
              plot_eval_rewards = False)
       
       plot_stats_dqn(num_instances = 5,
              train_episodes = 2000,
              chosen_env = env1, 
              plot_train_rewards = False, 
              plot_train_times = True, 
              plot_wall_clock_times = False,
              plot_total_steps = False, 
              plot_eval_rewards = False)
       
       plot_stats_ddqn(num_instances = 5,
              train_episodes = 2000,
              chosen_env = env1, 
              plot_train_rewards = False, 
              plot_train_times = True, 
              plot_wall_clock_times = False,
              plot_total_steps = False, 
              plot_eval_rewards = False)
       
       plot_stats_d3qn(num_instances = 5,
              train_episodes = 2000,
              chosen_env = env1, 
              plot_train_rewards = False, 
              plot_train_times = True, 
              plot_wall_clock_times = False,
              plot_total_steps = False, 
              plot_eval_rewards = False)
       
       plot_stats_d3qn_per(num_instances = 5,
              train_episodes = 10000,
              chosen_env = env1, 
              plot_train_rewards = False, 
              plot_train_times = True, 
              plot_wall_clock_times = False,
              plot_total_steps = False, 
              plot_eval_rewards = False)
       
    elif question_no == 8: # train times for mountain car
       
       plot_stats_nfq(num_instances = 5,
              train_episodes = 2000,
              chosen_env = env2, 
              plot_train_rewards = False, 
              plot_train_times = True, 
              plot_wall_clock_times = False,
              plot_total_steps = False, 
              plot_eval_rewards = False)
       
       plot_stats_dqn(num_instances = 5,
              train_episodes = 2000,
              chosen_env = env2, 
              plot_train_rewards = False, 
              plot_train_times = True, 
              plot_wall_clock_times = False,
              plot_total_steps = False, 
              plot_eval_rewards = False)
       
       plot_stats_ddqn(num_instances = 5,
              train_episodes = 2000,
              chosen_env = env2, 
              plot_train_rewards = False, 
              plot_train_times = True, 
              plot_wall_clock_times = False,
              plot_total_steps = False, 
              plot_eval_rewards = False)
       
       plot_stats_d3qn(num_instances = 5,
              train_episodes = 2000,
              chosen_env = env2, 
              plot_train_rewards = False, 
              plot_train_times = True, 
              plot_wall_clock_times = False,
              plot_total_steps = False, 
              plot_eval_rewards = False)
       
       plot_stats_d3qn_per(num_instances = 5,
              train_episodes = 3000,
              chosen_env = env2, 
              plot_train_rewards = False, 
              plot_train_times = True, 
              plot_wall_clock_times = False,
              plot_total_steps = False, 
              plot_eval_rewards = False)

    elif question_no == 9: # wall clock times for cartpole 
       
       plot_stats_nfq(num_instances = 5,
              train_episodes = 2000,
              chosen_env = env1, 
              plot_train_rewards = False, 
              plot_train_times = False, 
              plot_wall_clock_times = True,
              plot_total_steps = False, 
              plot_eval_rewards = False)
       
       plot_stats_dqn(num_instances = 5,
              train_episodes = 2000,
              chosen_env = env1, 
              plot_train_rewards = False, 
              plot_train_times = False, 
              plot_wall_clock_times = True,
              plot_total_steps = False, 
              plot_eval_rewards = False)
       
       plot_stats_ddqn(num_instances = 5,
              train_episodes = 2000,
              chosen_env = env1, 
              plot_train_rewards = False, 
              plot_train_times = False, 
              plot_wall_clock_times = True,
              plot_total_steps = False, 
              plot_eval_rewards = False)
       
       plot_stats_d3qn(num_instances = 5,
              train_episodes = 2000,
              chosen_env = env1, 
              plot_train_rewards = False, 
              plot_train_times = False, 
              plot_wall_clock_times = True,
              plot_total_steps = False, 
              plot_eval_rewards = False)
       
       plot_stats_d3qn_per(num_instances = 5,
              train_episodes = 10000,
              chosen_env = env1, 
              plot_train_rewards = False, 
              plot_train_times = False, 
              plot_wall_clock_times = True,
              plot_total_steps = False, 
              plot_eval_rewards = False)
           
    elif question_no == 10: # wall clock times for mountain car
           
       plot_stats_nfq(num_instances = 5,
              train_episodes = 2000,
              chosen_env = env2, 
              plot_train_rewards = False, 
              plot_train_times = False, 
              plot_wall_clock_times = True,
              plot_total_steps = False, 
              plot_eval_rewards = False)
       
       plot_stats_dqn(num_instances = 5,
              train_episodes = 2000,
              chosen_env = env2, 
              plot_train_rewards = False, 
              plot_train_times = False, 
              plot_wall_clock_times = True,
              plot_total_steps = False, 
              plot_eval_rewards = False)
       
       plot_stats_ddqn(num_instances = 5,
              train_episodes = 2000,
              chosen_env = env2, 
              plot_train_rewards = False, 
              plot_train_times = False, 
              plot_wall_clock_times = True,
              plot_total_steps = False, 
              plot_eval_rewards = False)
       
       plot_stats_d3qn(num_instances = 5,
              train_episodes = 2000,
              chosen_env = env2, 
              plot_train_rewards = False, 
              plot_train_times = False, 
              plot_wall_clock_times = True,
              plot_total_steps = False, 
              plot_eval_rewards = False)
       
       plot_stats_d3qn_per(num_instances = 5,
              train_episodes = 3000,
              chosen_env = env2, 
              plot_train_rewards = False, 
              plot_train_times = False, 
              plot_wall_clock_times = True,
              plot_total_steps = False, 
              plot_eval_rewards = False)
    

## Depending on what you want to plot and for which environment
## put your question_no as an argument

if __name__ == '__main__':
    runDeepValueBasedAgents(3)