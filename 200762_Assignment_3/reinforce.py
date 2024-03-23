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



# Use either Cart pole or Mountain Car Environment
env1 = gym.make('CartPole-v0')
env2 = gym.make('MountainCar-v0')

"""
This class implements the REINFORCE Agent, you are required to implement the various methods of this class
as outlined below. Note this class is generic and should work with any permissible Gym environment

```
class REINFORCE():
    def __init__(env, seed, gamma,
                 optimizerFn,
                 optimizerLR,
                 MAX_TRAIN_EPISODES, MAX_EVAL_EPISODES,
                 explorationStrategyTrainFn,
                 explorationStrategyEvalFn)
    def initBookKeeping(self)
    def performBookKeeping(self, train = True)
    def runREINFORCE(self)
    def trainAgent(self)
    def trainPolicyNetwork(self, experiences)
    def evaluateAgent(self)
```    
"""


class REINFORCE():
    def __init__(self, env, seed, gamma,
                 optimizerFn,
                 optimizerLR,
                 MAX_TRAIN_EPISODES,
                 MAX_EVAL_EPISODES,
                 explorationStrategyTrainFn,
                 explorationStrategyEvalFn):

        self.seed = seed
        self.env = env
        self.gamma = gamma
        self.optimizerFn = optimizerFn
        self.optimizerLR = optimizerLR
        self.MAX_TRAIN_EPISODES = MAX_TRAIN_EPISODES
        self.MAX_EVAL_EPISODES = MAX_EVAL_EPISODES
        self.explorationStrategyTrainFn = explorationStrategyTrainFn
        self.explorationStrategyEvalFn = explorationStrategyEvalFn
        self.initial_temp = 1.0
        self.initial_epsilon = 1.0
        self.decay_rate = 0.08
        self.min_epsilon = 0.2
        self.min_temp = 0.3
        self.iteration = 0
        self.EVAL_FREQUENCY = 1

        torch.manual_seed(seed)

        self.initBookKeeping()

        self.policy_network = createPolicyNetwork(env.observation_space.shape[0], env.action_space.n, hDim = [512, 128], activation = F.relu).to(DEVICE)


        if optimizerFn.lower() == "rmsprop":
            self.optimizer = optim.RMSprop(self.policy_network.parameters(), lr=optimizerLR)
        elif optimizerFn.lower() == "adam":
            self.optimizer = optim.Adam(self.policy_network.parameters(), lr=optimizerLR)


class REINFORCE(REINFORCE):
    def initBookKeeping(self):
        #this method creates and intializes all the variables required for book-keeping values and it is called
        #init method
        self.trainRewardsList = []
        self.trainTimeList = []
        self.evalRewardsList = []
        self.wallClockTimeList = []
        self.finalEvalRewards = []
        self.totalStepsList = []

class REINFORCE(REINFORCE):
    def performBookKeeping(self, train = True):
        #this method updates relevant variables for the bookKeeping, this can be called
        #multiple times during training
        #if you want you can print information using this, so it may help to monitor progress and also help to debug
        pass
    
    
class REINFORCE(REINFORCE):
    def runREINFORCE(self):

        self.initBookKeeping()

        self.trainRewardsList, self.trainTimeList, self.evalRewardsList, self.wallClockTimeList, self.totalStepsList = self.trainAgent()

        final_evaluation_start_time = time.time()
        self.finalEvalRewards = self.evaluateAgent()
        final_evaluation_time = time.time() - final_evaluation_start_time

        mean_train_rewards = np.mean(self.trainRewardsList) if self.trainRewardsList else 0
        mean_eval_rewards = np.mean(self.finalEvalRewards) if self.finalEvalRewards else 0
        cumulative_train_time = np.cumsum(self.trainTimeList).tolist() if self.trainTimeList else []
        cumulative_wall_clock_time = np.cumsum(self.wallClockTimeList).tolist() if self.wallClockTimeList else []
        cumulative_wall_clock_time.append(cumulative_wall_clock_time[-1] + final_evaluation_time if cumulative_wall_clock_time else final_evaluation_time)

        # print(f"Mean Train Rewards: {mean_train_rewards}")
        # print(f"Mean Evaluation Rewards: {mean_eval_rewards}")
        # print(f"Total Training Time: {cumulative_train_time[-1] if cumulative_train_time else 0}s")
        # print(f"Total Wall Clock Time: {cumulative_wall_clock_time[-1] if cumulative_wall_clock_time else 0}s")


        return self.trainRewardsList, self.trainTimeList, self.evalRewardsList, self.wallClockTimeList, self.finalEvalRewards, self.totalStepsList
    
    
class REINFORCE(REINFORCE):
    def trainAgent(self):

      training_start_time = time.time()
      total_steps = 0
      
      for episode in range(self.MAX_TRAIN_EPISODES):
        
        start_time = time.time()
        state = self.env.reset()
        
        if isinstance(state, tuple):
          state = state[0]
          
        state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
        total_reward = 0
        rewards, log_probs = [], []
        done = False
        counter = 0
        steps_per_episode = 0

        while not done:

          probs = self.policy_network(state)
          m = torch.distributions.Categorical(probs)
          action = m.sample()

          next_state, reward, done, truncated, _ = self.env.step(action.item())
                
          # Check for termination conditions
          if 'CartPole' in str(self.env):
            cart_pos, _, pole_angle, _ = next_state
            if abs(cart_pos) > 2.4 or abs(pole_angle) > (12 * (3.14 / 180)):
                done = True
            if counter > 500:
                  done = True
            counter += 1

          elif 'MountainCar' in str(self.env):
            position, velocity = next_state
            reward = reward + (self.env.goal_position + position) ** 2 # reward shaping
            if position >= 0.5:
                done = True
            if counter > 200:
                done = True
            counter += 1
              
          total_reward += reward
          total_steps +=1 
          steps_per_episode += 1
          
          rewards.append(reward)
          log_probs.append(m.log_prob(action))
          state = torch.from_numpy(next_state).float().unsqueeze(0).to(DEVICE)

        self.trainPolicyNetwork(rewards, log_probs)
        self.trainRewardsList.append(total_reward)
        self.trainTimeList.append(time.time() - start_time)
        self.totalStepsList.append(total_steps)

        if (episode + 1) % self.EVAL_FREQUENCY == 0:
            eval_reward = self.evaluateAgent()
            self.evalRewardsList.append(eval_reward)
            self.wallClockTimeList.append(time.time() - training_start_time)

      return self.trainRewardsList, self.trainTimeList, self.evalRewardsList, self.wallClockTimeList, self.totalStepsList
  

class REINFORCE(REINFORCE):
    def trainPolicyNetwork(self, rewards, log_probs):

        # print("Training policy network")

        R = 0
        rewards = list(reversed(rewards))
        rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)

        discounted_rewards = []
        for r in rewards:
            R = r + self.gamma * R
            discounted_rewards.insert(0, R)

        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(DEVICE)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-6)
        log_probs = torch.stack(log_probs).to(DEVICE)

        # policy gradient loss
        loss = -torch.sum(log_probs * discounted_rewards.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



class REINFORCE(REINFORCE):
    def evaluateAgent(self):

        evalRewards = []

        for _ in range(self.MAX_EVAL_EPISODES):

            state = self.env.reset()

            if isinstance(state, tuple):
                state = state[0]

            state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)

            done = False
            total_rewards = 0
            counter = 0

            while not done:
                with torch.no_grad():
                    probs = self.policy_network(state)
                action = torch.argmax(probs).item()
                next_state, reward, done, truncated, _ = self.env.step(action)
                
                
                # Check for termination conditions
                if 'CartPole' in str(self.env):
                    cart_pos, _, pole_angle, _ = next_state
                    if abs(cart_pos) > 2.4 or abs(pole_angle) > (12 * (3.14 / 180)):
                        done = True
                    if counter > 500:
                            done = True
                    counter += 1

                elif 'MountainCar' in str(self.env):
                    position, velocity = next_state
                    reward = reward + (self.env.goal_position + position) ** 2 # reward shaping
                    if position >= 0.5:
                        done = True
                    if counter > 200:
                        done = True
                    counter += 1
                
                state = torch.from_numpy(next_state).float().unsqueeze(0).to(DEVICE)
                total_rewards += reward

            evalRewards.append(total_rewards)

        return evalRewards
    

def plot_stats_reinforce(num_instances = 5,
               train_episodes = 2000,
                chosen_env = env1, 
                plot_train_rewards = True, 
                plot_train_times = False, 
                plot_wall_clock_times = False,
                plot_total_steps = False, 
                plot_eval_rewards = False):
    
    # Number of environment instances you want to run
    num_instances = num_instances
    env_name = chosen_env.spec.id

    # Dictionary to store rewards, training times, and wall clock times for all instances
    all_train_rewards = {}
    all_train_times = {}
    all_wall_clock_times = {}
    all_eval_rewards = {}
    all_total_steps = {}

    for instance in range(num_instances):
        # Initialize NFQ for the current instance
        instance_seed = 123 + instance
        reinforce = REINFORCE(chosen_env,
                        seed = instance_seed,
                        gamma = 0.99,
                        optimizerFn = 'adam',
                        optimizerLR = 0.0004,
                        MAX_TRAIN_EPISODES = train_episodes,
                        MAX_EVAL_EPISODES = 1,
                        explorationStrategyTrainFn = 'epsilon greedy',
                        explorationStrategyEvalFn = 'greedy')

        train_rewards, train_times, eval_rewards, wall_clock_times, _, total_steps = reinforce.runREINFORCE()
        all_train_rewards[f'Instance {instance + 1}'] = train_rewards
        all_train_times[f'Instance {instance + 1}'] = train_times
        all_wall_clock_times[f'Instance {instance + 1}'] = wall_clock_times
        all_eval_rewards[f'Instance {instance + 1}'] = eval_rewards
        all_total_steps[f'Instance {instance + 1}'] = total_steps
    
    if plot_train_rewards:
        plotQuantity(all_train_rewards, reinforce.MAX_TRAIN_EPISODES, ['Training Rewards {} REINFORCE'.format(env_name), 'Episodes', 'Rewards'], "reinforce_train_rewards_{}.png".format(env_name))
    
    if plot_train_times:
        plotQuantity(all_train_times, reinforce.MAX_TRAIN_EPISODES, ['Training Times {} REINFORCE'.format(env_name), 'Episodes', 'Time (s)'], "reinforce_train_times_{}.png".format(env_name))
    
    if plot_eval_rewards:
        plotQuantity(all_eval_rewards, reinforce.MAX_TRAIN_EPISODES // reinforce.MAX_EVAL_EPISODES, ['Eval Rewards {} REINFORCE'.format(env_name), 'Episodes', 'Time (s)'], "reinforce_eval_rewards_{}.png".format(env_name))
    
    if plot_wall_clock_times:
        plotQuantity(all_wall_clock_times, reinforce.MAX_TRAIN_EPISODES // reinforce.MAX_EVAL_EPISODES, ['Wall Clock Times {} REINFORCE', 'Episodes', 'Time (s)'], "reinforce_wall_clock_times_{}.png".format(env_name))
    
    if plot_total_steps:
        plotQuantity(all_total_steps, reinforce.MAX_TRAIN_EPISODES, ['Total Steps {} REINFORCE', 'Episodes' ,'Times (s)'], "reinforce_total_steps_{}.png".format(env_name))
        

# plot_stats_reinforce(num_instances = 5,
#                train_episodes = 20,
#                 chosen_env = env1, 
#                 plot_train_rewards = True, 
#                 plot_train_times = False, 
#                 plot_wall_clock_times = False,
#                 plot_total_steps = False, 
#                 plot_eval_rewards = False)

    



