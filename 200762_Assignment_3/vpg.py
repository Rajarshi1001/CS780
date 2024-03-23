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
This class implements the VPG Agent, you are required to implement the various methods of this class
as outlined below. Note this class is generic and should work with any permissible Gym environment

```
class VPG():
    def __init__(env, seed, gamma, beta,
                 optimizerFn,
                 optimizerLR,
                 MAX_TRAIN_EPISODES, MAX_EVAL_EPISODES,
                 explorationStrategyTrainFn,
                 explorationStrategyEvalFn)
    def initBookKeeping(self)
    def performBookKeeping(self, train = True)
    def runVPG(self)
    def trainAgent(self)
    def trainPolicyNetwork(self, experiences)
    def evaluateAgent(self)
"""



class VPG():
    def __init__(self, env, seed,
                 gamma,
                 beta,
                 optimizerFn,
                 optimizerLR,
                 MAX_TRAIN_EPISODES, MAX_EVAL_EPISODES,
                 explorationStrategyTrainFn,
                 explorationStrategyEvalFn):
        self.seed = seed
        self.env = env
        self.gamma = gamma
        self.beta = beta
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

        self.policy_network = createPolicyNetwork(env.observation_space.shape[0], env.action_space.n, hDim = [128, 64], activation = F.relu).to(DEVICE)
        self.value_network = createValueNetwork(env.observation_space.shape[0], 1, hDim = [128, 64], activation = F.relu).to(DEVICE)

        if optimizerFn.lower() == "rmsprop":
            self.policy_optimizer = optim.RMSprop(self.policy_network.parameters(), lr=optimizerLR[0])
            self.value_optimizer = optim.RMSprop(self.value_network.parameters(), lr=optimizerLR[1])
        elif optimizerFn.lower() == "adam":
            self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=optimizerLR[0])
            self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=optimizerLR[1])
                
class VPG(VPG):
    def initBookKeeping(self):
        #this method creates and intializes all the variables required for book-keeping values and it is called
        #init method
        self.trainRewardsList = []
        self.trainTimeList = []
        self.evalRewardsList = []
        self.wallClockTimeList = []
        self.finalEvalRewards = []
        self.totalStepsList = []
        
        
class VPG(VPG):
    def performBookKeeping(self, train = True):
        #this method updates relevant variables for the bookKeeping, this can be called
        #multiple times during training
        #if you want you can print information using this, so it may help to monitor progress and also help to debug
        pass
    
    
class VPG(VPG):
    def runVPG(self):

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
    
    
class VPG(VPG):
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
        rewards, log_probs, values, entropies = [], [], [], []
        done = False
        counter = 0
        steps_per_episode = 0

        while not done:

          probs = self.policy_network(state)
          m = torch.distributions.Categorical(probs)
          action = m.sample()
          log_prob = m.log_prob(action)
          entropy = m.entropy()

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
            
          # calculating the value using a value network
          value = self.value_network(state)

          total_reward += reward
          total_steps += 1
          steps_per_episode += 1
          
          rewards.append(reward)
          log_probs.append(log_prob)
          values.append(value)
          entropies.append(entropy)
          state = torch.from_numpy(next_state).float().unsqueeze(0).to(DEVICE)

        self.trainValueAndPolicyNetwork(rewards, log_probs, values, entropies)
        self.trainRewardsList.append(total_reward)
        self.trainTimeList.append(time.time() - start_time)
        self.totalStepsList.append(total_steps)

        if (episode + 1) % self.EVAL_FREQUENCY == 0:
            eval_reward = self.evaluateAgent()
            self.evalRewardsList.append(eval_reward)
            self.wallClockTimeList.append(time.time() - training_start_time)

      return self.trainRewardsList, self.trainTimeList, self.evalRewardsList, self.wallClockTimeList, self.totalStepsList  


class VPG(VPG):
    def trainValueAndPolicyNetwork(self, rewards, log_probs, values, entropies):

        # # print("Training policy network")

        values = torch.cat(values).squeeze().to(DEVICE)
        entropies = torch.cat(entropies).to(DEVICE)
        log_probs = torch.cat(log_probs).to(DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
        discounted_rewards = []

        R = 0
        for r in rewards.flip(dims=[0]):
            R = r + self.gamma * R
            discounted_rewards.insert(0, R)

        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(DEVICE)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-6)
        advantages = discounted_rewards.unsqueeze(1) - values

        # Calculate policy loss including entropy term
        policy_loss = -(log_probs * advantages.detach()).mean() - (self.beta * entropies.mean())

        # Calculate value loss
        value_loss = F.mse_loss(values.unsqueeze(1), discounted_rewards.unsqueeze(1))

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()   
    

class VPG(VPG):
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
    


def plot_stats_vpg(num_instances = 5,
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
        # Initialize VPG for the current instance
        instance_seed = 123 + instance
        vpg =           VPG(chosen_env,
                        seed = instance_seed,
                        gamma = 0.99,
                        beta = 0.5,
                        optimizerFn = 'adam',
                        optimizerLR = [0.0004, 0.0006],
                        MAX_TRAIN_EPISODES = train_episodes,
                        MAX_EVAL_EPISODES = 1,
                        explorationStrategyTrainFn = 'softmax',
                        explorationStrategyEvalFn = 'greedy')

        train_rewards, train_times, eval_rewards, wall_clock_times, _, total_steps = vpg.runVPG()
        all_train_rewards[f'Instance {instance + 1}'] = train_rewards
        all_train_times[f'Instance {instance + 1}'] = train_times
        all_wall_clock_times[f'Instance {instance + 1}'] = wall_clock_times
        all_eval_rewards[f'Instance {instance + 1}'] = eval_rewards
        all_total_steps[f'Instance {instance + 1}'] = total_steps
        
    if plot_train_rewards:
        plotQuantity(all_train_rewards, vpg.MAX_TRAIN_EPISODES, ['Training Rewards {} VPG'.format(env_name), 'Episodes', 'Rewards'], "vpg_train_rewards_{}.png".format(env_name))
        
    if plot_train_times:
        plotQuantity(all_train_times, vpg.MAX_TRAIN_EPISODES, ['Training Times {} VPG'.format(env_name), 'Episodes', 'Time (s)'], "vpg_train_times_{}.png".format(env_name))

    if plot_eval_rewards:
        plotQuantity(all_eval_rewards, vpg.MAX_TRAIN_EPISODES // vpg.MAX_EVAL_EPISODES, ['Eval Rewards {} VPG'.format(env_name), 'Episodes', 'Time (s)'], "vpg_eval_rewards_{}.png".format(env_name))

    if plot_wall_clock_times:
        plotQuantity(all_wall_clock_times, vpg.MAX_TRAIN_EPISODES // vpg.MAX_EVAL_EPISODES, ['Wall Clock Times {} VPG'.format(env_name), 'Episodes', 'Time (s)'], "vpg_wall_clock_times_{}.png".format(env_name))

    if plot_total_steps:
        plotQuantity(all_total_steps, vpg.MAX_TRAIN_EPISODES, ['Total Steps {} VPG'.format(env_name), 'Episodes', 'Time (s)'], "vpg_total_steps_{}.png".format(env_name))


# plot_stats_vpg(num_instances = 1,
#                train_episodes = 500,
#                 chosen_env = env1, 
#                 plot_train_rewards = True, 
#                 plot_train_times = False, 
#                 plot_wall_clock_times = False,
#                 plot_total_steps = False, 
#                 plot_eval_rewards = False)
    
    