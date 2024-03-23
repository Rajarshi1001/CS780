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
This class implements the Dueling Double DQN agent, you are required to implement the various methods of this class
as outlined below. Note this class is generic and should work with any permissible Gym environment

```
class D3QN():
    def __init__(env, seed, gamma, tau,
                 bufferSize,
                 batchSize,
                 optimizerFn,
                 optimizerLR,
                 MAX_TRAIN_EPISODES, MAX_EVAL_EPISODES,
                 explorationStrategyTrainFn,
                 explorationStrategyEvalFn,
                 updateFrequency)
    def initBookKeeping(self)
    def performBookKeeping(self, train = True)
    def runD3QN(self)
    def trainAgent(self)
    def trainNetwork(self, experiences)
    def updateNetwork(self, onlineNet, targetNet)
    def evaluateAgent(self)
```
"""

class D3QN():
    def __init__(self,
                 env, seed, gamma, tau,
                 epochs,
                 bufferSize,
                 batchSize,
                 optimizerFn,
                 optimizerLR,
                 MAX_TRAIN_EPISODES, MAX_EVAL_EPISODES,
                 explorationStrategyTrainFn,
                 explorationStrategyEvalFn,
                 updateFrequency):
        #this D3QN method
        # 1. creates and initializes (with seed) the environment, train/eval episodes, gamma, etc.
        # 2. creates and intializes all the variables required for book-keeping values via the initBookKeeping method
        # 3. creates tareget and online Q-networks using the createValueNetwork above
        # 4. creates and initializes (with network params) the optimizer function
        # 5. sets the explorationStartegy variables/functions for train and evaluation
        # 6. sets the batchSize for the number of experiences
        # 7. Creates the replayBuffer

        self.env = env
        self.seed = seed
        self.gamma = gamma
        self.tau = tau
        self.epochs = epochs
        self.bufferSize = bufferSize
        self.batchSize = batchSize
        self.optimizerFn = optimizerFn
        self.optimizerLR = optimizerLR
        self.MAX_TRAIN_EPISODES = MAX_TRAIN_EPISODES
        self.MAX_EVAL_EPISODES = MAX_EVAL_EPISODES
        self.explorationStrategyTrainFn = explorationStrategyTrainFn
        self.explorationStrategyEvalFn = explorationStrategyEvalFn
        self.updateFrequency = updateFrequency
        self.initial_temp = 1.0
        self.initial_epsilon = 1.0
        self.decay_rate = 0.08
        self.min_epsilon = 0.2
        self.min_temp = 0.3
        self.iteration = 0

        torch.manual_seed(seed)

        # Replay Buffer
        self.replaybuffer = ReplayBuffer(bufferSize, bufferType = "D3QN")

        # initialize all the variables
        self.initBookKeeping()

        self.online_network = createDuelingNetwork(env.observation_space.shape[0], env.action_space.n, hDim = [256, 128], activation = F.relu).to(DEVICE)
        self.target_network = createDuelingNetwork(env.observation_space.shape[0], env.action_space.n, hDim = [256, 128], activation = F.relu).to(DEVICE)
        self.updateNetwork(self.target_network, self.online_network)
        self.target_network.eval() # target network is fixed for multiple steps

        if optimizerFn.lower() == "rmsprop":
            self.optimizer = optim.RMSprop(self.online_network.parameters(), lr=optimizerLR)
        elif optimizerFn.lower() == "adam":
            self.optimizer = optim.Adam(self.online_network.parameters(), lr=optimizerLR)
            

class D3QN(D3QN):
    def initBookKeeping(self):
        #this method creates and intializes all the variables required for book-keeping values and it is called
        #init method
        self.trainRewardsList = []
        self.trainTimeList = []
        self.evalRewardsList = []
        self.wallClockTimeList = []
        self.finalEvalRewards = []
        self.totalStepsList = []
        
class D3QN(D3QN):
    def performBookKeeping(self, train = True):
        #this method updates relevant variables for the bookKeeping, this can be called
        #multiple times during training
        #if you want you can print information using this, so it may help to monitor progress and also help to debug
        pass
    
class D3QN(D3QN):
    def runD3QN(self):
        #this is the main method, it trains the agent, performs bookkeeping while training and finally evaluates
        #the agent and returns the following quantities:
        #1. episode wise mean train rewards
        #2. epsiode wise mean eval rewards
        #2. episode wise trainTime (in seconds): time elapsed during training since the start of the first episode
        #3. episode wise wallClockTime (in seconds): actual time elapsed since the start of training,
        #                               note this will include time for BookKeeping and evaluation
        # Note both trainTime and wallClockTime get accumulated as episodes proceed.
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

        totalEpisodeCount = len(self.trainRewardsList)

        return self.trainRewardsList, self.trainTimeList, self.evalRewardsList, self.wallClockTimeList, self.finalEvalRewards, self.totalStepsList
    
class D3QN(D3QN):
    def trainAgent(self):
        #this method collects experiences and trains the agent and does BookKeeping while training.
        #this calls the trainNetwork() method internally, it also evaluates the agent per episode
        #it trains the agent for MAX_TRAIN_EPISODES

        training_start_time = time.time()
        total_steps = 0
        
        for episode in range(self.MAX_TRAIN_EPISODES):
          start_time = time.time()
          state = self.env.reset()

          # checking inconsistencies in the state dtype
          if isinstance(state, tuple):
            state = state[0]

          total_reward = 0
          done = False
          counter = 0
          steps_per_episode = 0
          
          while not done:
            if self.explorationStrategyTrainFn == 'greedy':
              action = selectGreedyAction(self.online_network, state)
            elif self.explorationStrategyTrainFn == 'softmax':
              temp = decayTemperature(self.initial_temp, self.min_temp, self.decay_rate, episode)
              action = selectSoftMaxAction(self.online_network, state, temp)
            elif self.explorationStrategyTrainFn == 'epsilon greedy':
              epsilon = decayEpsilon(self.initial_epsilon, self.min_epsilon, self.decay_rate, episode)
              action = selectEpsilonGreedyAction(self.online_network, state, epsilon)

            next_state, reward, done, truncated, _ = self.env.step(action)
            self.replaybuffer.store((state, action, reward, next_state, done))
            
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
            
            state = next_state
            total_reward += reward
            steps_per_episode += 1
            total_steps += 1

          if self.bufferSize >= self.batchSize:
            experiences = self.replaybuffer.sample(self.batchSize)
            self.trainNetwork(experiences, self.epochs)

          self.trainRewardsList.append(total_reward)
          self.trainTimeList.append(time.time() - start_time)
          self.totalStepsList.append(total_steps)


          if episode % self.updateFrequency == 0:
            self.updateNetwork(self.online_network, self.target_network)
            eval_reward = self.evaluateAgent()
            self.evalRewardsList.append(eval_reward)
            self.wallClockTimeList.append(time.time() - training_start_time)

        return self.trainRewardsList, self.trainTimeList, self.evalRewardsList, self.wallClockTimeList, self.totalStepsList 



class D3QN(D3QN):
    def trainNetwork(self, experiences, epochs):
        # this method trains the value network epoch number of times and is called by the trainAgent function
        # it essentially uses the experiences to calculate target, using the targets it calculates the error, which
        # is further used for calulating the loss. It then uses the optimizer over the loss
        # to update the params of the network by backpropagating through the network
        # this function does not return anything
        # you can try out other loss functions other than MSE like Huber loss, MAE, etc.
        states, actions, rewards, next_states, dones = list(map(list, zip(*experiences)))

        states = torch.tensor(np.array(states)).float().to(DEVICE)
        next_states = torch.tensor(np.array(next_states)).float().to(DEVICE)
        actions = torch.LongTensor(actions).to(DEVICE).view(-1, 1)
        rewards = torch.FloatTensor(rewards).to(DEVICE).view(-1, 1)
        dones = torch.IntTensor(dones).to(DEVICE).view(-1, 1)

        # print("Q Network training Started, Iteration: {}".format(self.iteration))
        for epoch in range(epochs):

          current_q_value = self.online_network(states).gather(1, actions)
          next_actions = self.online_network(next_states).max(1)[1].unsqueeze(1)
          next_q_values = self.target_network(next_states).gather(1, next_actions).detach()
          expected_q_value = rewards + (self.gamma * next_q_values * (1 - dones))
          loss = F.mse_loss(current_q_value, expected_q_value)
          self.optimizer.zero_grad()
          loss.backward()
          self.optimizer.step()

          # print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

        self.iteration += 1
        

class D3QN(D3QN):
    def updateNetwork(self, onlineNet, targetNet):
        #this function updates the onlineNetwork with the target network using Polyak averaging
        for target_param, online_param in zip(targetNet.parameters(), onlineNet.parameters()):
            target_param.data.copy_(self.tau * online_param.data + (1.0 - self.tau) * target_param.data)



class D3QN(D3QN):
    def evaluateAgent(self):
        #this function evaluates the agent using the value network, it evaluates agent for MAX_EVAL_EPISODES
        #typcially MAX_EVAL_EPISODES = 1
        evalRewards = []
        for episode in range(self.MAX_EVAL_EPISODES):
          state = self.env.reset()
          if isinstance(state, tuple):
            state = state[0]

          total_reward = 0
          done = False
          counter = 0

          while not done:
            action = selectGreedyAction(self.target_network, state)
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
              reward = reward + (self.env.goal_position + position) ** 2
              if position >= 0.5:
                  done = True
              if counter > 200:
                  done = True
              counter += 1
              
            total_reward += reward
            state = next_state

          evalRewards.append(total_reward)

        return evalRewards
    

def plot_stats_d3qn(num_instances = 5,
                    chosen_env = env1, 
                    train_episodes = 100,
                    plot_train_rewards = True, 
                    plot_train_times = True, 
                    plot_wall_clock_times = True,
                    plot_total_steps = True, 
                    plot_eval_rewards = True):

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
        # Initialize D3QN for the current instance for cartpole environment
        instance_seed = 123 + instance
        d3qn_network = D3QN(chosen_env,
                        seed = instance_seed,
                        gamma = 0.99,
                        tau = 0.2,
                        epochs = 20,
                        bufferSize = 32,
                        batchSize = 16,
                        optimizerFn = "adam",
                        optimizerLR = 0.0005,
                        MAX_TRAIN_EPISODES = train_episodes,
                        MAX_EVAL_EPISODES = 1,
                        explorationStrategyTrainFn = 'epsilon greedy',
                        explorationStrategyEvalFn = 'greedy',
                        updateFrequency = 1)

        train_rewards, train_times, eval_rewards, wall_clock_times, _, total_steps = d3qn_network.runD3QN()
        all_train_rewards[f'Instance {instance + 1}'] = train_rewards
        all_train_times[f'Instance {instance + 1}'] = train_times
        all_wall_clock_times[f'Instance {instance + 1}'] = wall_clock_times
        all_eval_rewards[f'Instance {instance + 1}'] = eval_rewards
        all_total_steps[f'Instance {instance + 1}'] = total_steps

    
    if plot_train_rewards:
        plotQuantity(all_train_rewards, d3qn_network.MAX_TRAIN_EPISODES, ['Training Rewards {} D3QN'.format(env_name), 'Episodes', 'Rewards'], "images/d3qn_train_rewards_{}.png".format(env_name))
        
    if plot_train_times:
        plotQuantity(all_train_times, d3qn_network.MAX_TRAIN_EPISODES, ['Training Times {} D3QN'.format(env_name), 'Episodes', 'Time (s)'], "images/d3qn_train_times_{}.png".format(env_name))
    
    if plot_total_steps:
        plotQuantity(all_total_steps, d3qn_network.MAX_TRAIN_EPISODES, ['Total Steps {} D3QN'.format(env_name), 'Episodes', 'Time (s)'], "images/d3qn_total_steps_{}.png".format(env_name))
    
    if plot_eval_rewards:
        plotQuantity(all_eval_rewards, d3qn_network.MAX_TRAIN_EPISODES // d3qn_network.MAX_EVAL_EPISODES, ['Eval Rewards {} D3QN'.format(env_name), 'Episodes', 'Time (s)'], "images/d3qn_eval_rewards_{}.png".format(env_name))
    
    if plot_wall_clock_times:
        plotQuantity(all_wall_clock_times, d3qn_network.MAX_TRAIN_EPISODES // d3qn_network.MAX_EVAL_EPISODES, ['Wall Clock Times {} D3QN'.format(env_name), 'Episodes', 'Time (s)'], "images/d3qn_wall_clock_times_{}.png".format(env_name))
        
    
# plot_stats_d3qn(num_instances = 5,
#                   train_episodes = 10,
#                     chosen_env = env2, 
#                     plot_train_rewards = True, 
#                     plot_train_times = True, 
#                     plot_wall_clock_times = True,
#                     plot_total_steps = True, 
#                     plot_eval_rewards = True)

