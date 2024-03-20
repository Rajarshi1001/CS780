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
This class implements the NFQ Agent, you are required to implement the various methods of this class
as outlined below. Note this class is generic and should work with any permissible Gym environment.
Also please feel free to play with different exploration strategies with decaying paramters (epsilon/temperature)

```
class NFQ():
def __init__(env, seed, gamma, epochs,
                bufferSize,
                batchSize,
                optimizerFn,
                optimizerLR,
                MAX_TRAIN_EPISODES, MAX_EVAL_EPISODES,
                explorationStrategyTrainFn,
                explorationStrategyEvalFn)
def initBookKeeping(self)
def performBookKeeping(self, train = True)
def runNFQ(self)
def trainAgent(self)
def trainNetwork(self, experiences, epochs)
def evaluateAgent(self)
```
"""

class NFQ():
    def __init__(self, env, seed, gamma, epochs,
                 bufferSize,
                 batchSize,
                 optimizerFn,
                 optimizerLR,
                 MAX_TRAIN_EPISODES, MAX_EVAL_EPISODES,
                 explorationStrategyTrainFn,
                 explorationStrategyEvalFn):
        #this NFQ method
        # 1. creates and initializes (with seed) the environment, train/eval episodes, gamma, etc.
        # 2. creates and intializes all the variables required for book-keeping values via the initBookKeeping method
        # 3. creates Q-network using the createValueNetwork above
        # 4. creates and initializes (with network params) the optimizer function
        # 5. sets the explorationStartegy variables/functions for train and evaluation
        # 6. sets the batchSize for the number of experiences
        # 7. Creates the replayBuffer
        self.env = env
        self.seed = seed
        self.gamma = gamma
        self.epochs = epochs
        self.criterion = nn.MSELoss()
        self.bufferSize = bufferSize
        self.batchSize = batchSize
        self.optimizerFn = optimizerFn
        self.optimizerLR = optimizerLR
        self.MAX_TRAIN_EPISODES = MAX_TRAIN_EPISODES
        self.MAX_EVAL_EPISODES = MAX_EVAL_EPISODES
        self.EVAL_FREQUENCY = 1
        self.explorationStrategyTrainFn = explorationStrategyTrainFn
        self.explorationStrategyEvalFn = explorationStrategyEvalFn
        self.initial_temp = 1.0
        self.initial_epsilon = 1.0
        self.decay_rate = 0.08
        self.min_epsilon = 0.2
        self.min_temp = 0.3

        torch.manual_seed(seed)

        # Replay Buffer
        self.replaybuffer = ReplayBuffer(bufferSize, bufferType = 'NFQ')

        # initialize all the variables
        self.initBookKeeping()

        self.q_network = createValueNetwork(env.observation_space.shape[0], env.action_space.n, hDim = [32, 32], activation = F.relu).to(DEVICE)
        if optimizerFn.lower() == 'adam':
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=optimizerLR)

    
    def initBookKeeping(self):
        #this method creates and intializes all the variables required for book-keeping values and it is called
        #init method
        self.trainRewardsList = []
        self.trainTimeList = []
        self.evalRewardsList = []
        self.wallClockTimeList = []
        self.finalEvalRewards = []
        
    def performBookKeeping(self, train = True):
        #this method updates relevant variables for the bookKeeping, this can be called
        #multiple times during training
        #if you want you can print information using this, so it may help to monitor progress and also help to debug
        pass   

    
    def runNFQ(self):
        #this is the main method, it trains the agent, performs bookkeeping while training and finally evaluates
        #the agent and returns the following quantities:
        #1. episode wise mean train rewards
        #2. epsiode wise mean eval rewards
        #2. episode wise trainTime (in seconds): time elapsed during training since the start of the first episode
        #3. episode wise wallClockTime (in seconds): actual time elapsed since the start of training,
        #                               note this will include time for BookKeeping and evaluation
        # Note both trainTime and wallClockTime get accumulated as episodes proceed.


        self.initBookKeeping()

        self.trainRewardsList, self.trainTimeList, self.evalRewardsList, self.wallClockTimeList = self.trainAgent()

        final_evaluation_start_time = time.time()
        self.finalEvalRewards = self.evaluateAgent()  # Ensure this method returns total rewards for the final evaluation
        final_evaluation_time = time.time() - final_evaluation_start_time

        mean_train_rewards = np.mean(self.trainRewardsList) if self.trainRewardsList else 0
        mean_eval_rewards = np.mean(self.finalEvalRewards) if self.finalEvalRewards else 0
        cumulative_train_time = np.cumsum(self.trainTimeList).tolist() if self.trainTimeList else []
        cumulative_wall_clock_time = np.cumsum(self.wallClockTimeList).tolist() if self.wallClockTimeList else []
        cumulative_wall_clock_time.append(cumulative_wall_clock_time[-1] + final_evaluation_time if cumulative_wall_clock_time else final_evaluation_time)

        print(f"Mean Train Rewards: {mean_train_rewards}")
        print(f"Mean Evaluation Rewards: {mean_eval_rewards}")
        print(f"Total Training Time: {cumulative_train_time[-1] if cumulative_train_time else 0}s")
        print(f"Total Wall Clock Time: {cumulative_wall_clock_time[-1] if cumulative_wall_clock_time else 0}s")


        return self.trainRewardsList, self.trainTimeList, self.evalRewardsList, self.wallClockTimeList, self.finalEvalRewards
    
    
    def trainAgent(self):
        #this method collects experiences and trains the NFQ agent and does BookKeeping while training.
        #this calls the trainNetwork() method internally, it also evaluates the agent per episode
        #it trains the agent for MAX_TRAIN_EPISODES

        # trainRewardsList, trainTimeList, evalRewardsList, wallClockTimeList = [], [], [], []
        training_start_time = time.time()
        for episode in range(self.MAX_TRAIN_EPISODES):
          start_time = time.time()
          state = self.env.reset()
          if isinstance(state, tuple):
            state = state[0]
          total_reward = 0
          done = False
          counter = 0
          while not done:
            if self.explorationStrategyTrainFn == 'greedy':
              action = selectGreedyAction(self.q_network, state)
            elif self.explorationStrategyTrainFn == 'softmax':
              temp = decayTemperature(self.initial_temp, self.min_temp, self.decay_rate, episode)
              action = selectSoftMaxAction(self.q_network, state, temp)
            elif self.explorationStrategyTrainFn == 'epsilon greedy':
              epsilon = decayEpsilon(self.initial_epsilon, self.min_epsilon, self.decay_rate, episode)
              action = selectEpsilonGreedyAction(self.q_network, state, epsilon)

            next_state, reward, done, truncated, _ = self.env.step(action)
            self.replaybuffer.store((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            if self.replaybuffer.length() >= self.bufferSize:
              experiences = self.replaybuffer.sample(self.batchSize)
              # print(experiences)
              self.trainNetwork(experiences, self.epochs)

          self.trainRewardsList.append(total_reward)
          self.trainTimeList.append(time.time() - start_time)

          if episode % self.EVAL_FREQUENCY == 0:
              eval_reward = self.evaluateAgent()
              self.evalRewardsList.append(eval_reward)
              self.wallClockTimeList.append(time.time() - training_start_time)

    
    def trainNetwork(self, experiences, epochs):
        # this method trains the value network epoch number of times and is called by the trainAgent function
        # it essentially uses the experiences to calculate target, using the targets it calculates the error, which
        # is further used for calulating the loss. It then uses the optimizer over the loss
        # to update the params of the network by backpropagating through the network
        # this function does not return anything
        # you can try out other loss functions other than MSE like Huber loss, MAE, etc.

        # Unpack experiences
        states, actions, rewards, next_states, dones = list(map(list, zip(*experiences)))

        # print(states)
        # states = np.array(states)
        # print(states)
        states = torch.tensor(states).to(DEVICE)
        actions = torch.LongTensor(actions).to(DEVICE).view(-1, 1)
        rewards = torch.FloatTensor(rewards).to(DEVICE).view(-1, 1)
        next_states = torch.FloatTensor(next_states).to(DEVICE)
        dones = torch.IntTensor(dones).to(DEVICE).view(-1, 1)

        # print(states)
        print("Q Network training Started")
        for epoch in range(epochs):

          q_values = self.q_network(states)
          q_value = q_values.gather(1, actions)
          max_next_q_value = self.q_network(next_states).detach().max(1)[0].unsqueeze(1)
          expected_q_value = rewards + self.gamma * max_next_q_value * (1 - dones)

          # Compute loss
          loss = F.mse_loss(q_value, expected_q_value)
          self.optimizer.zero_grad()
          loss.backward()
          self.optimizer.step()

          print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')
          
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

          while not done:
            action = selectGreedyAction(self.q_network, state)
            next_state, reward, done, truncated, _ = self.env.step(action)
            total_reward += reward
            state = next_state

          evalRewards.append(total_reward)

        return evalRewards
    


# Executing the code
# Number of environment instances you want to run
num_instances = 3

# Dictionary to store rewards, training times, and wall clock times for all instances
all_train_rewards = {}
all_train_times = {}
all_wall_clock_times = {}
all_eval_rewards = {}

for instance in range(num_instances):
    # Initialize NFQ for the current instance
    instance_seed = 123 + instance
    nfq_network = NFQ(env1,
                      seed = instance_seed,
                      gamma = 0.99,
                      epochs = 15,
                      bufferSize = 32,
                      batchSize = 32,
                      optimizerFn = 'adam',
                      optimizerLR = 0.0004,
                      MAX_TRAIN_EPISODES = 80,
                      MAX_EVAL_EPISODES = 1,
                      explorationStrategyTrainFn = 'epsilon greedy',
                      explorationStrategyEvalFn = 'greedy')

    train_rewards, train_times, eval_rewards, wall_clock_times, _ = nfq_network.runNFQ()
    all_train_rewards[f'Instance {instance + 1}'] = train_rewards
    all_train_times[f'Instance {instance + 1}'] = train_times
    all_wall_clock_times[f'Instance {instance + 1}'] = wall_clock_times
    all_eval_rewards[f'Instance {instance + 1}'] = eval_rewards

plotQuantity(all_train_rewards, nfq_network.MAX_TRAIN_EPISODES, ['Training Rewards', 'Episodes', 'Rewards'], "nfq_train_rewards.png")
plotQuantity(all_train_times, nfq_network.MAX_TRAIN_EPISODES, ['Training Times', 'Episodes', 'Time (s)'], "nfq_train_times.png")
plotQuantity(all_wall_clock_times, nfq_network.MAX_TRAIN_EPISODES, ['Wall Clock Times', 'Episodes', 'Time (s)'], "nfq_wall_clock_times.png")
plotQuantity(all_eval_rewards, nfq_network.MAX_TRAIN_EPISODES, ['Eval Rewards', 'Episodes', 'Time (s)'], "nfq_eval_rewards.png")