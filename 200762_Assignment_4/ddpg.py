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
from helper import *
from replay_buffer import *


"""
    This class implements the DDPG agent, you are required to implement the various methods of this class
as outlined below. Note this class is generic and should work with any permissible Gym environment

```
class DDPG():
    def init(self, env, seed, gamma, tau, bufferSize, batch_size, updateFrequency,
             policyOptimizerFn, valueOptimizerFn,
             policyOptimizerLR,valueOptimizerLR,
             MAX_TRAIN_EPISODES,MAX_EVAL_EPISODE,
             optimizerFn)
    
    def runDDPG(self)
    def trainAgent(self)
    def gaussianStrategy(self, net , s , envActionRange , noiseScaleRatio,
        explorationMax = True)
    def greedyStrategy(self, net , s , envActionRange)
    def trainNetworks(self, experiences)
    def updateNetworks(self, onlineNet, targetNet, tau)
    def evaluateAgent(self)

"""

class DDPG():
    def __init__(self, env_id, seed, gamma,
                 tau,
                 epochs,
                 bufferSize,
                 batchSize,
                 updateFrequency,
                 valueOptimizerFn,
                 valueOptimizerLR,
                 policyOptimizerFn,
                 policyOptimizerLR,
                 MAX_TRAIN_EPISODES, MAX_EVAL_EPISODES):
        #this DQN method
        # 1. creates and initializes (with seed) the environment, train/eval episodes, gamma, etc.
        # 2. creates and intializes all the variables required for book-keeping values via the initBookKeeping method
        # 3. creates traget and online Q-networks using the createValueNetwork above
        # 4. creates and initializes (with network params) the optimizer function
        # 5. sets the explorationStartegy variables/functions for train and evaluation
        # 6. sets the batchSize for the number of experiences
        # 7. Creates the replayBuffer

        # 1. Environment initialization using env_id

        self.env = gym.make(env_id)
        self.seed = seed
        reset_env(self.env, seed=seed)  # Seed the environment (affects environment operations)

        # Note on seeding:
        # Seeding affects global state for NumPy and PyTorch, which might impact other parts of an application.
        # It's set here for reproducibility within the scope of this NFQ instance's operations.
        torch.manual_seed(seed)  # Seed PyTorch (global effect)
        np.random.seed(seed)  # Seed NumPy (global effect)
        random.seed(seed)

        self.envActionRange = (self.env.action_space.low, self.env.action_space.high)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        inDim = self.env.observation_space.shape[0]
        outDim = self.env.action_space.shape[0]
        hDim = [512,128]  
        activation = torch.nn.functional.relu  

        self.targetValueNetwork = createValueNetwork(inDim, outDim, hDim, activation)
        self.onlineValueNetwork = createValueNetwork(inDim, outDim, hDim, activation)

        self.targetPolicyNetwork = createPolicyNetwork(inDim, outDim, self.envActionRange, hDim, activation)
        self.onlinePolicyNetwork = createPolicyNetwork(inDim, outDim, self.envActionRange, hDim, activation)
        self.valueOptimizer = valueOptimizerFn(self.onlineValueNetwork, valueOptimizerLR)
        self.policyOptimizer = policyOptimizerFn(self.onlinePolicyNetwork, policyOptimizerLR)

        self.tau = tau
        self.gamma = gamma
        self.epochs = epochs
        self.batchSize = batchSize
        self.MAX_TRAIN_EPISODES = MAX_TRAIN_EPISODES
        self.MAX_EVAL_EPISODES = MAX_EVAL_EPISODES
        self.updateFrequency = updateFrequency

        self.replay_buffer_fn = lambda: ReplayBuffer(bufferSize=50000)
        

class DDPG(DDPG):
    def stepint(self, state, env):
        action = self.gaussianSelectAction(self.onlinePolicyNetwork, state, noiseScaleRatio = 1.5, explorationMax = self.replay_buffer.length() > self.min_samples)
        new_state, reward, is_terminal, is_truncated, info = env.step(action)
        is_failure = is_terminal and not is_truncated
        experience = (state, action, reward, new_state, float(is_failure))

        self.replay_buffer.store(experience)
        self.episode_reward[-1] += reward
        self.episode_timestep[-1] += 1
        return new_state, is_terminal

class DDPG(DDPG):
    def updateNetworks(self):
        for tParam, oParam in zip(self.targetValueNetwork.parameters(),
                                  self.onlineValueNetwork.parameters()):
            mixedWeights = (1 - self.tau) * tParam.data + self.tau * oParam.data
            tParam.data.copy_(mixedWeights)

        for tParam, oParam in zip(self.targetPolicyNetwork.parameters(),
                                  self.onlinePolicyNetwork.parameters()):
            mixedWeights = (1 - self.tau) * tParam.data + self.tau * oParam.data
            tParam.data.copy_(mixedWeights)
            
class DDPG(DDPG):
    def gaussianSelectAction(self, net, s, noiseScaleRatio, explorationMax = True):
        actionLowVal, actionHighVal = self.envActionRange
        if explorationMax:
            scale = actionHighVal
        else:
            scale = noiseScaleRatio * actionHighVal
        greedyAction = net(s).cpu().detach().squeeze().numpy()
        noise = np.random.normal(loc=0, scale=scale, size=len(actionHighVal))
        action = greedyAction + noise
        action = np.clip(action, actionLowVal, actionHighVal)

        return action
    

class DDPG(DDPG):
    def greedySelectAction(self, net, s):
        actionLowVal, actionHighVal = self.envActionRange
        action = net(s).cpu().detach().squeeze().numpy()
        action = np.clip(action, actionLowVal, actionHighVal)

        return action
    

class DDPG(DDPG):
    def runDDPG(self):
        #this is the main method, it trains the agent, performs bookkeeping while training and finally evaluates
        #the agent and returns the following quantities:
        #1. episode wise mean train rewards
        #2. epsiode wise mean eval rewards
        #2. episode wise trainTime (in seconds): time elapsed during training since the start of the first episode
        #3. episode wise wallClockTime (in seconds): actual time elapsed since the start of training,
        #                               note this will include time for BookKeeping and evaluation
        # Note both trainTime and wallClockTime get accumulated as episodes proceed.
        #
        #Your code goes in here
        resultList, trainTimeList, evalRewardsList, wallClockTimeList = self.trainAgent()
        resultEval = self.evaluateAgent()
        finalEvalReward  = np.mean(resultEval)

        return resultList, trainTimeList, evalRewardsList, wallClockTimeList, finalEvalReward
    
class DDPG(DDPG):
    def trainAgent(self):
        # this method collects experiences and trains the agent and does BookKeeping while training.
        # this calls the trainNetwork() method internally, it also evaluates the agent per episode
        # it trains the agent for MAX_TRAIN_EPISODES
        training_start = time.time()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.episode_timestep = []
        self.episode_reward = []
        self.episode_seconds = []
        self.evaluation_scores = []
        self.episode_exploration = []
        global train_count
        train_count +=1

        self.updateNetworks()

        self.replay_buffer = self.replay_buffer_fn()
        tempstate, _ = self.env.reset(seed=self.seed)

        max_episodes = self.MAX_TRAIN_EPISODES
        result = np.empty((2000, 5))
        result[:] = np.nan
        training_time = 0

        for episode in tqdm(range(1, max_episodes + 1)):
            episode_start = time.time()
            state, _ = self.env.reset(seed=self.seed)
            self.episode_reward.append(0.0)
            self.episode_timestep.append(0.0)
            self.episode_exploration.append(0.0)

            for step in range(self.env.spec.max_episode_steps):
                self.min_samples = self.replay_buffer.default_batch_size * self.batchSize
                state, is_terminal = self.stepint(state, self.env)

                if self.replay_buffer.length() > self.min_samples:
                    experiences = self.replay_buffer.sample(batchSize = self.batchSize)
                    experiences = self.onlineValueNetwork.load(experiences)
                    self.trainNetworks(experiences, self.epochs)

                if np.sum(self.episode_timestep) % self.updateFrequency == 0:
                    self.updateNetworks()

                if is_terminal:
                    break

            episode_elapsed = time.time() - episode_start
            self.episode_seconds.append(episode_elapsed)
            training_time += episode_elapsed
            evaluation_score = np.mean(self.evaluateAgent())
            total_step = int(np.sum(self.episode_timestep))
            self.evaluation_scores.append(evaluation_score)

            mean_10_reward = np.mean(self.episode_reward[-10:])
            std_10_reward = np.std(self.episode_reward[-10:])
            mean_100_reward = np.mean(self.episode_reward[-100:])
            std_100_reward = np.std(self.episode_reward[-100:])
            mean_100_eval_score = np.mean(self.evaluation_scores[-100:])
            std_100_eval_score = np.std(self.evaluation_scores[-100:])

            wallclock_elapsed = time.time() - training_start
            result[episode - 1] = total_step, mean_100_reward, mean_100_eval_score, training_time, wallclock_elapsed

        final_eval_rwd_list = self.evaluateAgent()
        mean_eval_rwd = np.mean(final_eval_rwd_list)
        wallclock_time = time.time() - training_start
        self.env.close()

        return result, training_time, final_eval_rwd_list, wallclock_time


class DDPG(DDPG):
    def trainNetworks(self, experiences, epochs):
        # this method trains the value network epoch number of times and is called by the trainAgent function
        # it essentially uses the experiences to calculate target, using the targets it calculates the error, which
        # is further used for calulating the loss. It then uses the optimizer over the loss
        # to update the params of the network by backpropagating through the network
        # this function does not return anything
        # you can try out other loss functions other than MSE like Huber loss, MAE, etc.
        #
        #Your code goes in here
        states, actions, rewards, next_states, is_terminals = experiences
        batch_size = len(is_terminals)
        next_states_cpu = next_states.cpu().detach().numpy()
        argmax_a_qs_v = self.targetPolicyNetwork(next_states_cpu).to(device)
        max_a_qs_v = self.targetValueNetwork(next_states, argmax_a_qs_v)

        target_qs = rewards + (self.gamma * max_a_qs_v * (1 - is_terminals))

        qs = self.onlineValueNetwork(states, actions)

        td_errors = target_qs.detach() - qs
        value_loss = td_errors.pow(2).mul(0.5).mean()
        self.valueOptimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.onlineValueNetwork.parameters(), 1.0)
        self.valueOptimizer.step()

        argmax_a_qs_p = self.onlinePolicyNetwork(next_states_cpu).to(device)
        max_a_qs_p = self.onlineValueNetwork(next_states, argmax_a_qs_p)

        policyLoss = max_a_qs_p.mean().mul(-1.0)
        self.policyOptimizer.zero_grad()
        policyLoss.backward()
        torch.nn.utils.clip_grad_norm_(self.onlinePolicyNetwork.parameters(), 1.0)
        self.policyOptimizer.step()

        return
    
class DDPG(DDPG):
    def evaluateAgent(self):
        #this function evaluates the agent using the value network, it evaluates agent for MAX_EVAL_EPISODES
        #typcially MAX_EVAL_EPISODES = 1
        rwd_list = []
        for _ in range(self.MAX_EVAL_EPISODES):
            state, _ = self.env.reset(seed=self.seed)
            rwd_list.append(0)
            for _ in count():
                action = self.greedySelectAction(self.onlinePolicyNetwork, state)
                state, rwd, done, truncated,_ = self.env.step(action)
                rwd_list[-1] += rwd
                if done or truncated: break
        return rwd_list



# Pendulum environment

train_count = 0
ddpg_results = []
best_agent, best_eval_score = None, float('-inf')
seed_list = [1, 100, 200, 300, 400]

for myseed in seed_list:

    val_optimizer_fn = lambda net, lr: optim.Adam(net.parameters(), lr=lr)
    pol_optimizer_fn = lambda net, lr: optim.Adam(net.parameters(), lr=lr)

    # Instantiation of the DDPG class
    ddpg_instance = DDPG(
        env_id='Pendulum-v1',
        seed=myseed,
        gamma=0.99,
        tau = 0.8,
        epochs = 20,
        bufferSize=50000,
        batchSize=15,
        updateFrequency = 1,
        valueOptimizerFn=val_optimizer_fn,
        valueOptimizerLR=2e-3,
        policyOptimizerFn=pol_optimizer_fn,
        policyOptimizerLR=1e-3,
        MAX_TRAIN_EPISODES=100,
        MAX_EVAL_EPISODES=1
    )

    # Running the DDPG method and appending results
    trainRewardsList, trainTimeList, evalRewardsList, wallClockTimeList, final_eval_score = ddpg_instance.runDDPG()
    ddpg_results.append(trainRewardsList)
    if final_eval_score > best_eval_score:
        best_eval_score = final_eval_score
        best_agent = ddpg_instance

# Convert dqn_results to a numpy array for any further processing
ddpg_results = np.array(ddpg_results)

reward_max, steps_max, eval_score_max, train_time_max, wall_clock_max = np.max(ddpg_results, axis=0).T
reward_min, steps_min, eval_score_min, train_time_min, wall_clock_min = np.min(ddpg_results, axis=0).T
reward_avg, steps_avg, eval_score_avg, train_time_avg, wall_clock_avg = np.mean(ddpg_results, axis=0).T
episode_indices = np.arange(len(reward_avg))

plt.style.use('ggplot')
fig, plot_areas = plt.subplots(5, 1, figsize=(12, 25), sharex='col')
fig.subplots_adjust(hspace=0.5)
colors = ['blue', 'green', 'red', 'purple', 'orange']
titles = ['Total Steps','Training Reward', 'Evaluation reward',  'Training Duration', 'Wall-clock Time']
y_labels = ['Steps', 'Reward', 'Reward', 'Seconds', 'Seconds']
data_max = [reward_max, steps_max, eval_score_max, train_time_max, wall_clock_max]
data_min = [reward_min, steps_min, eval_score_min, train_time_min, wall_clock_min]
data_avg = [reward_avg, steps_avg, eval_score_avg, train_time_avg, wall_clock_avg]

# Generate plots
for ax, title, color, max_data, min_data, avg_data, y_label in zip(plot_areas, titles, colors, data_max, data_min, data_avg, y_labels):
    ax.plot(max_data, linestyle='--', color=color, alpha=0.75)
    ax.plot(min_data, linestyle='--', color=color, alpha=0.75)
    ax.plot(avg_data, label='DDPG', color=color, linewidth=2)
    ax.fill_between(episode_indices, min_data, max_data, color=color, alpha=0.2)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.legend()

plot_areas[-1].set_xlabel('Episodes')

plt.savefig('ddpg_pendulum_plots.png', format='png', dpi=300)  
plt.show()



## Hopper environment

train_count = 0
ddpg_results = []
best_agent, best_eval_score = None, float('-inf')
seed_list = [3, 6, 7, 14, 17]

for myseed in seed_list:

    val_optimizer_fn = lambda net, lr: optim.RMSprop(net.parameters(), lr=lr)
    pol_optimizer_fn = lambda net, lr: optim.RMSprop(net.parameters(), lr=lr)

    # Instantiation of the DQN class
    ddpg_instance = DDPG(
        env_id='Hopper-v4',
        seed=myseed,
        gamma=0.99,
        tau = 0.8,
        epochs = 15,
        bufferSize=60000,
        batchSize=5,
        updateFrequency = 20,
        valueOptimizerFn=val_optimizer_fn,
        valueOptimizerLR=1e-3,
        policyOptimizerFn=pol_optimizer_fn,
        policyOptimizerLR=1e-3,
        MAX_TRAIN_EPISODES=150,
        MAX_EVAL_EPISODES=5
    )

    # Running the NFQ method and appending results
    trainRewardsList, trainTimeList, evalRewardsList, wallClockTimeList, final_eval_score = ddpg_instance.runDDPG()
    ddpg_results.append(trainRewardsList)
    if final_eval_score > best_eval_score:
        best_eval_score = final_eval_score
        best_agent = ddpg_instance

# Convert dqn_results to a numpy array for any further processing
ddpg_results = np.array(ddpg_results)


reward_max, steps_max, eval_score_max, train_time_max, wall_clock_max = np.max(ddpg_results, axis=0).T
reward_min, steps_min, eval_score_min, train_time_min, wall_clock_min = np.min(ddpg_results, axis=0).T
reward_avg, steps_avg, eval_score_avg, train_time_avg, wall_clock_avg = np.mean(ddpg_results, axis=0).T
episode_indices = np.arange(len(reward_avg))

plt.style.use('ggplot')
fig, plot_areas = plt.subplots(5, 1, figsize=(12, 25), sharex='col')
fig.subplots_adjust(hspace=0.5)
colors = ['blue', 'green', 'red', 'purple', 'orange']
titles = ['Total Steps','Training Reward', 'Evaluation reward',  'Training Duration', 'Wall-clock Time']
y_labels = ['Steps', 'Reward', 'Reward', 'Seconds', 'Seconds']
data_max = [reward_max, steps_max, eval_score_max, train_time_max, wall_clock_max]
data_min = [reward_min, steps_min, eval_score_min, train_time_min, wall_clock_min]
data_avg = [reward_avg, steps_avg, eval_score_avg, train_time_avg, wall_clock_avg]

# Generate plots
for ax, title, color, max_data, min_data, avg_data, y_label in zip(plot_areas, titles, colors, data_max, data_min, data_avg, y_labels):
    ax.plot(max_data, linestyle='--', color=color, alpha=0.75)
    ax.plot(min_data, linestyle='--', color=color, alpha=0.75)
    ax.plot(avg_data, label='DDPG', color=color, linewidth=2)
    ax.fill_between(episode_indices, min_data, max_data, color=color, alpha=0.2)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.legend()

plot_areas[-1].set_xlabel('Episodes')
plt.savefig('ddpg_hopper_plots.png', format='png', dpi=300)
plt.show()

## Half Cheetah environment



    
    