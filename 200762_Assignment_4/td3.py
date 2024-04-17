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
This class implements the TD3 agent, you are required to implement the various methods of this class
as outlined below. Note this class is generic and should work with any permissible Gym environment

```
class TD3():
    def init(env, gamma, tau,
    bufferSize ,
    updateFrequencyPolicy ,
    updateFrequencyValue ,
    trainPolicyFrequency ,
    policyOptimizerFn ,
    valueOptimizerFn ,
    policyOptimizerLR ,
    valueOptimizerLR ,
    MAX TRAIN EPISODES,
    MAX EVAL EPISODE,
    optimizerFn )
    
    def runTD3 (self)
    def trainAgent (self)
    def gaussianStrategy (self, net , s , envActionRange , noiseScaleRatio ,
        explorationMax = True)
    def greedyStrategy (self, net , s , envActionRange)
    def trainNetworks (self,experiences , envActionRange)
    def updateValueNetwork(self, onlineNet, targetNet, tau)
    def updatePolicyNetwork(self, onlineNet, targetNet, tau)
    def evaluateAgent (self)

"""

class TD3():
  def __init__(self, env_id,
               seed,
               gamma,
               tau,
               epochs,
               bufferSize,
               batchSize,
               updateFrequencyValue,
               updateFrequencyPolicy,
               trainPolicyFrequency,
               policyOptimizerFn,
               valueOptimizerFn,
               policyOptimizerLR,
               valueOptimizerLR,
               MAX_TRAIN_EPISODES,
               MAX_EVAL_EPISODES):

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
    
    self.gamma = gamma
    self.tau = tau
    self.epochs = epochs
    self.batchSize = batchSize
    self.MAX_TRAIN_EPISODES = MAX_TRAIN_EPISODES
    self.MAX_EVAL_EPISODES = MAX_EVAL_EPISODES
    self.updateFrequencyPolicy = updateFrequencyPolicy
    self.updateFrequencyValue = updateFrequencyValue
    self.trainPolicyFrequency = trainPolicyFrequency

    input_dim = self.env.observation_space.shape[0]
    output_dim = self.env.action_space.shape[0]
    hidden_dim = [512,128]  
    activation_fn = torch.nn.functional.relu  

    self.targetValueNetwork = createValueNetwork_TD3(input_dim, output_dim, hidden_dim, activation_fn)
    self.onlineValueNetwork = createValueNetwork_TD3(input_dim, output_dim, hidden_dim, activation_fn)
    self.targetPolicyNetwork = createPolicyNetwork(input_dim, output_dim, self.envActionRange, hidden_dim, activation_fn)
    self.onlinePolicyNetwork = createPolicyNetwork(input_dim, output_dim, self.envActionRange, hidden_dim, activation_fn)

    self.valueOptimizer = valueOptimizerFn(self.onlineValueNetwork, valueOptimizerLR)
    self.policyOptimizer = policyOptimizerFn(self.onlinePolicyNetwork, policyOptimizerLR)
    
    self.replay_buffer_fn = lambda : ReplayBuffer(bufferSize=60000)
    


class TD3(TD3):
    def stepint(self, state, env):
        action = self.gaussianSelectAction(self.onlinePolicyNetwork, state, noiseScaleRatio =  1.5, max_exploration = self.replay_buffer.length() > self.min_samples)
        new_state, reward, is_terminal, is_truncated, info = env.step(action)
        is_failure = is_terminal and not is_truncated
        experience = (state, action, reward, new_state, float(is_failure))
        self.replay_buffer.store(experience)
        self.episode_reward[-1] += reward
        self.episode_timestep[-1] += 1
        return new_state, is_terminal
    
class TD3(TD3):
    def updateValueNetwork(self):
        #this function updates the onlineNetwork with the target network using Polyak averaging 
        for tParam, oParam in zip(self.targetValueNetwork.parameters(),
                                  self.onlineValueNetwork.parameters()):
            mixedWeights = (1 - self.tau) * tParam.data + self.tau * oParam.data
            tParam.data.copy_(mixedWeights)
            
class TD3(TD3):
    def updatePolicyNetwork(self):
        #this function updates the onlineNetwork with the target network using Polyak averaging 
        for tParam, oParam in zip(self.targetPolicyNetwork.parameters(),
                                  self.onlinePolicyNetwork.parameters()):
            mixedWeights = (1 - self.tau) * tParam.data + self.tau * oParam.data
            tParam.data.copy_(mixedWeights)


class TD3(TD3):
    def gaussianSelectAction(self, net, state, noiseScaleRatio, max_exploration = True):
        action_min, action_max = self.envActionRange
        if max_exploration:
            scale = action_max
        else:
            scale = noiseScaleRatio * action_max
        greedyAction = net(state).cpu().detach().squeeze().numpy()
        noise = np.random.normal(loc=0, scale=scale, size=len(action_max))
        action_noise = greedyAction + noise
        action_clipped = np.clip(action_noise, action_min, action_max)

        return action_clipped
    
class TD3(TD3):
    def greedySelectAction(self, net, state):
        action_min, action_max = self.envActionRange
        action = net(state).cpu().detach().squeeze().numpy()
        action_clipped = np.clip(action, action_min, action_max)

        return action_clipped
    
class TD3(TD3):
    def runTD3(self):
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

class TD3(TD3):
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
        
        self.updateValueNetwork()
        self.updatePolicyNetwork()

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
                    experiences_load = self.onlineValueNetwork.load(experiences)
                    experiences = self.onlinePolicyNetwork.load(experiences)
                    self.trainNetworks(experiences_load, self.epochs)

                if np.sum(self.episode_timestep) % self.updateFrequencyValue == 0:
                    self.updateValueNetwork()

                if np.sum(self.episode_timestep) % self.updateFrequencyPolicy == 0:
                    self.updatePolicyNetwork()

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

class TD3(TD3):
    def trainNetworks(self, experiences, epochs):
        # this method trains the value network epoch number of times and is called by the trainAgent function
        # it essentially uses the experiences to calculate target, using the targets it calculates the error, which
        # is further used for calulating the loss. It then uses the optimizer over the loss
        # to update the params of the network by backpropagating through the network
        # this function does not return anything
        # you can try out other loss functions other than MSE like Huber loss, MAE, etc.

        states, actions, rewards, next_states, is_terminals = experiences
        next_states_cpu = next_states.cpu().detach().numpy()
        action_min, action_max = self.envActionRange
        action_min = torch.tensor(action_min).to(device)
        action_max = torch.tensor(action_max).to(device)     
        as_noise = (action_max - action_min) * torch.rand_like(actions.float())
        as_noise = torch.clamp(as_noise, action_min, action_max).to(device)

        argmax_a_q_val = self.targetPolicyNetwork(next_states_cpu).to(device)
        argmax_a_q_val_noisy = argmax_a_q_val + as_noise
        argmax_a_q_val_noisy = torch.clamp(argmax_a_q_val_noisy, action_min, action_max).to(device)
        
        q1, q2 = self.targetValueNetwork(next_states, argmax_a_q_val_noisy)
        target_q = torch.minimum(q1, q2).detach()
        target_q = rewards + (self.gamma * (1 - is_terminals) * target_q)
        current_q1, current_q2 = self.onlineValueNetwork(states, actions)

        # Compute the critic loss for both critics
        critic_loss1 = (current_q1 - target_q)
        critic_loss1 = critic_loss1.pow(2).mul(0.5).mean()
        critic_loss2 = (current_q2 - target_q)
        critic_loss2 = critic_loss2.pow(2).mul(0.5).mean()
        critic_loss = critic_loss1 + critic_loss2
        
        # Update the first critic
        self.valueOptimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.onlineValueNetwork.parameters(), 1.0)
        self.valueOptimizer.step()

        # Delayed policy updates
        if epochs % self.trainPolicyFrequency == 0:

            # Calculate policy loss
            argmax_a_q_pol = self.onlinePolicyNetwork(next_states_cpu).to(device)
            max_a_q_pol1, max_a_q_pol2 = self.onlineValueNetwork(next_states, argmax_a_q_pol)
            actor_loss = max_a_q_pol1.mean().mul(-1.0)
            
            # Update policy
            self.policyOptimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.onlinePolicyNetwork.parameters(), 1.0)
            self.policyOptimizer.step()

        return

class TD3(TD3):
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
td3_results = []
best_agent, best_eval_score = None, float('-inf')
seed_list = [1, 6, 7, 14, 17]

for myseed in seed_list:

    val_optimizer_fn = lambda net, lr: optim.RMSprop(net.parameters(), lr=lr)
    pol_optimizer_fn = lambda net, lr: optim.RMSprop(net.parameters(), lr=lr)

    # Instantiation of the TD3 class
    td3_instance = TD3(
        env_id='Pendulum-v1',
        seed=myseed,
        gamma=0.99,
        tau = 0.8,
        epochs = 15,
        bufferSize=50000,
        batchSize=5,
        updateFrequencyPolicy = 1,
        updateFrequencyValue = 3,
        trainPolicyFrequency = 2,
        valueOptimizerFn=val_optimizer_fn,
        valueOptimizerLR=1e-3,
        policyOptimizerFn=pol_optimizer_fn,
        policyOptimizerLR=1e-3,
        MAX_TRAIN_EPISODES=100,
        MAX_EVAL_EPISODES=1
    )

    # Running the NFQ method and appending results
    trainRewardsList, trainTimeList, evalRewardsList, wallClockTimeList, final_eval_score = td3_instance.runTD3()
    td3_results.append(trainRewardsList)
    if final_eval_score > best_eval_score:
        best_eval_score = final_eval_score
        best_agent = td3_instance

# Convert dqn_results to a numpy array for any further processing
td3_results = np.array(td3_results)

reward_max, steps_max, eval_score_max, train_time_max, wall_clock_max = np.max(td3_results, axis=0).T
reward_min, steps_min, eval_score_min, train_time_min, wall_clock_min = np.min(td3_results, axis=0).T
reward_avg, steps_avg, eval_score_avg, train_time_avg, wall_clock_avg = np.mean(td3_results, axis=0).T
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
    ax.plot(avg_data, label='TD3', color=color, linewidth=2)
    ax.fill_between(episode_indices, min_data, max_data, color=color, alpha=0.2)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.legend()

plot_areas[-1].set_xlabel('Episodes')
plt.savefig('td3_pendulum_plots.png', format='png', dpi=300) 
plt.show()


## Pendulum environment

train_count = 0
td3_results = []
best_agent, best_eval_score = None, float('-inf')
seed_list = [1, 4, 5, 10, 20]

for myseed in seed_list:

    val_optimizer_fn = lambda net, lr: optim.Adam(net.parameters(), lr=lr)
    pol_optimizer_fn = lambda net, lr: optim.Adam(net.parameters(), lr=lr)

    # Instantiation of the TD3 class
    td3_instance = TD3(
        env_id='Pendulum-v1',
        seed=myseed,
        gamma=0.99,
        tau = 0.8,
        epochs = 15,
        bufferSize=60000,
        batchSize=5,
        updateFrequencyValue = 4,
        updateFrequencyPolicy = 2,
        trainPolicyFrequency = 2,
        policyOptimizerFn = pol_optimizer_fn,
        valueOptimizerFn = val_optimizer_fn,
        valueOptimizerLR = 2e-3,
        policyOptimizerLR = 1e-3,
        MAX_TRAIN_EPISODES=200,
        MAX_EVAL_EPISODES=1
    )

    # Running the NFQ method and appending results
    trainRewardsList, trainTimeList, evalRewardsList, wallClockTimeList, final_eval_score = td3_instance.runTD3()
    td3_results.append(trainRewardsList)
    if final_eval_score > best_eval_score:
        best_eval_score = final_eval_score
        best_agent = td3_instance

# Convert dqn_results to a numpy array for any further processing
td3_results = np.array(td3_results)

reward_max, steps_max, eval_score_max, train_time_max, wall_clock_max = np.max(td3_results, axis=0).T
reward_min, steps_min, eval_score_min, train_time_min, wall_clock_min = np.min(td3_results, axis=0).T
reward_avg, steps_avg, eval_score_avg, train_time_avg, wall_clock_avg = np.mean(td3_results, axis=0).T
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
    ax.plot(avg_data, label='TD3', color=color, linewidth=2)
    ax.fill_between(episode_indices, min_data, max_data, color=color, alpha=0.2)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.legend()

plot_areas[-1].set_xlabel('Episodes')
plt.savefig('td3_hopper_plots.png', format='png', dpi=300) 
plt.show()

# Hopperr environment
train_count = 0
td3_results = []
best_agent, best_eval_score = None, float('-inf')
seed_list = [1, 4, 5, 10, 20]

for myseed in seed_list:

    val_optimizer_fn = lambda net, lr: optim.Adam(net.parameters(), lr=lr)
    pol_optimizer_fn = lambda net, lr: optim.Adam(net.parameters(), lr=lr)

    # Instantiation of the TD3 class
    td3_instance = TD3(
        env_id='Hopper-v4',
        seed=myseed,
        gamma=0.99,
        tau = 0.8,
        epochs = 15,
        bufferSize=60000,
        batchSize=5,
        updateFrequencyValue = 4,
        updateFrequencyPolicy = 2,
        trainPolicyFrequency = 2,
        policyOptimizerFn = pol_optimizer_fn,
        valueOptimizerFn = val_optimizer_fn,
        valueOptimizerLR = 2e-3,
        policyOptimizerLR = 1e-3,
        MAX_TRAIN_EPISODES=200,
        MAX_EVAL_EPISODES=1
    )

    # Running the NFQ method and appending results
    trainRewardsList, trainTimeList, evalRewardsList, wallClockTimeList, final_eval_score = td3_instance.runTD3()
    td3_results.append(trainRewardsList)
    if final_eval_score > best_eval_score:
        best_eval_score = final_eval_score
        best_agent = td3_instance

# Convert dqn_results to a numpy array for any further processing
td3_results = np.array(td3_results)


reward_max, steps_max, eval_score_max, train_time_max, wall_clock_max = np.max(td3_results, axis=0).T
reward_min, steps_min, eval_score_min, train_time_min, wall_clock_min = np.min(td3_results, axis=0).T
reward_avg, steps_avg, eval_score_avg, train_time_avg, wall_clock_avg = np.mean(td3_results, axis=0).T
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
    ax.plot(avg_data, label='TD3', color=color, linewidth=2)
    ax.fill_between(episode_indices, min_data, max_data, color=color, alpha=0.2)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.legend()

plot_areas[-1].set_xlabel('Episodes')
plt.savefig('td3_hopper_plots.png', format='png', dpi=300) 
plt.show()



## Half Cheetah environment

train_count = 0
td3_results = []
best_agent, best_eval_score = None, float('-inf')
seed_list = [2, 5, 9, 10, 16]

for myseed in seed_list:

    val_optimizer_fn = lambda net, lr: optim.RMSprop(net.parameters(), lr=lr)
    pol_optimizer_fn = lambda net, lr: optim.RMSprop(net.parameters(), lr=lr)

    # Instantiation of the TD3 class
    td3_instance = TD3(
        env_id='HalfCheetah-v4',
        seed=myseed,
        gamma=0.99,
        tau = 0.9,
        epochs = 15,
        bufferSize=50000,
        batchSize=5,
        updateFrequencyPolicy = 2,
        updateFrequencyValue = 2,
        trainPolicyFrequency = 2,
        valueOptimizerFn=val_optimizer_fn,
        valueOptimizerLR=1e-3,
        policyOptimizerFn=pol_optimizer_fn,
        policyOptimizerLR=1e-3,
        MAX_TRAIN_EPISODES=150,
        MAX_EVAL_EPISODES=1)

    # Running the NFQ method and appending results
    trainRewardsList, trainTimeList, evalRewardsList, wallClockTimeList, final_eval_score = td3_instance.runTD3()
    td3_results.append(trainRewardsList)
    if final_eval_score > best_eval_score:
        best_eval_score = final_eval_score
        best_agent = td3_instance

# Convert dqn_results to a numpy array for any further processing
td3_results = np.array(td3_results)

reward_max, steps_max, eval_score_max, train_time_max, wall_clock_max = np.max(td3_results, axis=0).T
reward_min, steps_min, eval_score_min, train_time_min, wall_clock_min = np.min(td3_results, axis=0).T
reward_avg, steps_avg, eval_score_avg, train_time_avg, wall_clock_avg = np.mean(td3_results, axis=0).T
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
    ax.plot(avg_data, label='TD3', color=color, linewidth=2)
    ax.fill_between(episode_indices, min_data, max_data, color=color, alpha=0.2)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.legend()

plot_areas[-1].set_xlabel('Episodes')
plt.savefig('td3_halfcheetah_plots.png', format='png', dpi=300) 
plt.show()



