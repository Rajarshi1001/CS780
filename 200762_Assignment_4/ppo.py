#All imports here

import os
import random
import time

import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from replay_buffer import *

#Hyperparameters
gym_id = "Pendulum-v1"  #The id of the gym environment
learning_rate = 0.0003
seed = 1
total_timesteps = 2048  #The total timesteps of the experiments
torch_deterministic = True   #If toggled, `torch.backends.cudnn.deterministic=False
cuda = True

num_envs = 4  #The number of parallel game environments (Yes PPO works with vectorized environments)
num_steps = 128 #The number of steps to run in each environment per policy rollout
anneal_lr = True #Toggle learning rate annealing for policy and value networks
gae = True #Use GAE for advantage computation
gamma = 0.99
gae_lambda =  0.95#The lambda for the general advantage estimation
num_minibatches = 4
update_epochs =  5 #The K epochs to update the policy
norm_adv = True  #Toggles advantages normalization
clip_coef = 0.2 #The surrogate clipping coefficient (See what is recommended in the paper!)
clip_vloss = True #Toggles whether or not to use a clipped loss for the value function, as per the paper
ent_coef =  0.01 #Coefficient of the entropy
vf_coef =  1 #Coefficient of the value function
max_grad_norm = 0.5
target_kl = None #The target KL divergence threshold


batch_size = int(num_envs * num_steps)
minibatch_size = int(batch_size // num_minibatches)
num_updates = 100
training_rewards = []
evaluation_rewards = []
wall_clock_times = []
episode_times = []


def compute_gae(next_value, rewards, masks, values, gamma=0.99, gae_lambda=0.95):

    if next_value.dim() == 2 and next_value.shape[1] == 1:
        next_value = next_value.transpose(0, 1).squeeze(0)
    if next_value.dim() == 1:
        next_value = next_value.unsqueeze(0)
    values = torch.cat([values, next_value], dim=0)
    returns = torch.zeros_like(rewards)
    advantages = torch.zeros_like(rewards)
    gae = 0
    
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * masks[t] - values[t]
        gae = delta + gamma * gae_lambda * masks[t] * gae
        returns[t] = gae + values[t]

    advantages = returns - values[:-1]  
    return returns

def compute_returns(next_value, rewards, masks, gamma=0.99):
    returns = torch.zeros_like(rewards)
    R = next_value
    for t in reversed(range(len(rewards))):
        R = rewards[t] + gamma * R * masks[t]
        returns[t] = R
    
    return returns

#PPO works with vectorized enviromnets lets make a function that returns a function that returns an environment.
#Refer how to make vectorized environments in gymnasium
#PPO works with vectorized enviromnets lets make a function that returns a function that returns an environment.
#Refer how to make vectorized environments in gymnasium
def make_env(gym_id, seed, num_envs):
    envs = gym.vector.make(gym_id, num_envs=num_envs)
    envs.reset(seed=seed)
    return envs


#We initialize the layers in PPO , refer paper.
#Lets initialize the layers with this function
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    #Initializes the weights and bias of the layers
    
    # He initilization
    
    if hasattr(layer, 'weight'):
        torch.nn.init.normal_(layer.weight, mean=0., std=np.sqrt(std / layer.weight.data.size(-1)))
    if hasattr(layer, 'bias') and layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


#Lets make the Main agent class
class Agent(nn.Module):
    
    def __init__(self, envs):
        super(Agent, self).__init__()
        
        in_dim = envs.single_observation_space.shape[0]
        out_dim = envs.single_action_space.shape[0]
        env_action_range = (envs.single_action_space.low, envs.single_action_space.high)
        
        hDim = [512, 128]
        self.actor = createPolicyNetworkPPO(in_dim, out_dim, env_action_range, hDim, activation=torch.nn.functional.relu)
        self.critic = createValueNetwork(in_dim, out_dim, hDim, activation=torch.nn.functional.relu)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def get_value(self, x):
        # Returns the value from the critic on the observation x
        x = x.to(self.device)
        value = self.critic(x)
        return value

    def get_action_and_value(self, x, action=None):
        x = x.to(self.device)
        if action is None:
            action, log_prob, entropy = self.actor(x)  
        else:
            action = action.to(self.device)
            _, log_prob, entropy = self.actor(x)  

        value = self.critic(x, action)  
        return action, log_prob, value, entropy


random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = torch_deterministic

device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")


#Make the vectorized environments, use the helper function that we have declared above
envs = make_env(gym_id, seed = seed, num_envs = num_envs)
agent = Agent(envs).to(device)
optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps = 2.5e-4) #eps is not the default that pytorch uses

# ALGO Logic: Storage setup
obs = torch.zeros((num_steps, num_envs) + envs.single_observation_space.shape).to(device)
actions = torch.zeros((num_steps, num_envs) + envs.single_action_space.shape).to(device)
logprobs = torch.zeros((num_steps, num_envs)).to(device)
rewards = torch.zeros((num_steps, num_envs)).to(device)
dones = torch.zeros((num_steps, num_envs)).to(device)
values = torch.zeros((num_steps, num_envs)).to(device)

#This is the main training loop where we collect the experience ,
#calculate the advantages, ratio , the total loss and learn the policy

window_size = 10
moving_avg_rewards = []
recent_rewards = []

for update in range(1, num_updates + 1):
    
    episode_start_time = time.time() 
    total_episode_reward = 0 

    # Annealing the rate if instructed to do so.
    if anneal_lr:
        scheduler.step()

    for step in range(0, num_steps):
        global_step += 1 * num_envs  # We are taking a step in each environment
        obs[step] = next_obs
        dones[step] = next_done

        # ALGO LOGIC: action logic
        with torch.no_grad():
            #Get the action , logprob , _ , value from the agent.
            action, logprob, _, value = agent.get_action_and_value(next_obs)
            values[step] = value.flatten()
            logprob = logprob.squeeze(1)
        actions[step] = action
        logprobs[step] = logprob

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, done, truncated, info = envs.step(action.cpu().numpy())
        rewards[step] = torch.tensor(reward).to(device).view(-1)
        next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
        total_episode_reward += reward.sum().item() 
        # for item in info:
        #     if item == "final_info" and info[item][0] is not None:
        #         print(f"global_step={global_step}, episodic_return={info[item][0]['episode']['r']}")
        #         break

    # bootstrap value if not done
    with torch.no_grad():
        next_value = agent.get_action_and_value(next_obs)[2].detach()

        # Reshape next_value to ensure it matches the expected dimensions
        if next_value.dim() == 1:
            next_value = next_value.unsqueeze(-1)  
        if gae:

            advantages = compute_gae(next_value, rewards, dones, values, gamma=gamma, gae_lambda = gae_lambda)
            returns = advantages + values  #(yes official implementation of ppo calculates it this way)
        else:
            returns = compute_returns(next_value, rewards, dones, gamma)
            advantages = returns - values

    recent_rewards.append(total_episode_reward)
    if len(recent_rewards) > window_size:
        recent_rewards.pop(0)  
    moving_average = np.mean(recent_rewards)  
    moving_avg_rewards.append(moving_average)

    episode_duration = time.time() - episode_start_time
    wall_clock_times.append(episode_duration)
    training_rewards.append(total_episode_reward / num_envs)
    
    # flatten the batch
    b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)

    # Optimizing the policy and value network
    b_inds = np.arange(batch_size)
    clipfracs = []
    for epoch in range(update_epochs):
        #Get a random sample of batch_size
        np.random.shuffle(b_inds)
        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_inds = b_inds[start:end]

            # Extract minibatch samples
            mb_obs = b_obs[mb_inds]
            mb_actions = b_actions[mb_inds]
            mb_logprobs = b_logprobs[mb_inds]
            mb_advantages = b_advantages[mb_inds]
            mb_returns = b_returns[mb_inds]
            mb_values = b_values[mb_inds]
            
            # Normalize advantages
            if norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
            _, newlogprob, entropy, new_value = agent.get_action_and_value(mb_obs, mb_actions)
            # logratio = 
            ratio = torch.exp(newlogprob - mb_logprobs)

            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                # Refer the blog for calculating kl in a simpler way
                approx_kl = torch.mean(newlogprob - mb_logprobs)
                clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

            mb_advantages = b_advantages[mb_inds]
            if norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # Policy loss (Calculate the policy loss pg_loss)
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * mb_advantages
            pg_loss = - torch.min(surr1, surr2).mean()

            # Value loss v_loss
            new_value = new_value.view(-1)
            
            if clip_vloss:
                v_loss_unclipped = (new_value - mb_returns) ** 2
                v_clipped = mb_values + torch.clamp(new_value - mb_values, -clip_coef, clip_coef)
                v_loss_clipped = (v_clipped - mb_returns) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
            else:
                v_loss = 0.5 * ((new_value - mb_returns) ** 2).mean()

            # Entropy loss
            entropy_loss = entropy.mean()

            # Total loss
            loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            optimizer.step()

        if target_kl is not None:
            if approx_kl > target_kl:
                print("Eatly stopping due to reaching max KL divergence")
                break

    y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
    print(f"Explained Variance: {explained_var}")

envs.close()




