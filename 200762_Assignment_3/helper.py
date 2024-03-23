
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.distributions import Categorical
import time
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as a


gamma = 0.99
epsilon = 1.0
temp = 1.0
delta = 0.25
tau = 0.2
alpha = 0.6
beta = 0.1
beta_rate = 0.9
MAX_TRAIN_EPISODES = 80
MAX_EVAL_EPISODES = 1
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def selectGreedyAction(net, state):
    #this function gets q-values via the network and selects greedy action from q-values and returns it

    # converting the state to a pytorch tensor
    if isinstance(state, tuple):
      state = state[0]
    else:
      state = state

    state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        q_values = net(state)
    greedyAction = np.argmax(q_values.cpu().data.numpy())

    return greedyAction



def selectEpsilonGreedyAction(net,  state, epsilon):
    #this function gets q-values via the network and selects an action from q-values using epsilon greedy strategy
    #and returns it
    #note this function can be used for decaying epsilon greedy strategy,
    #you would need to create a wrapper function that will handle decaying epsilon
    #you can create this wrapper in this helper function section
    #for the agents you would be implementing it would be nice to play with decaying parameter to get optimal results
    if isinstance(state, tuple):
        state = state[0]
    else:
        state = state

    state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        q_values = net(state)

    if np.random.rand() < epsilon:
      num_actions = q_values.shape[-1]

      eGreedyAction = np.random.randint(0, num_actions)
    else:
      eGreedyAction = np.argmax(q_values.cpu().data.numpy())
    return eGreedyAction

# decay epsilon
def decayEpsilon(initial_epsilon, min_epsilon, decay_rate, epoch):
  """
    This function applies exponential decay to epsilon.
    """
  epsilon = max(min_epsilon, initial_epsilon * np.exp(-decay_rate * epoch))
  return epsilon


def selectSoftMaxAction(net, state, temp):
    #this function gets q-values via the network and selects an action from q-values using softmax strategy
    #and returns it
    #note this function can be used for decaying temperature softmax strategy,
    #you would need to create a wrapper function that will handle decaying temperature
    #you can create this wrapper in this helper function section
    #for the agents you would be implementing it would be nice to play with decaying parameter to get optimal results

    # converting the state to a pytorch tensor
    if isinstance(state, tuple):
      state = state[0]
    else:
      state = state
    state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        q_values = net(state)

    probs = torch.softmax(q_values / temp, dim=1).cpu().numpy()
    probs = probs.ravel()
    probs /= probs.sum()  

    softAction = np.random.choice(len(probs), p = probs)

    return softAction

# decay temperature
def decayTemperature(initial_temp, min_temp, decay_rate, epoch):
  """
  This function applies exponential decay to epsilon.
  """
  temp = max(min_temp, initial_temp * np.exp(-decay_rate * epoch))
  return temp


#Value Network
class ValueNetwork(nn.Module):

  def __init__(self, inDim, outDim, hDim=[32, 32], activation=F.relu):
    super(ValueNetwork, self).__init__()
    self.activation = activation
    self.hidden_layers = nn.ModuleList([nn.Linear(inDim, hDim[0])])

    # Declare hidden layers
    layer_sizes = zip(hDim[:-1], hDim[1:])
    self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

    self.output_layer = nn.Linear(hDim[-1], outDim)

  def forward(self, x):
    for hidden_layer in self.hidden_layers:
      x = self.activation(hidden_layer(x))

    output = self.output_layer(x)
    return output

def createValueNetwork(inDim, outDim, hDim = [32,32], activation = F.relu):
    #this creates a Feed Forward Neural Network class and instantiates it and returns the class
    #the class should be derived from torch nn.Module and it should have init and forward method at the very least
    #the forward function should return q-value for each possible action

    valueNetwork = ValueNetwork(inDim, outDim, hDim, activation)

    return valueNetwork

#Dueling Network

class DuelingNetwork(nn.Module):

  def __init__(self, inDim, outDim, hDim = [32, 32], activation = F.relu):
    super(DuelingNetwork, self).__init__()

    self.activation = activation
    self.hidden_layers = nn.ModuleList([nn.Linear(inDim, hDim[0])])

    # declare hidden layers
    layer_sizes = zip(hDim[:-1], hDim[1:])
    self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
    self.value_layer = nn.Linear(hDim[1], 1) # outputs V(state)
    self.advantage_layer =  nn.Linear(hDim[1], outDim) # outputs advantage a(state, action)

  def forward(self, x):
    for hidden_layer in self.hidden_layers:
      x = self.activation(hidden_layer(x))

    state_value = self.value_layer(x)
    advantage = self.advantage_layer(x)
    advantage_mean = advantage.mean(dim = 1, keepdim = True)

    q_values = state_value + (advantage - advantage_mean)

    return q_values


def createDuelingNetwork(inDim, outDim, hDim = [32,32], activation = F.relu):
    #this creates a Feed Forward Neural Network class and instantiates it and returns the class
    #the class should be derived from torch nn.Module and it should have init and forward method at the very least
    #the forward function should return q-value which is derived
    #internally from action-advantage function and v-function,
    #Note we center the advantage values, basically we subtract the mean from each state-action value

    duelNetwork = DuelingNetwork(inDim, outDim, hDim, activation)

    return duelNetwork


def createDuelingNetwork(inDim, outDim, hDim = [32,32], activation = F.relu):
    #this creates a Feed Forward Neural Network class and instantiates it and returns the class
    #the class should be derived from torch nn.Module and it should have init and forward method at the very least
    #the forward function should return q-value which is derived
    #internally from action-advantage function and v-function,
    #Note we center the advantage values, basically we subtract the mean from each state-action value

    duelNetwork = DuelingNetwork(inDim, outDim, hDim, activation)

    return duelNetwork
#Policy Network

class PolicyNetwork(nn.Module):

  def __init__(self, inDim, outDim, hDim=[32, 32], activation=F.relu):
    
    super(PolicyNetwork, self).__init__()
    self.layers = nn.ModuleList()
    prev_dim = inDim
    for dim in hDim:
        self.layers.append(nn.Linear(prev_dim, dim))
        prev_dim = dim
    self.output_layer = nn.Linear(prev_dim, outDim)
    self.activation = activation
    self.softmax = nn.Softmax(dim = -1)

  def forward(self, x):
    for hidden_layer in self.layers:
      x = self.activation(hidden_layer(x))

    x = self.output_layer(x)
    output = self.softmax(x)
    return output

def createPolicyNetwork(inDim, outDim, hDim = [32,32], activation = F.relu):
    #this creates a Feed Forward Neural Network class and instantiates it and returns the class
    #the class should be derived from torch nn.Module and it should have init and forward method at the very least
    #the forward function should return action logit vector

    policyNetwork = PolicyNetwork(inDim, outDim, hDim, activation)

    return policyNetwork


def plotQuantity(quantityListDict, totalEpisodeCount, descriptionList, filename):
    #this function takes in the quantityListDict and plots quantity vs episodes.
    #quantityListListDict = {envInstanceCount: quantityList}
    #quantityList is list of the qunatity per episode,
    #for example it could be mean reward per episode, traintime per episode, etc.
    #
    #NOTE: len(quantityList) == totalEpisodeCount
    #
    #Since we run multiple instances of the environment, there will be variance across environments
    #so in the plot, you will plot per episode maximum, minimum and average value across all env instances
    #Basically, you need to envelop (e.g., via color) the quantity between max and min with mean value in between
    #
    #use the descriptionList parameter to put legends, title, etc.
    #For each of the plot, create the legend on the left/right side so that it doesn't overlay on the plot lines/envelop.
    #
    #this is a generic function and can be used to plot any of the quantity of interest
    #In particular we will be using this function to plot:
    #        mean train rewards vs episodes
    #        mean evaluation rewards vs episodes
    #        total steps vs episode
    #        train time vs episode
    #        wall clock time vs episode
    #
    #
    mean_values = []
    max_values = []
    min_values = []

    for episode in range(totalEpisodeCount):
        episode_values = [quantityList[episode] for quantityList in quantityListDict.values()]
        mean_values.append(np.mean(episode_values))
        max_values.append(np.max(episode_values))
        min_values.append(np.min(episode_values))

    episodes = np.arange(1, totalEpisodeCount + 1)

    plt.figure(figsize=(10, 6))
    plt.fill_between(episodes, min_values, max_values, color='skyblue', alpha=0.5, label='Min-Max Range')
    plt.plot(episodes, mean_values, label='Mean', linewidth=2)
    # plt.plot(episodes, max_values, label='Max', color='darkgreen', linewidth=1.5)
    # plt.plot(episodes, min_values, label='Min', color='darkred', linewidth=1.5)

    if descriptionList:
        plt.title(descriptionList[0])
        plt.xlabel(descriptionList[1])
        plt.ylabel(descriptionList[2])
    plt.legend(loc='best')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    plt.savefig(filename, format='png', dpi=300)
    plt.show()
    plt.close()