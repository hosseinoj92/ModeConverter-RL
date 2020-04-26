
########################## LIBRARIES AND PACKAGES #####################################

from simFrame.environment import Environment
import simFrame.permittivities as permittivities
import numpy as np
import os
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import time
import matplotlib.pyplot as plt
import csv
from csv import writer

from utility import *
from player import Player

########################### DEFINING SIMFRAME ENVIRONMENT ###########################

dims = [150, 100, 1]
environmentConfig = {"dimensions":dims,
          "designArea":[6, 6],  # the area that is to be changed pixel wise
          "structureScalingFactor":2,
          # pixels in designArea will be scaled up by this factor for the simulation.
          "waveguides":[[(0, 44), (dims[0] / 2, 56)], [(dims[0] / 2, 44), (dims[0], 56)]],
          # Array with components [(x1,y1), (x2,y2)]
          "thickness":1,  # thickness of designArea. '1' for 2D
          "wavelength":775,  # lambda in nm
          "pixelSize":40,  # pixelSize for simulation in nm
          "structurePermittivity":permittivities.SiN,  # permittivity of the structure
          "substratePermittivity":permittivities.SiO,  # permittivity of the substrate
          "surroundingPermittivity":permittivities.Air,  # permittivity of the surrounding (typically air)
          "inputModes":[{'pos': [[12, 12, 0], [12, dims[1] - 12, 0]], 'modeNum': 0}],
          # array of input Modes. modeNum 0 is TE00
          "outputModes":[{'pos': [[dims[0] - 12, 12, 0], [dims[0] - 12, dims[1] - 12, 0]],
                        'modeNum': 0}]}  # array of output Modes. modeNum 4 is TE20 in 2D

# Setting local simulation
os.environ["SIMULATE_ON_THIS_MACHINE"] = "1"

def figureOfMerit(overlaps):
    return overlaps[0][0]

####################################################################################################################
######################### EXTENDING THE ENVIRONMENT WITH DEFINING THE ACTION TAKER #################################
####################################################################################################################

'''note that x,y are not actual coordinates! x is the number of rows and y is the
number of columns.'''

########################################################################################################
####################### SETTING FUNCTIONS FOR SAVING USEFULL DATA TO CSV ##############################
########################################################################################################

with open('Track Rewards.csv', 'wb') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)

class ExperienceReplay(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, new_state, reward, done):
        transition = (state, action, new_state, reward, done)

        if self.position >= len(self.memory):
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return zip(*random.sample(self.memory, batch_size))

    def __len__(self):
        return len(self.memory)


#################### CREATING AN INSTANCE OF THE ACTION TAKER #########################

gamePlayer = Player(environmentConfig=environmentConfig,
                    figureOfMerit=figureOfMerit,
                    hashFilePath="hashes.pkl")

######################################################################################

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
Tensor = torch.Tensor
LongTensor = torch.LongTensor

##### PARAMS #####
learning_rate = 0.0001
num_episodes = 2000
gamma = 0.99

hidden_layer1 = 256
hidden_layer2 = 256
hidden_layer3 = 256


replay_mem_size = 1000000
batch_size = 32

updated_target_frequency = 10000

egreedy_initial = 1.0
egreedy_final = 0.1
egreedy_decay = 100000

clip_error = True
##################

###### DEFINING THE EXPLORATION/EXPLOITATION TRADE OFF FUNCTION ##############
def calculate_epsilon(steps_done):
    epsilon = egreedy_final + (egreedy_initial - egreedy_final) * \
              math.exp(-1 * steps_done / egreedy_decay)

    return epsilon


number_of_inputs = gamePlayer.observation_space
number_of_outputs = gamePlayer.action_space.n


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(number_of_inputs, hidden_layer1)
        self.linear2 = nn.Linear(hidden_layer1, hidden_layer2)
        self.linear3 = nn.Linear(hidden_layer2, hidden_layer3)
        self.linear4 = nn.Linear(hidden_layer3, number_of_outputs)

        # self.activation = nn.Tanh()
        self.activation = nn.ReLU()

    def forward(self, x):
        output1 = self.linear1(x)
        output1 = self.activation(output1)

        output2 = self.linear2(output1)
        output2 = self.activation(output2)

        output3 = self.linear3(output2)
        output3 = self.activation(output3)

        output4 = self.linear4(output3)
        output4 = self.activation(output4)


        return output4

class QNet_Agent():

    def __init__(self):
        self.nn = NeuralNetwork().to(device)
        self.target_nn = NeuralNetwork().to(device)

        self.loss_func = nn.MSELoss()
        self.optimizer = optim.Adam(params=self.nn.parameters(), lr=learning_rate)

        self.updated_target_counter = 0

    def select_action(self, state, epsilon):

        random_for_egreedy = torch.rand(1)[0]

        if random_for_egreedy > epsilon:
            with torch.no_grad():
                state = torch.Tensor(state).to(device)
                action_from_nn = self.nn(state)  # This part is the important one and the rest is for clarity
                action = torch.max(action_from_nn, 0)[1]
                action = action.item()

        else:

            action = gamePlayer.action_space.sample()

        return action

    def optimize(self):

        if (len(memory) < batch_size):
            return

        state, action, new_state, reward, done = memory.sample(batch_size)

        state = torch.Tensor(state).to(device)
        new_state = torch.Tensor(new_state).to(device)
        reward = torch.Tensor(reward).to(device)
        action = LongTensor(action).to(device)
        done = Tensor(done).to(device)

        new_state_values = self.target_nn(new_state).detach()
        max_new_state_values = torch.max(new_state_values, 1)[0]
        target_value = reward + (1 - done) * gamma * max_new_state_values

        predicted_value = self.nn(state).gather(1, action.unsqueeze(1)).squeeze(1)

        loss = self.loss_func(predicted_value, target_value)

        self.optimizer.zero_grad()
        loss.backward()

        if clip_error:
            for param in self.nn.parameters():
                param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

        if self.updated_target_counter % updated_target_frequency == 0:
            self.target_nn.load_state_dict(self.nn.state_dict())

        self.updated_target_counter += 1


memory = ExperienceReplay(replay_mem_size)
qnet_agent = QNet_Agent()

steps_total = []
episodes_total = 0
frames_total = 0

start = time.time()
for i_episode in range(num_episodes):

    state = gamePlayer.reset()

    time_step = 0
    cum_reward = 0

    while True:

        time_step += 1
        frames_total += 1

        epsilon = calculate_epsilon(frames_total)
        action = qnet_agent.select_action(state, epsilon)
        new_state, reward, done = gamePlayer.step(action, time_step)

        memory.push(state, action, new_state, reward, done)
        qnet_agent.optimize()

        state = new_state
        cum_reward += reward

        if done:
            steps_total.append(time_step)

            gamePlayer.loop_list.append(gamePlayer.loop_counter)

            gamePlayer.mean_efficiency.append(sum(gamePlayer.efficiency_list)/len(gamePlayer.efficiency_list))

            gamePlayer.sum_rewards.append(sum(gamePlayer.reward_list))
            import_data_to_csv('Track Rewards.csv', sum(gamePlayer.reward_list), i_episode, len(gamePlayer.reward_list),
                               gamePlayer.loop_counter)


            episodes_total += 1
            end = time.time()

            gamePlayer.convergence_control.append(gamePlayer.updated_efficiency)

            if episodes_total % 1 == 0:

                print('########################################')
                print('Episode {} finished after {} steps'.format(episodes_total, time_step))
                print('Rewards collected in the episode %0.3f' % cum_reward)
                print('Exploration variable is %0.3f' % epsilon)
                print('So far %i episodes have been completed:' % episodes_total)
                print('elapsed time was ( %d ) seconds: ' % (end - start))
                print('########################################\n')

            break

plt.figure(figsize=(12, 5))
plt.title("Rewards per Episode")
plt.bar(torch.arange(episodes_total), gamePlayer.sum_rewards, alpha=0.6, color='green', width=1)
plt.show()

plt.figure(figsize=(12, 5))
plt.title("Convergence Control")
plt.bar(torch.arange(episodes_total), gamePlayer.convergence_control, alpha=0.6, color='blue', width=1)
plt.show()

plt.figure(figsize=(12, 5))
plt.title("loop counts / episode")
plt.bar(torch.arange(episodes_total), gamePlayer.loop_list, alpha=0.6, color='red', width=1)
plt.show()

plt.figure(figsize=(12, 5))
plt.title("average efficiency / episode")
plt.bar(torch.arange(episodes_total), gamePlayer.mean_efficiency, alpha=0.6, color='blue', width=1)
plt.show()
