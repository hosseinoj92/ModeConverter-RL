
########################## LIBRARIES AND PACKAGES #####################################

from simFrame.environment import Environment
import simFrame.permittivities as permittivities
import numpy as np
import os
import gym
from gym.utils import seeding
from gym import spaces, logger
import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import time
import matplotlib.pyplot as plt

########################### IMPORTING SIMFRAME ENVIRONMENT ###########################


dims = [150, 100, 1]
env = Environment(dimensions=dims,
                  designArea=[20, 20],  # the area that is to be changed pixel wise
                  structureScalingFactor=3,
                  # pixels in designArea will be scaled up by this factor for the simulation.
                  waveguides=[[(0, 44), (dims[0] / 2, 56)], [(dims[0] / 2, 35), (dims[0], 65)]],
                  # Array with components [(x1,y1), (x2,y2)]
                  thickness=1,  # thickness of designArea. '1' for 2D
                  wavelength=775,  # lambda in nm
                  pixelSize=40,  # pixelSize for simulation in nm
                  structurePermittivity=permittivities.SiN,  # permittivity of the structure
                  substratePermittivity=permittivities.SiO,  # permittivity of the substrate
                  surroundingPermittivity=permittivities.Air,  # permittivity of the surrounding (typically air)
                  inputModes=[{'pos': [[12, 12, 0], [12, dims[1] - 12, 0]], 'modeNum': 0}],
                  # array of input Modes. modeNum 0 is TE00
                  outputModes=[{'pos': [[dims[0] - 12, 12, 0], [dims[0] - 12, dims[1] - 12, 0]],
                                'modeNum': 4}])  # array of output Modes. modeNum 4 is TE20 in 2D

# Setting local simulation
os.environ["SIMULATE_ON_THIS_MACHINE"] = "1"
producePlots = 0

# Setting initial structure

env.setStructure(np.ones(400).reshape(20, 20))


def figureOfMerit(overlaps):
    return overlaps[0][0]


env.setFOM(figureOfMerit)

######################### EXTENDING THE ENVIRONMENT WITH DEFINING THE ACTION TAKER #################################

x_threshold = 19
y_threshold = 19
x_min = 0
y_min = 0

UP, DOWN, LEFT, RIGHT, FLIP = 0, 1, 2, 3, 4

######## DESCRIPTION OF IMPORTANT FUNCTIONS #########
'''
----------------------------------------
THE MOVEMENT (step(action,step)): 
----------------------------------------
has a discrete space of 5. The Agent can go UP,DOWN,RIGHT,LEFT and FLIP. By flipping, the agent 
changes the silicon-nitride pixel with air. The Agent then collects the reward via reward function
and checks if the episode should be terminated via done function and finally reshapes the state.
It is worth mentioning that if the agent hits the walls, it will reset form the opposite side. for example,
if we are at maximum right (max(x), y), taking the "RIGHT" action will take the agent to position (0,y) and the same for y.
STATE RESHAPE: In the beginning our structure is set as a 20x20 (400 pixels) matrix with values set to all 1 
(meaning silicon-nitride).by taking actions we move in this matrix and if we flipp a pixel we change the value 
from 1 to 0 (meaning air). For feeding this matrix to a neural network, we can reshape it to a 1x400 array containing
all the values of 0s and 1s. In the end we should also append the values of position to the 1x400 array.
For helping the neural network to make better decisions, it is better if we normilize the position values also between 
0 and 1. But it should be done carefully.
----------------------------------------
THE REWARD FUNCTION (get_reward()):
----------------------------------------
base efficiency, when no pixels are flipped is about 8%. It can be calculated via "env.evaluate()". 
In each time step the efficieny is observed and compared to the previous time step. If we have an increse in efficiency, 
the "reward = 1" is given to the agent and the base efficiency is updated to current efficiency. If not the "reward = 0"
---------------------------------------- 
THE DONE FUNCTION (get_done()):
----------------------------------------
This function determines when the episode should be terminated. In this version we consider the episode done, when
200 steps is taken.
----------------------------------------
THE RESET FUNCTION (reset()):
----------------------------------------
Using the mother function "env.resetStructure()", we can reset all the values of structure to 1. then reshape the state
and append the position (0,0) which indicates top left corner to the reshaped state. set the value of efficiency and
setting the base efficiency back to 8% again.
'''

'''note that x,y are not actual coordinates! x is the number of rows and y is the 
number of columns.'''


class Player(gym.Env):

    def __init__(self, environment: Environment):

        self.environment = environment

        # self.state = env.structure

        self.x = 0
        self.y = 0
        self.base_efficiency = 0.088
        self.updated_efficiency = 0

        self.action_space = spaces.Discrete(5)
        # self.observation_space = spaces.Discrete((2 ** 25) * 25)

    def get_reward(self):

        efficiency_evaluation = env.evaluate()
        self.updated_efficiency = efficiency_evaluation

        if efficiency_evaluation > self.base_efficiency:
            reward = 1
            self.base_efficiency = efficiency_evaluation
        else:
            reward = -0.2

        return reward

    def get_done(self, step):

        if step == 200:
            done = True
        else:
            done = False

        return done

    def step(self, action, step):

        # assert self.action_space.contains(action)

        reward = 0

        if action == RIGHT:
            if self.y < y_threshold:
                self.y = self.y + 1
            else:
                self.y = y_min

        elif action == LEFT:
            if self.y > y_min:
                self.y = self.y - 1
            else:
                self.y = y_threshold

        elif action == UP:
            if self.x > x_min:
                self.x = self.x - 1
            else:
                self.x = x_threshold

        elif action == DOWN:
            if self.x < x_threshold:
                self.x = self.x + 1
            else:
                self.x = x_min

        elif action == DOWN:
            if self.y > y_threshold:
                self.y = self.y - 1
            else:
                self.y = y_min

        elif action == FLIP:
            env.flipPixel([self.x, self.y])

        # print('FIGURE OF MERIT IS:', env.evaluate())
        reward += self.get_reward()

        done = self.get_done(step)

        # Reshaping the state from a nxn matrix to a 1xn^2 matrix,
        # AND Appending the position of the action taker to the states:

        reshaped_structure = np.reshape(env.structure, (1, 400))
        reshaped_state = np.append(reshaped_structure, (self.x, self.y))

        state = reshaped_state

        print('CURRENT POSITION OF AGENT: ', (self.x, self.y))  # This part is for control
        print(env.structure)  # This part is for control
        print('EFFICIENCY IS:', self.updated_efficiency)  # This part is for control

        return state, reward, done

    def reset(self):

        env.resetStructure()  # resets the Environment to all silicon ( with values = 1 )
        reset_reshaped_structure = np.reshape(env.structure, (1, 400))
        reset_reshaped_state = np.append(reset_reshaped_structure, (0, 0))
        state = reset_reshaped_state

        self.base_efficiency = 0.088

        return state


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

gamePlayer = Player(environment=env)

######################################################################################

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
Tensor = torch.Tensor
LongTensor = torch.LongTensor

##### PARAMS #####
learning_rate = 0.01
num_episodes = 500
gamma = 0.9

hidden_layer = 64

replay_mem_size = 50000
batch_size = 32

egreedy = 0.9
egreedy_final = 0.02
egreedy_decay = 500


##################

###### DEFINING THE EXPLORATION/EXPLOITATION TRADE OFF FUNCTION ##############
def calculate_epsilon(steps_done):
    epsilon = egreedy_final + (egreedy - egreedy_final) * \
              math.exp(-1 * steps_done / egreedy_decay)

    return epsilon


number_of_inputs = 402
number_of_outputs = gamePlayer.action_space.n


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(number_of_inputs, hidden_layer)
        self.linear2 = nn.Linear(hidden_layer, number_of_outputs)

        self.activation = nn.Tanh()
        # self.activation = nn.ReLU()

    def forward(self, x):
        output1 = self.linear1(x)
        output1 = self.activation(output1)
        output2 = self.linear2(output1)

        return output2


class QNet_Agent():

    def __init__(self):
        self.nn = NeuralNetwork().to(device)
        self.loss_func = nn.MSELoss()
        self.optimizer = optim.Adam(params=self.nn.parameters(), lr=learning_rate)

    def select_action(self, state, epsilon):

        random_for_egreedy = torch.rand(1)[0]

        if random_for_egreedy > epsilon:

            with torch.no_grad():

                state = Tensor(state).to(device)
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

        state = Tensor(state).to(device)
        new_state = Tensor(new_state).to(device)
        reward = Tensor(reward).to(device)
        action = LongTensor(action).to(device)
        done = Tensor(done).to(device)

        new_state_values = self.nn(new_state).detach()
        max_new_state_values = torch.max(new_state_values, 1)[0]
        target_value = reward + (1 - done) * gamma * max_new_state_values

        predicted_value = self.nn(state).gather(1, action.unsqueeze(1)).squeeze(1)

        loss = self.loss_func(predicted_value, target_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


memory = ExperienceReplay(replay_mem_size)
qnet_agent = QNet_Agent()

steps_total = []

frames_total = 0

for i_episode in range(num_episodes):

    state = gamePlayer.reset()

    step = 0
    # for step in range(100):
    while True:

        step += 1
        frames_total += 1

        epsilon = calculate_epsilon(frames_total)

        # action = env.action_space.sample()
        action = qnet_agent.select_action(state, epsilon)

        new_state, reward, done = gamePlayer.step(action, step)

        memory.push(state, action, new_state, reward, done)

        qnet_agent.optimize()

        state = new_state

        if done:
            steps_total.append(step)
            print('Episode finished after %i steps' % step)

            break

print(env.structure)
print('FIGURE OF MERIT IS:', env.evaluate())

print('figure of Merit is: ', env.evaluate(plot=1, plotDir='/home/hosseinoj/Desktop/simulations/01/'))
