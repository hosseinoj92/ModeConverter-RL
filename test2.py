
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
                  designArea=[10, 10],  # the area that is to be changed pixel wise
                  structureScalingFactor=6,
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

env.setStructure(np.ones(100).reshape(10, 10))


def figureOfMerit(overlaps):
    return overlaps[0][0]


env.setFOM(figureOfMerit)

######################### EXTENDING THE ENVIRONMENT WITH DEFINING THE ACTION TAKER #################################

x_threshold = 9
y_threshold = 9
x_min = 0
y_min = 0

UP, DOWN, LEFT, RIGHT, FLIP = 0, 1, 2, 3, 4


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
            reward = (efficiency_evaluation - self.base_efficiency) * 100
            self.base_efficiency = efficiency_evaluation
            print('YOHOO I GOT A REWARD! : ', reward)

        else:
            reward = -0.1

        return reward

    def get_done(self, step):

        if step == 5:
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
                self.y = y_threshold

        elif action == LEFT:
            if self.y > y_min:
                self.y = self.y - 1
            else:
                self.y = y_min

        elif action == UP:
            if self.x > x_min:
                self.x = self.x - 1
            else:
                self.x = x_min

        elif action == DOWN:
            if self.x < x_threshold:
                self.x = self.x + 1
            else:
                self.x = x_threshold

        elif action == FLIP:
            env.flipPixel([self.x, self.y])

        # print('FIGURE OF MERIT IS:', env.evaluate())
        reward += self.get_reward()

        done = self.get_done(step)

        # Reshaping the state from a nxn matrix to a 1xn^2 matrix,
        # AND Appending the position of the action taker to the states:

        reshaped_structure = np.reshape(env.structure, (1, 100))
        reshaped_state = np.append(reshaped_structure, (self.x, self.y))

        state = reshaped_state

        if self.updated_efficiency >= 0.4:
            np.savetxt("Structure.csv", env.structure, delimiter=",",fmt='%d')

        print('CURRENT POSITION OF AGENT: ', (self.x / 10, self.y / 10))  # This part is for control
        print(env.structure)  # This part is for control
        print('EFFICIENCY IS:', self.updated_efficiency)  # This part is for control

        return state, reward, done

    def reset(self):

        env.resetStructure()  # resets the Environment to all silicon ( with values = 1 )
        reset_reshaped_structure = np.reshape(env.structure, (1, 100))
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
num_episodes = 1000
gamma = 0.999

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


number_of_inputs = 102
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

episodes_total = 0

frames_total = 0

start = time.time()
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
            episodes_total += 1
            end = time.time()
            print('########################################')
            print('Episode finished after %i steps' % step)
            print('Sofar %i episodes have been completed:' % episodes_total)
            print('elapsed time was ( %d ) seconds: ' %(end-start))
            print('########################################')

            break

print(env.structure)
print('FIGURE OF MERIT IS:', env.evaluate())

print('figure of Merit is: ', env.evaluate(plot=1, plotDir='/home/hosseinoj/Desktop/simulations/01/'))
