
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
import csv
from csv import writer

########################### IMPORTING SIMFRAME ENVIRONMENT ###########################


dims = [150, 100, 1]
env = Environment(dimensions=dims,
                  designArea=[6, 6],  # the area that is to be changed pixel wise
                  structureScalingFactor=2,
                  # pixels in designArea will be scaled up by this factor for the simulation.
                  waveguides=[[(0, 44), (dims[0] / 2, 56)], [(dims[0] / 2, 44), (dims[0], 56)]],
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
                                'modeNum': 0}])  # array of output Modes. modeNum 4 is TE20 in 2D

# Setting local simulation
os.environ["SIMULATE_ON_THIS_MACHINE"] = "1"
producePlots = 0

# Setting initial structure

env.setStructure(np.zeros(36).reshape(6, 6))


def figureOfMerit(overlaps):
    return overlaps[0][0]


env.setFOM(figureOfMerit)

######################### EXTENDING THE ENVIRONMENT WITH DEFINING THE ACTION TAKER #################################

x_threshold = 5
y_threshold = 5
x_min = 0
y_min = 0

UP, FLIP_UP, DOWN, FLIP_DOWN, LEFT, FLIP_LEFT, RIGHT, FLIP_RIGHT = 0, 1, 2, 3, 4, 5, 6, 7


'''note that x,y are not actual coordinates! x is the number of rows and y is the 
number of columns.'''

with open('Track Rewards.csv', 'wb') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)


def import_data_to_csv(file_name, sum_rewards, episode, loop_moves, loop_penalty, steps_taken):
    with open(file_name, 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)

        csv_writer.writerow(['The Agent Got ( %f ) Penalties:' % loop_penalty])

        csv_writer.writerow(['The Agent Stuck ( %i ) Times in the Loop:' % loop_moves])

        csv_writer.writerow(['The Agent Took ( %i ) Steps:' % steps_taken])

        csv_writer.writerow(['The Sum of Rewards in Episode ( %i ) Was:' % episode])
        csv_writer.writerow([sum_rewards])

        csv_writer.writerow(['----------------------------'])


def structure_to_csv(file_name, matrix, efficiency):
    with open(file_name, 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)

        for i in range(len(matrix)):
            csv_writer.writerow(matrix[i])
        csv_writer.writerow(['The efficiency was: %.3f' %efficiency])
        csv_writer.writerow(['----------------------------'])

class Player:

    def __init__(self, environment: Environment):

        self.environment = environment

        # self.state = env.structure

        self.x = 0
        self.y = 0
        self.base_efficiency = 0.51
        self.updated_efficiency = 0
        self.efficiency_to_save = 0.6   # minimum efficiency to save the structure (increases by 10% each time)

        self.max_time_steps = 200    # maximum steps the agent can take in each episode unless it does very good

        self.reward_list = []        # gathers the reward in each episode and then resets in the new episode

        self.printer_counter = 0     # Used for saving the structures and animations

        self.sum_rewards = []        # gathers the SUM of rewards in each episode and appends

        self.penalty_for_loop = 0    # counts the sum of penalties the agent got for stucking in a loop

        self.count_loop_moves = 0   # checks how many times the agent stucks in a loop

        self.action_space = spaces.Discrete(8)

    def scale_reward(self, eff, diff):

        if diff >= 0:
            output = (-1 + math.exp((eff*100)*diff))

        else:
            output = 0

        return output

    def get_reward(self):

        efficiency_evaluation = env.evaluate()
        efficiency_difference = (efficiency_evaluation - self.base_efficiency)

        reward = np.around(self.scale_reward(efficiency_evaluation, efficiency_difference), decimals=3)

        if len(self.reward_list) > 5:

            if self.reward_list[len(self.reward_list) - 1] == self.reward_list[len(self.reward_list) - 3] and \
                    self.reward_list[len(self.reward_list) - 2] == self.reward_list[len(self.reward_list) - 4] and \
                    self.reward_list[len(self.reward_list) - 1] != self.reward_list[len(self.reward_list) - 2] and \
                    self.reward_list[len(self.reward_list) - 3] != self.reward_list[len(self.reward_list) - 4]:

                reward = -2.2 * (self.reward_list[len(self.reward_list) - 1] + self.reward_list[len(self.reward_list) - 2])
                self.penalty_for_loop += reward

                self.count_loop_moves += 1




        self.base_efficiency = efficiency_evaluation

        self.updated_efficiency = np.around(efficiency_evaluation, decimals=3)



        print(' ##### THE REWARD WAS ##### : ', reward)

        return reward

    def get_done(self, time_step):

        if time_step < self.max_time_steps:
            done = False

        elif time_step == self.max_time_steps:
            if sum(self.reward_list[-20:]) < 10 or self.max_time_steps == 350:
                done = True
            else:
                done = False
                self.max_time_steps += 30

        return done

    def reset_max_time_steps(self):
        self.max_time_steps = 200

    def step(self, action, time_step):

        assert self.action_space.contains(action)

        reward = 0

        if action == RIGHT:
            if self.y < y_threshold:
                self.y = self.y + 1
            else:
                self.y = y_threshold


        if action == FLIP_RIGHT:
            env.flipPixel([self.x, self.y])

            if self.y < y_threshold:
                self.y = self.y + 1
            else:
                self.y = y_threshold

        elif action == LEFT:
            if self.y > y_min:
                self.y = self.y - 1
            else:
                self.y = y_min

        elif action == FLIP_LEFT:
            env.flipPixel([self.x, self.y])

            if self.y > y_min:
                self.y = self.y - 1
            else:
                self.y = y_min

        elif action == UP:
            if self.x > x_min:
                self.x = self.x - 1
            else:
                self.x = x_min

        elif action == FLIP_UP:
            env.flipPixel([self.x, self.y])

            if self.x > x_min:
                self.x = self.x - 1
            else:
                self.x = x_min

        elif action == DOWN:
            if self.x < x_threshold:
                self.x = self.x + 1
            else:
                self.x = x_threshold

        elif action == FLIP_DOWN:
            env.flipPixel([self.x, self.y])

            if self.x < x_threshold:
                self.x = self.x + 1
            else:
                self.x = x_threshold


        # print('FIGURE OF MERIT IS:', env.evaluate())
        reward += self.get_reward()
        self.reward_list.append(reward)

        done = self.get_done(time_step)

        # Reshaping the state from a nxn matrix to a 1xn^2 matrix,
        # AND Appending the position of the action taker to the states:

        reshaped_structure = np.reshape(env.structure, (1, 36))
        reshaped_state = np.append(reshaped_structure, (self.x/5, self.y/5))

        state = reshaped_state

        if self.updated_efficiency > self.efficiency_to_save:
            self.efficiency_to_save += 0.05
            structure_to_csv('structure.csv', env.structure.astype(int), self.updated_efficiency)
            self.printer_counter += 1
            print('figure of Merit is: ', env.evaluate(plot=1, plotDir='/home/hosseinoj/Desktop/simulations/%.2f/' % self.updated_efficiency))

        print('CURRENT POSITION OF AGENT: ', (self.x, self.y))  # This part is for control
        print(env.structure)  # This part is for control
        print('EFFICIENCY IS:', self.updated_efficiency)  # This part is for control

        return state, reward, done

    def reset(self):

        env.resetStructure()  # resets the Environment to all silicon ( with values = 1 )
        reset_reshaped_structure = np.reshape(env.structure, (1, 36))
        reset_reshaped_state = np.append(reset_reshaped_structure, (0, 0))

        self.x = 0
        self.y = 0

        state = reset_reshaped_state

        self.reward_list = []
        self.base_efficiency = 0.51
        self.count_loop_moves = 0
        self.penalty_for_loop = 0

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
num_episodes = 300
gamma = 0.999

hidden_layer1 = 128
hidden_layer2 = 64

replay_mem_size = 50000
batch_size = 32

egreedy = 0.9
egreedy_final = 0.02
egreedy_decay = 30000
##################

###### DEFINING THE EXPLORATION/EXPLOITATION TRADE OFF FUNCTION ##############
def calculate_epsilon(steps_done):
    epsilon = egreedy_final + (egreedy - egreedy_final) * \
              math.exp(-1 * steps_done / egreedy_decay)

    return epsilon


number_of_inputs = 38
number_of_outputs = gamePlayer.action_space.n


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(number_of_inputs, hidden_layer1)
        self.linear2 = nn.Linear(hidden_layer1, hidden_layer2)
        self.linear3 = nn.Linear(hidden_layer2, number_of_outputs)

        self.activation = nn.Tanh()
        # self.activation = nn.ReLU()

    def forward(self, x):
        output1 = self.linear1(x)
        output1 = self.activation(output1)
        output2 = self.linear2(output1)
        output2 = self.activation(output2)
        output3 = self.linear3(output2)

        return output3


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

episodes_total = 0

frames_total = 0

start = time.time()
for i_episode in range(num_episodes):

    state = gamePlayer.reset()

    time_step = 0
    # for step in range(100):
    while True:

        time_step += 1
        frames_total += 1

        epsilon = calculate_epsilon(frames_total)

        # action = env.action_space.sample()
        action = qnet_agent.select_action(state, epsilon)

        new_state, reward, done = gamePlayer.step(action, time_step)

        memory.push(state, action, new_state, reward, done)

        qnet_agent.optimize()

        state = new_state

        if done:
            steps_total.append(time_step)

            gamePlayer.sum_rewards.append(sum(gamePlayer.reward_list))
            import_data_to_csv('Track Rewards.csv', sum(gamePlayer.reward_list), i_episode,
                                       gamePlayer.count_loop_moves, gamePlayer.penalty_for_loop,
                                       len(gamePlayer.reward_list))

            gamePlayer.reset_max_time_steps()

            episodes_total += 1
            end = time.time()

            print('########################################')
            print('Episode finished after %i steps' % time_step)
            print('Sofar %i episodes have been completed:' % episodes_total)
            print('elapsed time was ( %d ) seconds: ' %(end-start))
            print('########################################')

            break

plt.figure(figsize=(12,5))
plt.title("Rewards per Episode")
plt.bar(torch.arange(len(steps_total)), steps_total, alpha=0.6, color='green', width=1)
plt.show()

