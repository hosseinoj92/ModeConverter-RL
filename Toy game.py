import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import csv
from csv import writer
import math
from gym import spaces
import random
import sys
import pickle
import hashlib
import os

ret = sys.exit

def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    if os.path.exists(name):
        with open(name, 'rb') as f:
            return pickle.load(f)
    else:
        return {}

knownResults = load_obj("hashes.pkl")

with open('Check DQN Rewards Track.csv', 'wb') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)


def import_data_to_csv(file_name, sum_rewards, episode, steps_taken):
    with open(file_name, 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)

        csv_writer.writerow(['The Agent Took ( %i ) Steps:' % steps_taken])

        csv_writer.writerow(['The Sum of Rewards in Episode ( %i ) Was:' % episode])
        csv_writer.writerow([sum_rewards])

        csv_writer.writerow(['----------------------------'])


n=4

# structure = np.zeros(25).reshape(5, 5)
np.zeros((n, n))

# goal_structure = np.ones(25).reshape(5, 5)
# goal_structure = np.ones((5, 5))

right, left, up, down, flip = 0, 1, 2, 3, 4

x_max = n-1
y_max = n-1
x_min = 0
y_min = 0


class Player:

    def __init__(self):
        self.x = 0
        self.y = 0

        self.max_time = 150
        self.time_step = 0
        self.reward_list = []
        self.sum_rewards = []
        # self.gather_positions = []
        self.gather_values = []      # gathers 0s and 1s
        self.sum_gather_values = []

        self.position_state = np.zeros((n, n))
        self.structure_state = np.zeros((n, n))

        self.action_space = spaces.Discrete(5)
        self.observation_space = n*n*2

    def get_done(self, time_step):

        if time_step == self.max_time or self.structure_state.all():
            done = True
        else:
            done = False

        return done

    def flip_pixel(self):

        self.structure_state[self.x, self.y] = 1 - self.structure_state[self.x, self.y]

        # if self.structure_state[self.x][self.y] == 1:
        #     self.structure_state[self.x][self.y] = 0.0
        #
        # elif self.structure_state[self.x][self.y] == 0:
        #     self.structure_state[self.x][self.y] = 1

    def step(self, action, time_step):

        # self.position_state = np.zeros((n, n))
        reward = -1

        if action == right:

            if self.y < y_max:
                self.y = self.y + 1

        if action == left:

            if self.y > y_min:
                self.y = self.y - 1

        if action == up:

            if self.x > x_min:
                self.x = self.x - 1

        if action == down:

            if self.x < x_max:
                self.x = self.x + 1

        if action == flip:
            self.flip_pixel()

            if self.structure_state[self.x, self.y] == 1:
                reward += 5
                self.gather_values.append(1)
            else:
                reward += -5
                self.gather_values.append(-1)

        self.position_state = np.zeros((n, n))
        self.position_state[self.x][self.y] = 1

        self.reward_list.append(reward)

        done = self.get_done(time_step)

        if self.structure_state.all():
            reward += 20

        reshaped_structure = np.reshape(self.structure_state, (1, n**2))
        reshaped_position = np.reshape(self.position_state, (1, n**2))
        reshaped_state = np.append(reshaped_structure, reshaped_position)

        state = reshaped_state

        return state, reward, done

    def reset(self):

        self.structure_state= np.zeros((n, n))
        self.position_state = np.zeros((n, n))

        self.position_state[0, 0] = 1

        reset_reshaped_structure = np.reshape(self.structure_state, (1, n**2))
        reset_reshaped_position = np.reshape(self.position_state, (1, n**2))
        reset_reshaped_state = np.append(reset_reshaped_structure, reset_reshaped_position)

        state = reset_reshaped_state

        self.x = 0
        self.y = 0
        self.reward_list = []
        self.gather_values = []
        # self.gather_positions = []

        return state

################################################

class ExperienceReplay(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, new_state, reward, done):          # Used for adding data to the ER
        transition = (state, action, new_state, reward, done)

        if self.position >= len(self.memory):
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):                                  # Used for taking sample data from Batch
        return zip(*random.sample(self.memory, batch_size))

    def __len__(self):                                            # Used for calculating current memory size
        return len(self.memory)


######## Creating Instance #######
gamePlayer = Player()
##################################

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
Tensor = torch.Tensor
LongTensor = torch.LongTensor

##### PARAMS #####################
learning_rate = 0.0001
num_episodes = 10000
gamma = 0.99

hidden_layer1 = 64
#hidden_layer2 = 256

replay_mem_size = 1000000
batch_size = 32

update_target_frequency = 10000

egreedy_initial = 1.0
egreedy_final = 0.1
egreedy_decay = 500000

clip_error = True
##################################

def calculate_epsilon(steps_done):
    epsilon = egreedy_final + (egreedy_initial - egreedy_final) * math.exp(-1 * steps_done / egreedy_decay)
    return epsilon

####################################################

number_of_inputs = gamePlayer.observation_space
number_of_outputs = gamePlayer.action_space.n

# print(number_of_outputs)
#
# ret()

####################################################

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(number_of_inputs, hidden_layer1)
        self.linear2 = nn.Linear(hidden_layer1, number_of_outputs)

        #self.activation = nn.Tanh()
        self.activation = nn.ReLU()

    def forward(self, x):
        output1 = self.linear1(x)
        output1 = self.activation(output1)
        output2 = self.linear2(output1)

        return output2


######################################################

class QNet_Agent():

    def __init__(self):
        self.nn = NeuralNetwork().to(device)  # first neural net
        self.target_nn = NeuralNetwork().to(device)

        self.loss_func = nn.MSELoss()
        self.optimizer = optim.Adam(params=self.nn.parameters(), lr=learning_rate)

        self.update_target_counter = 0

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
        reward = Tensor(reward).to(device)
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

        if self.update_target_counter % update_target_frequency == 0:
            self.target_nn.load_state_dict(self.nn.state_dict())

        self.update_target_counter += 1




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

        # print(action)

        memory.push(state, action, new_state, reward, done)
        qnet_agent.optimize()

        state = new_state
        cum_reward += reward

        # print(action)
        # print()
        # print(state[0:25].reshape(5, 5))
        # print()
        # print(state[25:].reshape(5, 5))
        # print()
        # print(reward)
        # print()
        #
        # time.sleep(1)

        if done:
            steps_total.append(time_step)

            gamePlayer.sum_rewards.append(sum(gamePlayer.reward_list))
            gamePlayer.sum_gather_values.append(sum(gamePlayer.gather_values))

            import_data_to_csv('Check DQN Rewards Track.csv', sum(gamePlayer.reward_list), i_episode,
                                       len(gamePlayer.reward_list))

            # gamePlayer.reset_max_time_steps()

            episodes_total += 1
            end = time.time()

            if episodes_total % 1 == 0:

                print('########################################')
                print('Episode {} finished after {} steps'.format(episodes_total,time_step))
                print('Rewards collected in the episode %0.3f' % cum_reward)
                print('Exploration variable is %0.3f' % epsilon)
                # print('Sofar %i episodes have been completed:' % episodes_total)
                # print('elapsed time was ( %d ) seconds: ' % (end - start))
                print('########################################\n')

            break


plt.figure(figsize=(12, 5))
plt.title("Rewards per Episode")
plt.bar(torch.arange(episodes_total), gamePlayer.sum_rewards, alpha=0.6, color='green', width=1)
plt.show()

plt.figure(figsize=(12, 5))
plt.title("Rewards per Episode")
plt.bar(torch.arange(episodes_total), gamePlayer.sum_gather_values, alpha=0.6, color='blue', width=1)
plt.show()


