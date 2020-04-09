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


structure = np.zeros(25).reshape(5, 5)

goal_structure = np.ones(25).reshape(5, 5)

right, left, up, down, flip = 0, 1, 2, 3, 4

x_threshold = 4
y_threshold = 4
x_min = 0
y_min = 0


class Player:

    def __init__(self):
        self.x = 0
        self.y = 0

        self.max_time_step = 50
        self.time_step = 0
        self.reward_list = []
        self.sum_reward_list = []
        self.sum_rewards = []


        self.action_space = spaces.Discrete(5)
        self.observation_space = 27

    def get_done(self, time_step):

        if time_step == self.max_time_step:
            done = True

        else:
            done = False

        return done

    def flip_pixel(self):

        if structure[self.x][self.y] == 1:
            structure[self.x][self.y] = 0.0

        elif structure[self.x][self.y] == 0:
            structure[self.x][self.y] = 1

    def get_reward(self):

        if structure[self.x][self.y] == 1:
            reward = 1

        elif structure[self.x][self.y] == 0.0:
            reward = 0

        return reward

    def step(self, action, time_step):


        if action == right:

            if self.y < y_threshold:
                self.y = self.y + 1
            else:
                self.y = y_threshold

        if action == left:

            if self.y > y_min:
                self.y = self.y - 1
            else:
                self.y = y_min

        if action == up:

            if self.x > x_min:
                self.x = self.x - 1
            else:
                self.x = x_min

        if action == down:

            if self.x < x_threshold:
                self.x = self.x + 1
            else:
                self.x = x_threshold

        if action == flip:
            self.flip_pixel()

        reward = self.get_reward()
        self.reward_list.append(reward)
        done = self.get_done(time_step)

        reshaped_structure = np.reshape(structure, (1, 25))
        reshaped_state = np.append(reshaped_structure, (np.float64(self.x / 4), np.float64(self.y / 4)))
        state = reshaped_state

        return state, reward, done

    def reset(self):

        structure = np.zeros(25).reshape(5, 5)
        reset_reshaped_structure = np.reshape(structure, (1, 25))
        reset_reshaped_state = np.append(reset_reshaped_structure, (0, 0))
        state = reset_reshaped_state

        self.x = 0
        self.y = 0
        self.reward_list = []

        return state


################################################

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


######## Creating Instance #######                
gamePlayer = Player()
##################################

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
Tensor = torch.Tensor
LongTensor = torch.LongTensor

##### PARAMS #####################
learning_rate = 0.01
num_episodes = 30000
gamma = 0.999

hidden_layer1 = 128
hidden_layer2 = 64

replay_mem_size = 50000
batch_size = 32

egreedy = 0.9
egreedy_final = 0.02
egreedy_decay = 30000


##################################

def calculate_epsilon(steps_done):
    epsilon = egreedy_final + (egreedy - egreedy_final) * \
              math.exp(-1 * steps_done / egreedy_decay)

    return epsilon


####################################################
number_of_inputs = gamePlayer.observation_space
number_of_outputs = gamePlayer.action_space.n


####################################################

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


######################################################

class QNet_Agent():

    def __init__(self):
        self.nn = NeuralNetwork().to(device)
        self.loss_func = nn.MSELoss()
        self.optimizer = optim.Adam(params=self.nn.parameters(), lr=learning_rate)

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
            import_data_to_csv('Check DQN Rewards Track.csv', sum(gamePlayer.reward_list), i_episode,
                                       len(gamePlayer.reward_list))

            # gamePlayer.reset_max_time_steps()

            episodes_total += 1
            end = time.time()

            print('########################################')
            print('Episode finished after %i steps' % time_step)
            print('Sofar %i episodes have been completed:' % episodes_total)
            print('elapsed time was ( %d ) seconds: ' % (end - start))
            print('########################################')


            break

print( structure)
plt.figure(figsize=(12, 5))
plt.title("Rewards per Episode")
plt.bar(torch.arange(episodes_total), gamePlayer.sum_rewards, alpha=0.6, color='green', width=1)
plt.show()

