##################################################################################################
####################### CREATING THE PLAYER CLASS FOR INTERACTING WITH ENVIRONMENT#################
###################################################################################################

from simFrame.environment import Environment
import numpy as np
from gym import spaces
from utility import *
import hashlib

class Player:

    def __init__(self, environmentConfig: dict, figureOfMerit, hashFilePath="hashes.pkl"):

        self.env = Environment(**environmentConfig)
        ################# SETTING INITIAL STRUCTURE ################

        self.env.setStructure(np.zeros(self.env.designArea[0]*self.env.designArea[1]).reshape(self.env.designArea[0], self.env.designArea[1]))

        self.env.setFOM(figureOfMerit)

        self.hashFilePath = hashFilePath
        self.knownResults = load_obj(hashFilePath)

        self.x = 0
        self.y = 0

        #MARCO
        #set this dynamically by calling env.evaluate on the initial structure
        self.base_efficiency = self.evaluateWithHash()
        self.updated_efficiency = 0.50

        self.efficiency_to_save = 0.6   # minimum efficiency to save the structure (increases by 10% each time)

        self.max_time_steps = 200    # maximum steps the agent can take in each episode unless it does very good

        self.reward_list = []        # gathers the reward in each episode and then resets in the new episode
        self.printer_counter = 0     # Used for saving the structures and animations
        self.sum_rewards = []        # gathers the SUM of rewards in each episode and appends
        self.penalty_for_loop = 0    # counts the sum of penalties the agent got for sticking in a loop
        self.all_flipped = 0          # increases by one every time the whole structure is filled with value 1
        self.convergence_control = []  # collects efficiency at the end of each episode

        self.flipped_or_not = []       # by flipping, value 1 is added, otherwise 0. Its for checking if the agent is in a loop
        self.loop_counter = 0
        self.loop_list = []

        self.efficiency_list = []       # collects efficiency in each step. in order to track the mean eff in episodes
        self.mean_efficiency = []

        self.action_space = spaces.Discrete(5)
        self.observation_space = 2 * self.env.designArea[0] * self.env.designArea[1]

        self.position_state = np.zeros((self.env.designArea[0], self.env.designArea[1]))


    def get_done(self, time_step):

        #MARCO
        #use self.environment to consistently use the players instance of the environment
        if time_step == self.max_time_steps or self.env.structure.all():
            done = True

        else:
            done = False

        return done

    def step(self, action, time_step):

        right, left, up, down, flip = 0, 1, 2, 3, 4

        reward = 0

        if action == right:

            if self.y < self.env.designArea[1]-1:
                self.y = self.y + 1

            self.flipped_or_not.append(0)

        if action == left:

            if self.y > 0:
                self.y = self.y - 1

            self.flipped_or_not.append(0)

        if action == up:

            if self.x > 0:
                self.x = self.x - 1

            self.flipped_or_not.append(0)

        if action == down:

            if self.x < self.env.designArea[0]-1:
                self.x = self.x + 1

            self.flipped_or_not.append(0)

        if action == flip:

            self.env.flipPixel([self.x, self.y])

            results = self.evaluateWithHash()

            self.updated_efficiency = results

            self.flipped_or_not.append(1)

            if self.updated_efficiency > self.base_efficiency:

                reward = self.updated_efficiency*10
                self.base_efficiency = self.updated_efficiency

            print(self.env.structure)  # This part is for control
            print('EFFICIENCY IS:', self.updated_efficiency)  # This part is for control

        self.efficiency_list.append(self.updated_efficiency)

        self.position_state = np.zeros((self.env.designArea[0], self.env.designArea[1]))
        self.position_state[self.x][self.y] = 1

        if len(self.flipped_or_not) >= 3:

            if self.flipped_or_not[len(self.flipped_or_not) - 1] == self.flipped_or_not[len(self.flipped_or_not) - 2] == \
                self.flipped_or_not[len(self.flipped_or_not) - 3] == 1:

                self.loop_counter += 1

        self.reward_list.append(reward)
        done = self.get_done(time_step)

        if self.env.structure.all():
            reward += 60
            self.all_flipped += 1

        # Reshaping the state from a nxn matrix to a 1xn^2 matrix,
        # AND Appending the position of the action taker to the states:

        reshaped_structure = np.reshape(self.env.structure, (1, self.env.designArea[0]*self.env.designArea[1]))
        reshaped_position = np.reshape(self.position_state, (1, self.env.designArea[0]*self.env.designArea[1]))
        reshaped_state = np.append(reshaped_structure, reshaped_position)

        state = reshaped_state

        #if self.updated_efficiency > self.efficiency_to_save and self.updated_efficiency != self.base_efficiency:

            #structure_to_csv('structure.csv', self.env.structure.astype(int), self.updated_efficiency)

            #print('figure of Merit is: ', self.env.evaluate(plot=1, plotDir='simulations/%.3f/'
                                                                       #% float(self.updated_efficiency+self.printer_counter)))
            #self.printer_counter += 1
            #self.efficiency_to_save += 0.02

        return state, reward, done

    def evaluateWithHash(self):
        structureCheckSum = hashlib.sha224(self.env.structure.reshape(1, self.env.designArea[0] * self.env.designArea[1])[0]).hexdigest()

        if structureCheckSum in self.knownResults:
            print('using hash database: ', structureCheckSum)
            results = self.knownResults[structureCheckSum]
        else:
            print('simulating new structure: ', structureCheckSum)
            results = self.env.evaluate()
            self.knownResults.update({structureCheckSum: results})
            # Save the hashes every now and then
            if len(self.knownResults) % 100 == 0:
                save_obj(self.knownResults, self.hashFilePath)

        return results

    def reset(self):

        print("reseting...")
        self.env.resetStructure()  # resets the Environment to all air ( with values = 0 )
        self.position_state = np.zeros((self.env.designArea[0], self.env.designArea[1]))

        self.position_state[0, 0] = 1

        reset_reshaped_structure = np.reshape(self.env.structure, (1, self.env.designArea[0]*self.env.designArea[1]))
        reset_reshaped_position = np.reshape(self.position_state, (1, self.env.designArea[0]*self.env.designArea[1]))
        reset_reshaped_state = np.append(reset_reshaped_structure, reset_reshaped_position)

        state = reset_reshaped_state

        self.x = 0
        self.y = 0

        self.reward_list = []

        self.base_efficiency = self.evaluateWithHash()
        self.updated_efficiency = 0

        self.count_loop_moves = 0
        self.penalty_for_loop = 0

        self.flipped_or_not = []
        self.loop_counter = 0

        self.efficiency_list = []

        return state


