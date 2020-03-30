# ModeConverter-RL


The objective of this project is to design a mode converter which inputs the mode TE00 and outputs TE20. In this Project we use Reinforcement learning algorithms to come up with a design that can convert the modes with the best efficiency possible. In the beginning a single wafer with a homogenous layer of silicon-nitride on top is assumed. This wafer contains pixels and the number of pixels based on their size is adjustable. We can flip these pixels and replace each silicon-nitride pixel with air.
The simulation environment is created by Marco Butz in University of Muenster. It can be found in the Gitlab of University of Muenster and be accessed by creating a conda .yml file, containing the needed dependecies.

The Goal is to come up with a configuration, which enhances the efficiency of mode conversion as much as possible.

Taking the main sturcture of the reinforcement learning algorithms aside, there are a couple of very important blocks of the code which are crucial for the agent to learn. If they are implemented poorly, the agent wont learn effectivley or wont learn at all! one of which is debated to be the most important part, is the reward function. How to shape this function in order to give the agent the ability to learn is without any doubt of great concern and needs creativity along with knowing the problem deeply. The other important block of code is termination function. The termination function which in the end gives a "True" or "False" value is in charge of terminating an episode. Of course the program can be designed in a way that the episode dont terminate at all. These kind of algorithms are knwon to be non-episodic or continous tasks. In our case, setting a termination function can help the agent to learn but it should be implemented in a smart way in order to help the agent learn better.

In course of time, Hopefuly various methods  will be used. Each time the essential notes about the codes will be updated in this README file. BUCKLE UP!

# DQN-Experience Replay

THE MOVEMENT (step(action,step)):
-------------------------------------
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

THE REWARD FUNCTION (get_reward()):
-------------------------------------
base efficiency, when no pixels are flipped is about 8%. It can be calculated via "env.evaluate()". 
In each time step the efficieny is observed and compared to the previous time step. If we have an increse in efficiency, 
the "reward = 1" is given to the agent and the base efficiency is updated to current efficiency. If not the "reward = 0"

THE DONE FUNCTION (get_done()):
---------------------------------
This function determines when the episode should be terminated. In this version we consider the episode done, when
200 steps is taken.

THE RESET FUNCTION (reset()):
--------------------------
Using the mother function "env.resetStructure()", we can reset all the values of structure to 1. then reshape the state
and append the position (0,0) which indicates top left corner to the reshaped state. set the value of efficiency and
setting the base efficiency back to 8% again.

# DQN 5x5

Here the size of the frame is reduced or better to say, the size of pixels increased fourfold. also 2 little changes has been applied. first, instead of a zero reward for not increasing the efficiency, a negative reward or penalty is given to the agent, the reward is not just one +1 if it increases efficiency in a step, rather the difference of the efficiency is taken into account reward. For making it more conveniant, the difference value is multiplied by 100.

Another thing which has been changed is the movement. So if the agent hits the wall, it doesnt start from the beginning of the axis. it just stays there in the same place.

# DQN 10x10

In this try a couple of things have changed.

(a): REWARD FUNCTION:

i: The reward function not only checks the previous reward to decide whether to give a reward or not (which is the difference of two rewards. but also checks if the new updated reward is at least 2% greater than the previous reward. in The previous try, even if the difference was 0.000001, it stil concidered a reward.
    
ii: There is a little negative reward given as a penalty to the agent if it takes a bad action. This might help because the agent always look forward to maximize the cumulative rewards, so it might be helpful if it get a penalty, so it avoids the wrong moves more.
    
 (b):TERMINATION FUNCTION:
 
i: The get_done() function doesnt terminate the episode explicitly after a certain number of episodes. It takes for example 100 steps without termination, and then in the 100th episode, it checks the sum of last 20 episodes. If its less than a certain number, it will terminate the episode, but if it achives good result, it will be allowed to take an extra 10 steps to explore more!
    
(c): SAVING THE GRAPH AND KEEPING RECORDS:

i: There is also a part implemented, which the agent reaches 40% of efficiency, The structure will be saved in a csv file and the simulation animations will be saved in another file. in each time it saves, the goal efficiency will be increased by 10%. so next time it should reache 50%, 60% and so on.
