# ModeConverter-RL


The objective of this project is to design a mode converter which inputs the mode TE00 and outputs TE20. In this Project we use Reinforcement learning algorithms to come up with a design that can convert the modes with the best efficiency possible. In the beginning a single wafer with a homogenous layer of silicon-nitride on top is assumed. This wafer contains pixels and the number of pixels based on their size is adjustable. We can flip these pixels and replace each silicon-nitride pixel with air.
The simulation environment is created by Marco Butz in University of Muenster. It can be found in the Gitlab of University of Muenster and be accessed by creating a conda .yml file, containing the needed dependecies.

The Goal is to come up with a configuration, which enhances the efficiency of mode conversion as much as possible.

Taking the main sturcture of the reinforcement learning algorithms aside, there are a couple of very important blocks of the code which are crucial for the agent to learn. If they are implemented poorly, the agent wont learn effectivley or wont learn at all! one of which is debated to be the most important part, is the reward function. How to shape this function in order to give the agent the ability to learn is without any doubt of great concern and needs creativity along with knowing the problem deeply. The other important block of code is termination function. The termination function which in the end gives a "True" or "False" value is in charge of terminating an episode. Of course the program can be designed in a way that the episode dont terminate at all. These kind of algorithms are knwon to be non-episodic or continous tasks. In our case, setting a termination function can help the agent to learn but it should be implemented in a smart way in order to help the agent learn better.

In course of time, Hopefuly various methods  will be used. Each time the essential notes about the codes will be updated in this README file. BUCKLE UP!




