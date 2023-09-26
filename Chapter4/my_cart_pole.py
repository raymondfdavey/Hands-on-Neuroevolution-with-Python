
import gymnasium as gym
import math
# The Python standard library import
import os
# The NEAT-Python library imports
import neat
# The helper used to visualize experiment results
import visualize
import numpy as np
import itertools
import os

from neat import nn, population, statistics

np.set_printoptions(threshold=np.inf)
env = gym.make('CartPole-v0')

# run through the population


def eval_fitness(genomes):
	for g in genomes:
		observation = env.reset()
		env.render()
		net = nn.create_feed_forward_phenotype(g)
		fitness = 0
		while 1:
			inputs = observation

			# active neurons
			output = net.serial_activate(inputs)
			if (output[0] >= 0):
				observation, reward, done, info = env.step(1)
			else:
				observation, reward, done, info = env.step(0)

			fitness += reward

			env.render()
			if done:
				print(fitness)
				env.reset()
				break
		# evaluate the fitness
		g.fitness = fitness

local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'cartPole_config')

pop = population.Population(config_path)
pop.run(eval_fitness, 300)

env.monitor.start('cartpole-experiment/', force=True)
winner = pop.statistics.best_genome()

streak = 0
winningnet = nn.create_feed_forward_phenotype(winner)

observation = env.reset()
env.render()
while streak < 100:
	fitness = 0
	frames = 0
	while 1:
		inputs = observation

		# active neurons
		output = winningnet.serial_activate(inputs)
		if (output[0] >= 0):
			observation, reward, done, info = env.step(1)
		else:
			observation, reward, done, info = env.step(0)

		fitness += reward

		env.render()
		frames += 1
		if frames >= 200:
			done = True
		if done:
			if fitness >= 195:
				print ('streak: ', streak)
				streak += 1
			else:
				print(fitness)
				print('streak: ', streak)
				streak = 0
			env.reset()
			break
print("completed!")
env.monitor.close()
gym.upload('cartpole-experiment/', api_key='XXX')