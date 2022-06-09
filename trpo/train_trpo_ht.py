import torch
import gym
import argparse
import sys
sys.path.append('../')
from env.custom_hopper import *
from sb3_contrib import TRPO

from sklearn.model_selection import ParameterGrid
import numpy as np


def main():
	
	# Setting the parametrs for the hyperparameters tuning
	exp_alpha = np.linspace(5, 15, 6)
	params = {
	    "alpha" : 2**-exp_alpha,
	    "n_episodes" : [50],
	    "timesteps" : [5000, 100000, 700000, 1e6]

	}

	max_return = 0
	best_config = {}
	
	# Comment or uncomment the environment on which you want to train the TRPO model
	env = gym.make('CustomHopper-source-v0')
	#env = gym.make('CustomHopper-target-v0')
	
	print('State space:', env.observation_space)
	print('Action space:', env.action_space)
	print('Dynamics parameters:', env.get_parameters())


	for c in ParameterGrid(params):

		# Training TRPO model, based on a particular configuration of hyperparameters
		model = TRPO("MlpPolicy", env, c.get("alpha"), verbose=0)
		model.learn(c.get("timesteps"))
		
		tot_return = 0
		
		# Testing TRPO model
		for episode in range(c.get("n_episodes")):
			
			done = False
			obs = env.reset()
			
			render = False

			while not done:
				
				action, _states = model.predict(obs, deterministic=True)
				obs, reward, done, info = env.step(action)

				if done:
					obs = env.reset()
				
				if render:
					env.render()

				tot_return = tot_return + reward				
				
		print(c, tot_return/c.get("n_episodes"))
		
		if tot_return/c.get("n_episodes") > max_return:
			max_return = tot_return/c.get("n_episodes")
			best_config = c
	
	print("Best hyperparameter configuration: ", c)
	
	# Training the model based on the best configuration of hyperparameters
	model = TRPO("MlpPolicy", env, best_config.get("alpha"), verbose=0)
	model.learn(best_config.get("timesteps"))
	
	model.save("model_trpo.mdl")
	

if __name__ == '__main__':
	main()

