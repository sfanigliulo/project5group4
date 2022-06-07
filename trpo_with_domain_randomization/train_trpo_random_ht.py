import torch
import gym
import argparse
import numpy as np
import sys
sys.path.append('../')
from env.custom_hopper_random import *
from sb3_contrib import TRPO
from sklearn.model_selection import ParameterGrid

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
	# flag = 1 means that the masses will be randomized at the beggining of each episode
	train_env = gym.make('CustomHopper-source-v0', flag = 1)
	#train_env = gym.make('CustomHopper-target-v0', flag = 1)
	
	print('State space:', train_env.observation_space)
	print('Action space:', train_env.action_space)
	print('Dynamics parameters:', train_env.get_parameters())
	
	# Comment or uncomment the environment on which you want to test the TRPO model
	# flag = 0 means that there will not be randomization
	test_env = gym.make('CustomHopper-source-v0', flag = 0)
	#test_env = gym.make('CustomHopper-target-v0', flag = 0)
	
	for c in ParameterGrid(params):
		
		# Train the model on the environment with the randomization of masses
		# with the specified hyperparameters configuration
		model = TRPO("MlpPolicy", train_env, c.get("alpha"), verbose=0)
		model.learn(c.get("timesteps"))
		
		tot_return = 0
		
		# Test the model on the standard environment (without the randomization of masses)
		for episode in range(c.get("n_episodes")):
			
			done = False
			obs = test_env.reset()
			
			render = False

			while not done:
				
				action, _states = model.predict(obs, deterministic=True)
				obs, reward, done, info = test_env.step(action)

				if done:
					obs = test_env.reset()
				
				if render:
					test_env.render()

				tot_return = tot_return + reward				
				
		print(c, tot_return/c.get("n_episodes"))
		
		if tot_return/c.get("n_episodes") > max_return:
			max_return = tot_return/c.get("n_episodes")
			best_config = c
	
	print("Best hyperparameter configuration: ", c)
	
	# Train the TRPO model on the environment with the randomization of masses,
	# using the best hyperparameter configuration
	model = TRPO("MlpPolicy", train_env, best_config.get("alpha"), verbose=0)
	model.learn(best_config.get("timesteps"))
	model.save("model_trpo.mdl")

	

if __name__ == '__main__':
	main()
