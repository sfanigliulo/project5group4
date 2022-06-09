import torch
import gym
import argparse
import nevergrad as ng
import numpy as np
import sys
sys.path.append('../')
import env.custom_hopper_simopt as env_simopt

from stable_baselines3 import PPO

def train_main():
	
	# Select the environment with the randomization enabled
	env = env_simopt.gym.make('CustomHopper-source-v0', flag = 1)
	
	# Train PPO model
	model = PPO("MlpPolicy", env, verbose=1)
	model.learn(total_timesteps=700000)
	
	# Save the model to perform simopt
	model.save("simopt.mdl")


if __name__ == '__train_main__':
	train_main()
