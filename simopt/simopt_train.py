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
	
	env = env_simopt.gym.make('CustomHopper-source-v0', flag = 1)
	
	# Train di una policy (PPO) 
	model = PPO("MlpPolicy", env, verbose=1)
	model.learn(total_timesteps=700000)
	
	model.save("simopt.mdl")


if __name__ == '__train_main__':
	train_main()
