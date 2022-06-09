import torch
import gym
import argparse
import sys
sys.path.append('../')
from env.custom_hopper import *
from stable_baselines3 import PPO

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="model_ppo.mdl", type=str, help='Model path')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--render', default=False, action='store_true', help='Render the simulator')
    parser.add_argument('--episodes', default=50, type=int, help='Number of test episodes')

    return parser.parse_args()

args = parse_args()


def main():
	
	# Comment or uncomment the environment on which you want to test the model
	env = gym.make('CustomHopper-source-v0')
	#env = gym.make('CustomHopper-target-v0')

	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Dynamics parameters:', env.get_parameters())
	
	# Loading the trained PPO model
	model = PPO.load(args.model)
	
	tot_reward = 0
	
	# Testing the model
	for episode in range(args.episodes):
		done = False
		train_reward = 0
		obs = env.reset()
		test_reward = 0
		render = True

		while not done:
			action, _states = model.predict(obs, deterministic=True)
			obs, reward, done, info = env.step(action)		
			
			if done:
				obs = env.reset()
			
			if render:
				env.render()
			test_reward += reward
	        	
		print(f"Episode: {episode} | Return: {test_reward}")
		tot_reward +=test_reward
	        
	print(f"Average Return: {tot_reward/args.episodes}")
	

if __name__ == '__main__':
	main()
