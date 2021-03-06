import torch
import gym
import argparse
import sys
sys.path.append('../')
from env.custom_hopper import *
from agent_reinforce import Agent, Policy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="model_reinforce.mdl", type=str, help='Model path')
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
	
	observation_space_dim = env.observation_space.shape[-1]
	action_space_dim = env.action_space.shape[-1]

	policy = Policy(observation_space_dim, action_space_dim)
	policy.load_state_dict(torch.load(args.model), strict=True)

	agent = Agent(policy, device=args.device)
	
	tot_reward = 0
	
	# Test
	for episode in range(args.episodes):
	    done = False
	    test_reward = 0
	    state = env.reset()

	    while not done:

	        action, _ = agent.get_action(state, evaluation=True)
	        
	        state, reward, done, info = env.step(action.detach().cpu().numpy())

	        if args.render:
	            env.render()

	        test_reward += reward
	        print(f"Episode: {episode} | Return: {test_reward}")
	    tot_reward +=test_reward
	        
	print(f"Average Return: {tot_reward/args.episodes}")
	

if __name__ == '__main__':
	main()
