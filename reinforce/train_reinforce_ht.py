import torch
import gym
import argparse
import sys
sys.path.append('../')
from env.custom_hopper import *
from agent_reinforce import Agent, Policy
from sklearn.model_selection import ParameterGrid

import wandb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=10000, type=int, help='Number of training episodes')
    parser.add_argument('--print-every', default=2000, type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')

    return parser.parse_args()

args = parse_args()

def main():
	max_return = 0
	best_config = {}
	exp_alpha = np.linspace(5, 15, 6)

	params = {
	    "alpha" : 2**-exp_alpha,
	    "baseline" : np.linspace(0.001, 0.01, 5),
	    "n_episodes" : [5000, 10000, 60000]
	}
	
	#env = gym.make('CustomHopper-source-v0')
	env = gym.make('CustomHopper-target-v0')
	
	observation_space_dim = env.observation_space.shape[-1]
	action_space_dim = env.action_space.shape[-1]

	for config in ParameterGrid(params):
	    
		policy = Policy(observation_space_dim, action_space_dim)
		agent = Agent(policy, config.get("alpha"), config.get("baseline"), device=args.device)
		tot_return = 0

		for episode in range(config.get("n_episodes")):

			done = False
			train_reward = 0
			state = env.reset()

			while not done:  
				action, action_probabilities = agent.get_action(state)
				previous_state = state
				state, reward, done, info = env.step(action.detach().cpu().numpy())
				agent.store_outcome(previous_state, state, action_probabilities, reward, done)
				train_reward += reward
				agent.update_policy()
			tot_return = tot_return + train_reward
		
		print(config,tot_return/config.get("n_episodes"))
		if tot_return/config.get("n_episodes") > max_return:
			max_return = tot_return/config.get("n_episodes")
			best_config = config
			
	print("Best configuration: ", best_config)
	    
	"""
		TRAINING
	"""	
	policy = Policy(observation_space_dim, action_space_dim)
	agent = Agent(policy, best_config.get("alpha"), best_config.get("baseline"), device=args.device)
	tot_return = 0

	for episode in range(best_config.get("n_episodes")):

		done = False
		train_reward = 0
		state = env.reset()  

		while not done:  
			action, action_probabilities = agent.get_action(state)
			previous_state = state
			state, reward, done, info = env.step(action.detach().cpu().numpy())
			agent.store_outcome(previous_state, state, action_probabilities, reward, done)
			train_reward += reward
			agent.update_policy()
	
	torch.save(agent.policy.state_dict(), "model_reinforce.mdl")
		
	    

if __name__ == '__main__':
	main()
