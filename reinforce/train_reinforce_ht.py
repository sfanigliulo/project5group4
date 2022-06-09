import torch
import gym
import argparse
import sys
sys.path.append('../')
from env.custom_hopper import *
from agent_reinforce import Agent, Policy
from sklearn.model_selection import ParameterGrid

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', default=50, type=int, help='Number of training episodes')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')

    return parser.parse_args()

args = parse_args()

def main():
	max_return = 0
	best_config = {}
	exp_alpha = np.linspace(5, 15, 3)

	params = {
	    "alpha" : 2**-exp_alpha,
	    "baseline" : np.linspace(0.001, 0.01, 3),
	    "n_episodes" : [5000, 10000, 50000]
	}
	
	# Comment or uncomment the environment on which you want to train the model
	env = gym.make('CustomHopper-source-v0')
	#env = gym.make('CustomHopper-target-v0')
	
	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Dynamics parameters:', env.get_parameters())
	
	observation_space_dim = env.observation_space.shape[-1]
	action_space_dim = env.action_space.shape[-1]

	for config in ParameterGrid(params):
		
		print(config)
		
		# Training the model, based on a particular configuration of hyperparameters	    
		policy = Policy(observation_space_dim, action_space_dim)
		agent = Agent(policy, config.get("alpha"), config.get("baseline"), device=args.device)
		tot_reward = 0

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
		
		# Testing the model trained
		for episode in range(args.episodes):
			done = False
			test_reward = 0
			state = env.reset()
			render = False

			while not done:

				action, _ = agent.get_action(state, evaluation=True)

				state, reward, done, info = env.step(action.detach().cpu().numpy())

				if render:
				    env.render()

				test_reward += reward

			tot_reward +=test_reward
		
		print(f"Average Return: {tot_reward/args.episodes}")
		
		# Selection of the best hyperparameters configuration
		if tot_reward/args.episodes > max_return:
			max_return = tot_reward/args.episodes
			best_config = config
		    	
	print("Best configuration: ", best_config)
	    
	# Training the model based on the best configuration of hyperparameters	
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
