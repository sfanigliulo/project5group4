import torch
import gym
import argparse
import nevergrad as ng
import numpy as np
import pandas as pd
import sys
sys.path.append('../')
import env.custom_hopper_simopt as env_standard
from stable_baselines3 import PPO
from simopt_train import train_main

def main():
	# Setting the tolerance for the variance
	tol = 1e-3
	
	# (*) Retrieving mean and variance of the masses from the .csv file
	distributions = pd.read_csv("distributions.csv", header = None)
	mu_std1 = distributions.iloc[0,:].tolist()
	mu_std2 = distributions.iloc[1,:].tolist()
	mu_std3 = distributions.iloc[2,:].tolist()
	
	# Iterate till each variance is less than the tolerance
	while mu_std1[1]>tol and mu_std2[1]>tol and mu_std3[1]>tol:
		
		# Train the policy
		train_main()
		
		# Retrieving the updated mean and variance of the masses from the .csv file
		# Note that for the first iteration, this step and the one indicated with (*) are the same
		distributions = pd.read_csv("distributions.csv", header = None)
		mu_std1 = distributions.iloc[0,:].tolist()
		mu_std2 = distributions.iloc[1,:].tolist()
		mu_std3 = distributions.iloc[2,:].tolist()
		
		# Load the trained PPO model
		model = PPO.load("simopt.mdl")
		
		real = env_standard.gym.make('CustomHopper-target-v0', flag = 0)
		
		tot_rw_real = 0
		episodes = 50
		obs_real = []
		
		# Test the policy in real
		for episode in range(episodes):
			done = False
			train_reward = 0
			obs = real.reset()  
			test_reward = 0
			
			k = 0
			obss = []

			while not done:  
				action, _states = model.predict(obs, deterministic=True)
				obs, reward, done, info = real.step(action)
				# Collect the observation of each step
				if k == 0:
					obss = obs
				else:
					obss = obss+obs
				k = k+1

				test_reward += reward
			# Collect the observations of each episode	
			obs_real.append(obs)
			
			tot_rw_real +=test_reward
			
		print(f"Average Return REAL: {tot_rw_real/episodes}")
			
		sim = env_standard.gym.make('CustomHopper-source-v0', flag = 0)
		# Sample the values of the masses from the normal distributions with mean and variance as above
		masses = sim.get_parameters()
		masses[1] = np.random.normal(mu_std1[0], mu_std1[1], 1)
		masses[2] = np.random.normal(mu_std2[0], mu_std2[1], 1)
		masses[3] = np.random.normal(mu_std3[0], mu_std3[1], 1)
		sim.set_parameters(*masses)
		
		tot_rw_sim = 0
		obs_sim = []
			
		# Test the policy in simulation
		for episode in range(episodes):
			done = False
			train_reward = 0
			obs = sim.reset()  
			test_reward = 0
			k = 0
			obss = []
			while not done:  
				action, _states = model.predict(obs, deterministic=True)
				obs, reward, done, info = sim.step(action)
				# Collect the observation of each step
				if k == 0:
					obss = obs
				else:
					obss = obss+obs
				k = k+1

				test_reward += reward
			# Collect the observations of each episode
			obs_sim.append(obss/k)	
			
			tot_rw_sim +=test_reward
		print(f"Average Return SIM: {tot_rw_sim/episodes}")
		
		# Calculate the discrepancy between simulation and reality using L2-norm
		discrepancy = np.linalg.norm(np.array(obs_real)-np.array(obs_sim))
		print("Discrepancy: ", discrepancy)
		
		# Setting the parameters for the optimizer
		# x1, x2, x3 will represent the mean values of the distributions of the masses
		# Initialize them as scalars
		param = ng.p.Dict(
			x1 = ng.p.Scalar(),
			x2 = ng.p.Scalar(),
			x3 = ng.p.Scalar()
		)		
		
		# Set their value
		param["x1"].value = mu_std1[0]
		param["x2"].value = mu_std2[0]
		param["x3"].value = mu_std3[0]
		
		# Set their variation using the variance values of the distributions of the masses
		param["x1"].set_mutation(sigma = mu_std1[1])
		param["x2"].set_mutation(sigma = mu_std2[1])
		param["x3"].set_mutation(sigma = mu_std3[1])
		
		# Take a value for each x1, x2, x3 in the interval [mean - variance, mean + variance]
		param["x1"].mutate()
		param["x2"].mutate()
		param["x3"].mutate()
			
		# Initialize the optimizer
		optim = ng.optimizers.CMA(parametrization = param, budget=1200)
		
		# Use the retrieved value x1, x2, x3 to start the iterations of the optimizers
		for _ in range(optim.budget):
			x = optim.ask()
			optim.tell(x, discrepancy)
			
		# Use "recommend" to obtain the candidates x1, x2, x3 values with minimal loss (i.e. that minimize the discrepancy)
		recommendation = optim.recommend()
		print(recommendation.value)
		
		# In order to update the distributions, select 300 samples from the actual distributions
		samples1 = np.random.normal(mu_std1[0], mu_std1[1], 300)
		samples2 = np.random.normal(mu_std2[0], mu_std2[1], 300)
		samples3 = np.random.normal(mu_std3[0], mu_std3[1], 300)
		
		# Add to these samples the new recommended values in order to modify the distributions
		# in the direction of the minimization of the discrepancy
		samples1 = np.append(samples1, np.array(recommendation['x1'].value).reshape(1,), axis=0)
		samples2 = np.append(samples2, np.array(recommendation['x2'].value).reshape(1,), axis=0)
		samples3 = np.append(samples3, np.array(recommendation['x3'].value).reshape(1,), axis=0)
		
		# Update the mean values of the distributions
		mu_std1[0] = np.mean(samples1)
		mu_std2[0] = np.mean(samples2)
		mu_std3[0] = np.mean(samples3)
		
		# Update the variance values of the distributions
		mu_std1[1] = np.var(samples1)
		mu_std2[1] = np.var(samples2)
		mu_std3[1] = np.var(samples3)
		
		# Overwrite the distribution.csv file to save mean and variance of each distribution
		distributions = pd.DataFrame([mu_std1,mu_std2,mu_std3])
		print(distributions)
		distributions.to_csv("distributions.csv", header= False, index= False)
	
	

if __name__ == '__main__':
	main()	
