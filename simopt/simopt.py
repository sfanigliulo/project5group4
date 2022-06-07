# https://github.com/simopt-admin/simopt/tree/master/simopt

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
	tol = 1e-3
	
	distributions = pd.read_csv("distributions.csv", header = None)
	mu_std1 = distributions.iloc[0,:].tolist()
	mu_std2 = distributions.iloc[1,:].tolist()
	mu_std3 = distributions.iloc[2,:].tolist()
	

	while mu_std1[1]>tol and mu_std2[1]>tol and mu_std3[1]>tol:
	
		train_main()

		distributions = pd.read_csv("distributions.csv", header = None)
		mu_std1 = distributions.iloc[0,:].tolist()
		mu_std2 = distributions.iloc[1,:].tolist()
		mu_std3 = distributions.iloc[2,:].tolist()
		
		model = PPO.load("simopt.mdl")
		
		# Testare la policy su real
		real = env_standard.gym.make('CustomHopper-target-v0', flag = 0)
		
		tot_rw_real = 0
		episodes = 50
		obs_real = []
		
		for episode in range(episodes):
			#env_standard.flag = 0
			#print(real.get_parameters())
			done = False
			train_reward = 0
			obs = real.reset()  # Reset the environment and observe the initial state
			test_reward = 0
			
			k = 0
			obss = []

			while not done:  # Loop until the episode is over
				action, _states = model.predict(obs, deterministic=True)
				obs, reward, done, info = real.step(action)
				if k == 0:
					obss = obs
				else:
					obss = obss+obs
				k = k+1
				#print(real.get_parameters())
				
				#if done:
				#	obs = real.reset()

				test_reward += reward
				
			obs_real.append(obs)
			
			#print(f"Episode REAL: {episode} | Return REAL: {test_reward}")
			tot_rw_real +=test_reward
			
		print(f"Average Return REAL: {tot_rw_real/episodes}")
			
		sim = env_standard.gym.make('CustomHopper-source-v0', flag = 0)
		# Pescare masse da distribuzioni
		masses = sim.get_parameters()
		masses[1] = np.random.normal(mu_std1[0], mu_std1[1], 1)
		masses[2] = np.random.normal(mu_std2[0], mu_std2[1], 1)
		masses[3] = np.random.normal(mu_std3[0], mu_std3[1], 1)
		sim.set_parameters(*masses)
		
		tot_rw_sim = 0
		obs_sim = []
			
		for episode in range(episodes):
			#env_standard.flag = 0
			#print(sim.get_parameters())
			done = False
			train_reward = 0
			obs = sim.reset()  # Reset the environment and observe the initial state
			test_reward = 0
			k = 0
			obss = []
			while not done:  # Loop until the episode is over
				action, _states = model.predict(obs, deterministic=True)
				obs, reward, done, info = sim.step(action)
				if k == 0:
					obss = obs
				else:
					obss = obss+obs
				k = k+1
				#print(sim.get_parameters())
				
				#if done:
				#	obs = sim.reset()

				test_reward += reward
			obs_sim.append(obss/k)	
			#print(f"Episode SIM: {episode} | Return SIM: {test_reward}")
			tot_rw_sim +=test_reward
		print(f"Average Return SIM: {tot_rw_sim/episodes}")
			
		
		
		
		# Calcolare discrepancy
		discrepancy = np.linalg.norm(np.array(obs_real)-np.array(obs_sim))
		print("Discrepancy: ", discrepancy)
		
		# minimizzare la discrepancy
		
		#param = ng.p.Dict(
	    	#	x1 = ng.p.Array(shape=(2,)),#.set_bounds(lower=0),
	    	#	x2 = ng.p.Array(shape=(2,)),#.set_bounds(lower=0),
	    	#	x3 = ng.p.Array(shape=(2,)),#.set_bounds(lower=0),#, upper=12),
		#	)
			
		#param["x1"].value = mu_std1
		#param["x2"].value = mu_std2
		#param["x3"].value = mu_std3	
		
		#print(param["x1"].value)
		
		param = ng.p.Dict(
			x1 = ng.p.Scalar(),
			x2 = ng.p.Scalar(),
			x3 = ng.p.Scalar()
		)		
		
		param["x1"].value = mu_std1[0]
		param["x2"].value = mu_std2[0]
		param["x3"].value = mu_std3[0]
		
		param["x1"].set_mutation(sigma = mu_std1[1])
		param["x2"].set_mutation(sigma = mu_std2[1])
		param["x3"].set_mutation(sigma = mu_std3[1])
		
		param["x1"].mutate()
		param["x2"].mutate()
		param["x3"].mutate()
		
		#param = ng.p.Dict(
		#	x1 = ng.p.Log(lower = mu_std1[0]-mu_std1[1], upper = mu_std1[0]+mu_std1[1]),
		#	x2 = ng.p.Log(lower = mu_std2[0]-mu_std2[1], upper = mu_std2[0]+mu_std2[1]),
		#	x3 = ng.p.Log(lower = mu_std3[0]-mu_std3[1], upper = mu_std3[0]+mu_std3[1])
		#)
			
		
		optim = ng.optimizers.CMA(parametrization = param, budget=1200)
		
		for _ in range(optim.budget):
			x = optim.ask()
			optim.tell(x, discrepancy)
			
		# recommend provides the candidate with minimal loss
		recommendation = optim.recommend()
		print(recommendation.value)
		
		samples1 = np.random.normal(mu_std1[0], mu_std1[1], 300)
		samples2 = np.random.normal(mu_std2[0], mu_std2[1], 300)
		samples3 = np.random.normal(mu_std3[0], mu_std3[1], 300)
		
		samples1 = np.append(samples1, np.array(recommendation['x1'].value).reshape(1,), axis=0)
		samples2 = np.append(samples2, np.array(recommendation['x2'].value).reshape(1,), axis=0)
		samples3 = np.append(samples3, np.array(recommendation['x3'].value).reshape(1,), axis=0)
		
		# update delle distribuzioni
		mu_std1[0] = np.mean(samples1)
		mu_std2[0] = np.mean(samples2)
		mu_std3[0] = np.mean(samples3)
		
		mu_std1[1] = np.var(samples1)
		mu_std2[1] = np.var(samples2)
		mu_std3[1] = np.var(samples3)
		
		distributions = pd.DataFrame([mu_std1,mu_std2,mu_std3])
		print(distributions)
		distributions.to_csv("distributions.csv", header= False, index= False)
	
	

if __name__ == '__main__':
	main()	
