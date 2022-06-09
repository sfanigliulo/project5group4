# project5group4

This is the implementation of **project 5 - Sim-to-Real transfer of Reinforcement Learning policies in robotics** by group 4 (Feraud Elisa, Fanigliulo Sofia, Grandi Francesco) for the *Machine Learning and Deep Learning* course (a.a. 2021/2022).

Each directory contains the implementation of a different algorithm as requested.

### Reinforce
* `agent_reinforce.py` contains the implementation of the agent and policy classes to use to train the policy with the *REINFORCE* algorithm.
* `train_reinforce_ht.py` contains the hyperparameters tuning and consequently the training on the best hyperparameter combination of the *REINFORCE* algorithm. It returns the trained model.
* `test_reinforce.py` contains the test of the model trained with the previous file .py and return the average reward over a specified number of episodes.

### Actor Critic
* `agent_actor_critic.py` contains the implementation of the three classes agent, actor and critic. Note that actor and critic have been implemented separately to handle in an easier way the backpropagation of the errors while training the neural network.
* `train_actor_critc_ht.py` contains the hyperparameters tuning and consequently the training on the best hyperparameter combination of the *One-Step Actor Critic* algorithm. It returns the trained model.
* `test_actor_critic.py` contains the test of the model trained with the previous file .py and return the average reward over a specified number of episodes.

### PPO
* `train_ppo_ht.py` contains the hyperparameters tuning and consequently the training on the best hyperparameter combination of the *Proximal Policy Optimization (PPO)* algorithm. It returns the trained model.
* `test_ppo.py` contains the test of the model trained with the previous file .py and return the average reward over a specified number of episodes.

### TRPO
* `train_trpo_ht.py` contains the hyperparameters tuning and consequently the training on the best hyperparameter combination of the *Trust Region Policy Optimization (TRPO)* algorithm. It returns the trained model.
* `test_trpo.py` contains the test of the model trained with the previous file .py and return the average reward over a specified number of episodes.

### TRPO with Domain Randomization
* `train_trpo_random_ht.py` contains the hyperparameters tuning and consequently the training on the best hyperparameter combination of the *Trust Region Policy Optimization (TRPO)* algorithm performed with Domain randomization among the masses of the Hopper. It returns the trained model.
* `test_trpo_random.py` contains the test of the model trained with the previous file .py and return the average reward over a specified number of episodes.
* `masse1.csv`, `masse2.csv`, `masse3.csv` contain three possible set of distributions of the masses to perform domain randomization and to compare results.

### SimOpt
* `simopt.py` contains the implementation of the SimOpt improvement for Domain Randomization.
* `simopt_train.py` is recalled in `simopt.py` to train the PPO model at the beggining of each SimOpt iteration. The training is performed using Domain Randomization over the updated distribution of masses.
* `distributions.csv` contains the initial mean and variance for each part of the hopper in order to represent distributions. This file is received as input for SimOpt and it is updated at the end of each iteration with the new mean and variance values.

### Env
Apart from the already-provided files, here there are two new implementation of `custom_hopper.py`.
* `custom_hopper_random.py` contains the methods for the standard domain randomization using uniform distribution for the masses. It can be used as `custom_hopper.py` if the value of `flag = 0`. To enable the domain randomization, set `flag = 1` or leave it as it is since it is the default option. It retrieves the values for sampling the masses from the uniform distributions from `masse1.csv`, `masse2.csv`, `masse3.csv`.
* `custom_hopper_simopt.py` is similar to `custom_hopper_random.py`. The main difference it that in this case the domain randomization is performed by means of a normal distribution over the masses. It retrieves the values for sampling the masses from the normal distributions from `distributions.csv`.
