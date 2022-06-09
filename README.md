# project5group4

This is the implementation of **project 5 - Sim-to-Real transfer of Reinforcement Learning policies in robotics** by group 4 (Feraud Elisa, Fanigliulo Sofia, Grandi Francesco) for the *Machine Learning and Deep Learning* course (a.a. 2021/2022).

Each directory contains the implementation of a different algorithm as requested.

### Reinforce
* `agent_reinforce.py` contains the implementation of the reinforce algorithm.

* `train_reinforce_ht.py` contains the hyperparameters tuning and consequently the training on the best hyperparameter combination of the *REINFORCE* algorithm. It returns the trained model.

* `test_reinforce.py` contains the test of the model trained with the previous file .py and return the average reward over a specified number of episodes.

### Actor Critic
