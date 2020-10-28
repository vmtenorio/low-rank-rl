# Readme

This repository contains the code of the paper *"Low-Rank State-Action Value-Function Approximation"*. The code is organized in three directories:
* **1 - FrozenLake-v0:** code with the simulations relating FrozenLake environment.
* **2 - Pendulum-v0:** code with the simulations relating Pendulum environment.
* **3 - Acrobot-v1:** code with the simulations relating Acrobot environment.
* **4 - Figures:** key figures obtained from the simulations.

In the first, the second and the third repositories, they can be found two folders with:
* **Experiments:** json files with each experiment run included in the paper.
* **Results:** pickle files with the data obtained from training used to build the figures and the results.

Training scripts vary depending on the environment, but there are four types of scripts that are relevant:
* **utils.py:** script with the implementation of Q-learning, low-rank and linear-based architectures learning, apart from test utils and more.
* **q_learning.py:** script that loads and runs Q-learning experiments.
* **deep_q_learning.py:** script that loads and runs Deep Q-learning experiments.
* **low_rank.py:** script that loads and runs low-rank experiments.
* **test.py:** script that takes the training outputs and build up the results. The reviewer may want to run this script in order to obtain the graphs and figures available in the paper.
