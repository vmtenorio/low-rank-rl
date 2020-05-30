from utils import Saver, LowRankLearning, TestUtils, Mapper
import matplotlib.pyplot as plt
import numpy as np
import gym

utils_test = TestUtils()
env = gym.make(Mapper().environment)

q_learner_02_1 = Saver.load_from_pickle("results_greedy/q_learning_ss02_sa005.pickle")
q_learner_02_005 = Saver.load_from_pickle("results_greedy/q_learning_ss02_sa005.pickle")
#low_rank_learner_k_2 = Saver.load_from_pickle("results/low_rank_k_2_apenal_05.pickle")
low_rank_learner_k_2 = Saver.load_from_pickle("results_greedy/q_learning_ss02_sa005.pickle")

q_learner_02_1.env = env
q_learner_02_005.env = env
low_rank_learner_k_2.env = env

#Q_low_rank_learner_k_2 = low_rank_learner_k_2.L@low_rank_learner_k_2.R
Q_low_rank_learner_k_2 = low_rank_learner_k_2.Q

mu_q_02_1, sigma_q_02_1 = q_learner_02_1.plot_smoothed_steps(w=100, plot=False)
mu_q_02_005, sigma_q_02_005 = q_learner_02_005.plot_smoothed_steps(w=100, plot=False)
mu_lr_k_2, sigma_lr_k_2 = low_rank_learner_k_2.plot_smoothed_steps(w=100, plot=False)

mu_rewards = [mu_q_02_1,
			  mu_q_02_005,
			  mu_lr_k_2]

sigma_rewards = [sigma_q_02_1,
				 sigma_q_02_005,
				 sigma_lr_k_2]

legend = ['Q-learning: 49005 parameters (9801 x 5)',
		  'Q-learning: 793881 parameters (9801 x 81)',
		  'Low-rank: 19764 (L: 9801 x 2 / R: 2 x 81)']

colors = ['k', 'g', 'b']

plt.figure(figsize=[14, 4])
plt.title("Smoothed steps per episode")
for i in range(len(legend)):
	plt.plot(mu_rewards[i], color=colors[i])
	plt.fill_between(range(len(mu_rewards[i])), mu_rewards[i]-sigma_rewards[i], mu_rewards[i]+sigma_rewards[i], alpha=.1, color=colors[i])
plt.legend(legend)
plt.show()

plt.figure(figsize=[14, 4])
plt.plot(q_learner_02_005.greedy_rewards_mu)
plt.show()

st_q_learner_02_1, a_q_learner_02_1, r_q_learner_02_1 = utils_test.test_one_episode(q_learner_02_1.Q, 1000, q_learner_02_1)
st_q_learner_02_005, a_q_learner_02_005, r_q_learner_02_005 = utils_test.test_one_episode(q_learner_02_005.Q, 1000, q_learner_02_005)
st_low_rank_learner_k_2, a_low_rank_learner_k_2, r_low_rank_learner_k_2 = utils_test.test_one_episode(Q_low_rank_learner_k_2, 1000, low_rank_learner_k_2)

states = [st_q_learner_02_1,
		  st_q_learner_02_005,
		  st_low_rank_learner_k_2]

actions = [a_q_learner_02_1,
		   a_q_learner_02_005,
		   a_low_rank_learner_k_2]

rewards = [r_q_learner_02_1,
		   r_q_learner_02_005,
		   r_low_rank_learner_k_2]

plt.figure(figsize=[14, 3])
plt.title("Theta")
for i in range(len(legend)):
	plt.plot(states[i], color=colors[i])
plt.legend(legend)
plt.show()

plt.figure(figsize=[14, 3])
plt.title("Action")
for i in range(len(legend)):
	plt.plot(actions[i], color=colors[i])
plt.legend(legend)
plt.show()

plt.figure(figsize=[14, 3])
plt.title("Reward")
for i in range(len(legend)):
	plt.plot(rewards[i], color=colors[i])
plt.legend(legend)
plt.show()

"""
u, s_lr_k_5_reg_0005, vt = np.linalg.svd(low_rank_learner_k_5_apenal_05_reg_0005.L@low_rank_learner_k_5_apenal_05_reg_0005.R)

utils_test.plot_log_singular_values(s_lr_k_5_reg_0005)"""


mu_r_q_learner_02_1, sigma_r_q_learner_02_1 = utils_test.test_average_reward(q_learner_02_1.Q, 10000, 1000, q_learner_02_1)
mu_r_q_learner_02_005, sigma_r_q_learner_02_005 = utils_test.test_average_reward(q_learner_02_005.Q, 10000, 1000, q_learner_02_005)
mu_r_low_rank_learner_k_2, sigma_r_low_rank_learner_k_2 = utils_test.test_average_reward(Q_low_rank_learner_k_2, 10000, 1000, low_rank_learner_k_2)

mu_rewards = [mu_r_q_learner_02_1,
			  mu_r_q_learner_02_005,
			  mu_r_low_rank_learner_k_2]

plt.figure(figsize=[14, 3])
plt.title("Average cost 1000 episodes")
plt.bar(x=np.arange(len(mu_rewards)), height=np.abs(mu_rewards))
plt.xticks(np.arange(len(legend)), legend, rotation=90)
plt.show()

