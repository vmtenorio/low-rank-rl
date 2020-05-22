from utils import Saver, LowRankLearning
import matplotlib.pyplot as plt
import numpy as np

q_learner_baseline = Saver.load_from_pickle("results/q_learning_ss02_sa005.pickle")
q_learner_025_005 = Saver.load_from_pickle("results/q_learning_ss025_sa005.pickle")
q_learner_02_01 = Saver.load_from_pickle("results/q_learning_ss02_sa01.pickle")
q_learner_025_01 = Saver.load_from_pickle("results/q_learning_ss025_sa01.pickle")
q_learner_05_005 = Saver.load_from_pickle("results/q_learning_ss05_sa005.pickle")
q_learner_02_02 = Saver.load_from_pickle("results/q_learning_ss02_sa02.pickle")
q_learner_02_05 = Saver.load_from_pickle("results/q_learning_ss02_sa05.pickle")
q_learner_02_1 = Saver.load_from_pickle("results/q_learning_ss02_sa1.pickle")
q_learner_02_2 = Saver.load_from_pickle("results/q_learning_ss02_sa2.pickle")
q_learner_02_2 = Saver.load_from_pickle("results/q_learning_ss02_sa2.pickle")
q_learner_02_2_apenal_01 = Saver.load_from_pickle("results/q_learning_ss02_sa2_apenal_01.pickle")
q_learner_02_2_apenal_1 = Saver.load_from_pickle("results/q_learning_ss02_sa2_apenal_1.pickle")
q_learner_02_1_apenal_1 = Saver.load_from_pickle("results/q_learning_ss02_sa1_apenal_1.pickle")
q_learner_02_05_apenal_1 = Saver.load_from_pickle("results/q_learning_ss02_sa05_apenal_1.pickle")
q_learner_02_05_apenal_1 = Saver.load_from_pickle("results/q_learning_ss02_sa05_apenal_1.pickle")
q_learner_02_01_apenal_1 = Saver.load_from_pickle("results/q_learning_ss02_sa01_apenal_1.pickle")
q_learner_02_005_apenal_1 = Saver.load_from_pickle("results/q_learning_ss02_sa005_apenal_1.pickle")


#q_learner_02_1.test(1000)
#q_learner_02_2.test(1000)
#q_learner_02_1_apenal_1.test(1000)
q_learner_02_01_apenal_1.test(1000)
q_learner_02_005_apenal_1.test(1000)

low_rank_learner_k_10 = Saver.load_from_pickle("results/low_rank_k_10.pickle")
low_rank_learner_k_5 = Saver.load_from_pickle("results/low_rank_k_5.pickle")
low_rank_learner_k_3 = Saver.load_from_pickle("results/low_rank_k_3.pickle")
low_rank_learner_k_10_regularized = Saver.load_from_pickle("results/low_rank_k_10_regularized.pickle")
low_rank_learner_k_10_apenal_01 = Saver.load_from_pickle("results/low_rank_k_10_apenal_01.pickle")
low_rank_learner_k_3_apenal_1 = Saver.load_from_pickle("results/low_rank_k_3_apenal_1.pickle")

#low_rank_learner_k_10.test(1000)
#low_rank_learner_k_5.test(1000)
#low_rank_learner_k_3.test(1000)
#low_rank_learner_k_10_regularized.test(1000)
#low_rank_learner_k_3_apenal_1.test(1000)

mu_q_baseline, sigma_q_baseline = q_learner_baseline.plot_smoothed_steps(w=100, plot=False)
mu_q_02_02, sigma_q_02_02 = q_learner_02_02.plot_smoothed_steps(w=100, plot=False)
mu_q_02_05, sigma_q_02_05 = q_learner_02_05.plot_smoothed_steps(w=100, plot=False)
mu_q_02_1, sigma_q_02_1 = q_learner_02_1.plot_smoothed_steps(w=100, plot=False)
mu_q_02_2, sigma_q_02_2 = q_learner_02_2.plot_smoothed_steps(w=100, plot=False)
mu_q_02_2_apenal_1, sigma_q_02_2_apenal_1 = q_learner_02_2_apenal_1.plot_smoothed_steps(w=100, plot=False)
mu_q_02_1_apenal_1, sigma_q_02_1_apenal_1 = q_learner_02_1_apenal_1.plot_smoothed_steps(w=100, plot=False)
mu_q_02_05_apenal_1, sigma_q_02_05_apenal_1 = q_learner_02_05_apenal_1.plot_smoothed_steps(w=100, plot=False)
mu_q_02_01_apenal_1, sigma_q_02_01_apenal_1 = q_learner_02_01_apenal_1.plot_smoothed_steps(w=100, plot=False)
mu_q_02_005_apenal_1, sigma_q_02_005_apenal_1 = q_learner_02_005_apenal_1.plot_smoothed_steps(w=100, plot=False)
mu_lr_k_10, sigma_lr_k_10 = low_rank_learner_k_10.plot_smoothed_steps(w=100, plot=False)
mu_lr_k_5, sigma_lr_k_5 = low_rank_learner_k_5.plot_smoothed_steps(w=100, plot=False)
mu_lr_k_3, sigma_lr_k_3 = low_rank_learner_k_3.plot_smoothed_steps(w=100, plot=False)
mu_lr_k_10_reg, sigma_lr_k_10_reg = low_rank_learner_k_10_regularized.plot_smoothed_steps(w=100, plot=False)
mu_lr_k_3_apenal_1, sigma_lr_k_3_apenal_1 = low_rank_learner_k_3_apenal_1.plot_smoothed_steps(w=100, plot=False)

mu_rewards = [mu_q_02_01_apenal_1,
			  mu_q_02_005_apenal_1]

sigma_rewards = [sigma_q_02_01_apenal_1,
				 sigma_q_02_005_apenal_1]

legend = ['Q-learning 19 acciones penalty 1',
		  'Q-learning 39 acciones penalty 1']

colors = ['k', 'g']

plt.figure(figsize=[14, 4])
plt.title("Smoothed steps per episode")
for i in range(len(legend)):
	plt.plot(mu_rewards[i], color=colors[i])
	plt.fill_between(range(len(mu_rewards[i])), mu_rewards[i]-sigma_rewards[i], mu_rewards[i]+sigma_rewards[i], alpha=.1, color=colors[i])
plt.legend(legend)
plt.show()

"""
mu_rewards = [mu_q_baseline,
			  mu_lr_k_10,
			  mu_lr_k_5,
			  mu_lr_k_3,
			  mu_lr_k_10_reg]

sigma_rewards = [sigma_q_baseline,
                 sigma_lr_k_10,
			     sigma_lr_k_5,
			     sigma_lr_k_3,
			     sigma_lr_k_10_reg]

legend = ['Q-learning baseline',
		  'LR-learning k=10',
          'LR-learning k=5',
          'LR-learning k=3',
          'LR-learning k=10 regularized']

colors = ['k','g', 'b', 'r', 'm']

plt.figure(figsize=[14, 4])
plt.title("Smoothed steps per episode")
for i in range(len(legend)):
	plt.plot(mu_rewards[i], color=colors[i])
	plt.fill_between(range(len(mu_rewards[i])), mu_rewards[i]-sigma_rewards[i], mu_rewards[i]+sigma_rewards[i], alpha=.1, color=colors[i])
plt.legend(legend)
plt.show()

u, s_q_learning_baseline, vt = np.linalg.svd(q_learner_baseline.Q)
u, s_lr_k_10, vt = np.linalg.svd(low_rank_learner_k_10.L@low_rank_learner_k_10.R)
u, s_lr_k_5, vt = np.linalg.svd(low_rank_learner_k_5.L@low_rank_learner_k_5.R)
u, s_lr_k_3, vt = np.linalg.svd(low_rank_learner_k_3.L@low_rank_learner_k_3.R)
u, s_lr_k_10_reg, vt = np.linalg.svd(low_rank_learner_k_10_regularized.L@low_rank_learner_k_10_regularized.R)

def plot_log_singular_values(s):
	s = np.log(s)
	s[s < 0] = 0

	plt.figure(figsize=[14, 4])
	plt.title("Singular values")
	plt.bar(x=np.arange(len(s)), height=np.log(s))
	plt.show()

plot_log_singular_values(s_q_learning_baseline)
plot_log_singular_values(s_lr_k_10)
plot_log_singular_values(s_lr_k_5)
plot_log_singular_values(s_lr_k_3)
plot_log_singular_values(s_lr_k_10_reg)


def test_one_episode(Q, n_steps, learner):
    states = []
    actions = []
    rewards = []

    state = learner.env.reset()
    for i in range(n_steps):
        state_idx = learner.get_s_idx(state)
        action_idx = np.argmax(Q[state_idx, :])
        action = learner.action_map[action_idx]
        new_state, reward, done, info = learner.env.step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)

        state = new_state

    learner.env.close()

    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    
    return states, actions, rewards

states_q_baseline, actions_q_baseline, rewards_q_baseline = test_one_episode(q_learner_baseline.Q, 1000, q_learner_baseline)
states_lr_k_10, actions_lr_k_10, rewards_lr_k_10 = test_one_episode(low_rank_learner_k_10.L@low_rank_learner_k_10.R, 1000, low_rank_learner_k_10)
states_lr_k_5, actions_lr_k_5, rewards_lr_k_5 = test_one_episode(low_rank_learner_k_5.L@low_rank_learner_k_5.R, 1000, low_rank_learner_k_5)
states_lr_k_3, actions_lr_k_3, rewards_lr_k_3 = test_one_episode(low_rank_learner_k_3.L@low_rank_learner_k_3.R, 1000, low_rank_learner_k_3)
states_lr_k_10_reg, actions_lr_k_10_reg, rewards_lr_k_10_reg = test_one_episode(low_rank_learner_k_10_regularized.L@low_rank_learner_k_10_regularized.R, 1000, low_rank_learner_k_10_regularized)

plt.figure(figsize=[14, 3])
plt.title("Cos(theta)")
plt.plot(states_q_baseline[:, 0], c='r')
plt.plot(states_lr_k_10[:, 0], c='b')
plt.plot(states_lr_k_5[:, 0], c='y')
plt.plot(states_lr_k_3[:, 0], c='m')
plt.plot(states_lr_k_10_reg[:, 0], c='orange')
plt.legend(legend)
plt.show()

plt.figure(figsize=[14, 3])
plt.title("Sin(theta)")
plt.plot(states_q_baseline[:, 1], c='r')
plt.plot(states_lr_k_10[:, 1], c='b')
plt.plot(states_lr_k_5[:, 1], c='y')
plt.plot(states_lr_k_3[:, 1], c='m')
plt.plot(states_lr_k_10_reg[:, 1], c='orange')
plt.legend(legend)
plt.show()

plt.figure(figsize=[14, 3])
plt.title("Theta dot")
plt.plot(states_q_baseline[:, 2], c='r')
plt.plot(states_lr_k_10[:, 2], c='b')
plt.plot(states_lr_k_5[:, 2], c='y')
plt.plot(states_lr_k_3[:, 2], c='m')
plt.plot(states_lr_k_10_reg[:, 0], c='orange')
plt.legend(legend)
plt.show()"""

