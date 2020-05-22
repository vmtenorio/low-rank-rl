import numpy as np
import matplotlib.pyplot as plt
import PIL
from IPython.display import clear_output
import timeit
import itertools
import pickle


class QLearning:
    def __init__(self,
                 env,
                 state_map,
                 action_map,
                 state_reverse_map,
                 action_reverse_map,
                 n_states,
                 n_actions,
                 decimal_state,
                 decimal_action,
                 step_state,
                 step_action,
                 episodes=100000,
                 max_steps=1000,
                 decaying_epsilon=True,
                 decayment_rate=.999999,
                 epsilon=.9,
                 epsilon_lower_bound=0.0,
                 exploration_limit=1000,
                 alpha=.9,
                 gamma=.9,
                 action_penalty=.001):

        self.env = env
        self.state_map = state_map
        self.action_map = action_map
        self.state_reverse_map = state_reverse_map
        self.action_reverse_map = action_reverse_map
        self.n_states = n_states
        self.n_actions = n_actions
        self.step_state = step_state
        self.step_action = step_action
        self.decimal_state = decimal_state
        self.decimal_action = decimal_action
        self.episodes = episodes
        self.max_steps = max_steps
        self.decaying_epsilon = decaying_epsilon
        self.decayment_rate = decayment_rate
        self.epsilon = epsilon
        self.epsilon_lower_bound = epsilon_lower_bound
        self.exploration_limit = exploration_limit
        self.alpha = alpha
        self.gamma = gamma
        self.action_penalty = action_penalty

        self.Q = np.zeros((n_states, n_actions))

        self.episodic_cumm_reward = []

        self.elapsed_time = None

    def get_s_idx(self, st):
        st_ = [self.step_state * (np.round(s / self.step_state)) for s in st]
        return self.state_reverse_map[str(np.around(st_, self.decimal_state) + 0.)]

    def get_a_idx(self, at):
        at_ = [self.step_action * (np.round(a / self.step_action)) for a in at]
        return self.action_reverse_map[str(np.around(at_, self.decimal_action) + 0.)]

    def choose_action(self, st_idx):
        if np.random.rand() < self.epsilon:
            a = self.env.action_space.sample()
            return a, self.get_a_idx(a)

        a_idx = np.argmax(self.Q[st_idx, :])
        return self.action_map[a_idx], a_idx

    def plot_steps(self):
        clear_output()
        plt.figure(figsize=[14, 4])
        plt.title("Cummulative reward per episode - epsilon: {}".format(np.around(self.epsilon, 2)))
        plt.plot(self.episodic_cumm_reward)
        plt.show()

    def plot_smoothed_steps(self, w, plot=True):

        avg = np.array([np.mean(self.episodic_cumm_reward[i:i + w]) for i in
                        range(len(self.episodic_cumm_reward) - w)])
        std = np.array([np.std(self.episodic_cumm_reward[i:i + w]) for i in
                        range(len(self.episodic_cumm_reward) - w)])

        if plot:
            clear_output()
            plt.figure(figsize=[14, 4])
            plt.title("Smoothed steps per episode")
            plt.plot(avg)
            plt.fill_between(range(len(self.episodic_cumm_reward) - w), avg - std, avg + std,
                             alpha=.1)
            plt.show()

        return avg, std

    def train(self):
        start = timeit.timeit()
        for episode in range(self.episodes):
            s = self.env.reset()
            s_idx = self.get_s_idx(s)

            cumm_reward = 0

            for step in range(self.max_steps):
                a, a_idx = self.choose_action(s_idx)

                s_prime, r, done, _ = self.env.step(a)
                r = -(np.arccos(s[0])**2 + .1*s[2]**2 + self.action_penalty*a[0]**2)
                s_prime_idx = self.get_s_idx(s_prime)
                cumm_reward += r

                target_q = r + self.gamma * np.max(self.Q[s_prime_idx, :])
                error_signal = target_q - self.Q[s_idx, a_idx]
                self.Q[s_idx, a_idx] += self.alpha * error_signal

                s = s_prime
                s_idx = s_prime_idx

                if self.decaying_epsilon & (episode > self.exploration_limit):
                    if self.epsilon > self.epsilon_lower_bound:
                        self.epsilon *= self.decayment_rate

            self.episodic_cumm_reward.append(cumm_reward)
            print("EPISODE: ", episode,
                  " - Cumm. reward: ", cumm_reward,
                  " - Epsilon: ", np.around(self.epsilon, 2))

        end = timeit.timeit()
        self.elapsed_time = end - start

    def test(self, n_steps):
        s = self.env.reset()

        for i in range(n_steps):
            s_idx = self.get_s_idx(s)
            a_idx = np.argmax(self.Q[s_idx, :])
            a = self.action_map[a_idx]
            s_prime, r, done, info = self.env.step(a)
            PIL.Image.fromarray(self.env.render(mode='rgb_array')).resize((320, 420))
            s = s_prime
        self.env.close()


class LowRankLearning:
    def __init__(self,
                 env,
                 state_map,
                 action_map,
                 state_reverse_map,
                 action_reverse_map,
                 n_states,
                 n_actions,
                 k,
                 decimal_state,
                 decimal_action,
                 step_state,
                 step_action,
                 episodes=100000,
                 max_steps=1000,
                 decaying_epsilon=True,
                 decayment_rate=.999999,
                 epsilon=.9,
                 epsilon_lower_bound=0.0,
                 exploration_limit=1000,
                 alpha=.9,
                 gamma=.9,
                 lambda_l=0.0,
                 lambda_r=0.0,
                 action_penalty=.001):

        self.env = env
        self.state_map = state_map
        self.action_map = action_map
        self.state_reverse_map = state_reverse_map
        self.action_reverse_map = action_reverse_map
        self.k = k
        self.n_states = n_states
        self.n_actions = n_actions
        self.step_state = step_state
        self.step_action = step_action
        self.decimal_state = decimal_state
        self.decimal_action = decimal_action
        self.episodes = episodes
        self.max_steps = max_steps
        self.decaying_epsilon = decaying_epsilon
        self.decayment_rate = decayment_rate
        self.epsilon = epsilon
        self.epsilon_lower_bound = epsilon_lower_bound
        self.exploration_limit = exploration_limit
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_l = lambda_l
        self.lambda_r = lambda_r
        self.action_penalty = action_penalty

        self.L = np.random.rand(n_states, k) / 1e5
        self.R = np.random.rand(k, n_actions) / 1e5

        self.episodic_cumm_reward = []

        self.elapsed_time = None

    def get_s_idx(self, st):
        st_ = [self.step_state * (np.round(s / self.step_state)) for s in st]
        return self.state_reverse_map[str(np.around(st_, self.decimal_state) + 0.)]

    def get_a_idx(self, at):
        at_ = [self.step_action * (np.round(a / self.step_action)) for a in at]
        return self.action_reverse_map[str(np.around(at_, self.decimal_action) + 0.)]

    def choose_action(self, q_current_state):
        if np.random.rand() < self.epsilon:
            a = self.env.action_space.sample()
            return a, self.get_a_idx(a)

        a_idx = np.argmax(q_current_state)
        return self.action_map[a_idx], a_idx

    def plot_steps(self):
        clear_output()
        plt.figure(figsize=[14, 4])
        plt.title("Cummulative reward per episode - epsilon: {}".format(np.around(self.epsilon, 2)))
        plt.plot(self.episodic_cumm_reward)
        plt.show()

    def plot_smoothed_steps(self, w, plot=True):

        avg = np.array([np.mean(self.episodic_cumm_reward[i:i + w]) for i in
                        range(len(self.episodic_cumm_reward) - w)])
        std = np.array([np.std(self.episodic_cumm_reward[i:i + w]) for i in
                        range(len(self.episodic_cumm_reward) - w)])

        if plot:
            clear_output()
            plt.figure(figsize=[14, 4])
            plt.title("Smoothed steps per episode")
            plt.plot(avg)
            plt.fill_between(range(len(self.episodic_cumm_reward) - w), avg - std, avg + std,
                             alpha=.1)
            plt.show()

        return avg, std

    def train(self):
        start = timeit.timeit()
        for episode in range(self.episodes):

            s = self.env.reset()
            s_idx = self.get_s_idx(s)
            q_current_state = self.L[s_idx, :] @ self.R

            cumm_reward = 0

            for step in range(self.max_steps):
                a, a_idx = self.choose_action(q_current_state)

                s_prime, r, done, _ = self.env.step(a)
                r = -(np.arccos(s[0])**2 + .1*s[2]**2 + self.action_penalty*a[0]**2)
                s_prime_idx = self.get_s_idx(s_prime)
                cumm_reward += r

                q_next_state = self.L[s_prime_idx, :] @ self.R
                q_bootstrapped = r + self.gamma * np.max(q_next_state)
                q_hat = q_current_state[a_idx]

                err = (q_bootstrapped - q_hat)

                self.L[s_idx, :] -= self.alpha * (
                            -err * self.R[:, a_idx] + self.lambda_l * self.L[s_idx, :])
                self.R[:, a_idx] -= self.alpha * (
                            -err * self.L[s_idx, :] + self.lambda_r * self.R[:, a_idx])

                s = s_prime
                s_idx = s_prime_idx
                q_current_state = q_next_state

                if self.decaying_epsilon & (episode > self.exploration_limit):
                    if self.epsilon > self.epsilon_lower_bound:
                        self.epsilon *= self.decayment_rate

            self.episodic_cumm_reward.append(cumm_reward)
            print("EPISODE: ", episode,
                  " - Cumm. reward: ", cumm_reward,
                  " - Epsilon: ", np.around(self.epsilon, 2))

        end = timeit.timeit()
        self.elapsed_time = end - start

    def test(self, n_steps):
        s = self.env.reset()
        Q_hat = self.L @ self.R

        for i in range(n_steps):
            s_idx = self.get_s_idx(s)
            a_idx = np.argmax(Q_hat[s_idx, :])
            a = self.action_map[a_idx]
            s_prime, r, done, info = self.env.step(a)
            PIL.Image.fromarray(self.env.render(mode='rgb_array')).resize((320, 420))
            s = s_prime

        self.env.close()


class Mapper:
    def __init__(self):

        self.environment = "Pendulum-v0"

        self.min_cos_theta = -1.0
        self.max_cos_theta = 1.0
        self.min_sin_theta = -1.0
        self.max_sin_theta = 1.0
        self.min_theta_dot = -8.0
        self.max_theta_dot = 8.0

        self.min_joint_effort = -2.0
        self.max_joint_effort = 2.0

    def get_map(self, iterable):
        mapping = [np.array(combination) for combination in itertools.product(*iterable)]
        reverse_mapping = {str(mapping[i]):i for i in range(len(mapping))}

        return mapping, reverse_mapping

    def get_state_map(self, step, decimal):
        cos_theta = np.around(np.arange(self.min_cos_theta, self.max_cos_theta +
                                        step, step), decimal) + 0.
        sin_theta = np.around(np.arange(self.min_sin_theta, self.max_sin_theta +
                                        step, step), decimal) + 0.
        theta_dot = np.around(np.arange(self.min_theta_dot, self.max_theta_dot +
                                        step, step), decimal) + 0.

        return self.get_map([cos_theta, sin_theta, theta_dot])

    def get_action_map(self, step, decimal):
        joint_effort = np.around(np.arange(self.min_joint_effort, self.max_joint_effort +
                                           step, step), decimal) + 0.

        return self.get_map([joint_effort])


class Saver:
    @staticmethod
    def save_to_pickle(path, obj):
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

    @staticmethod
    def load_from_pickle(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
