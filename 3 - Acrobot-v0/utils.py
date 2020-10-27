import numpy as np
import matplotlib.pyplot as plt
import PIL
import itertools
import pickle
import gym
from gym import spaces
from gym.utils import seeding


class LowRankTD:
    def __init__(self,
                 env,
                 mapper,
                 k,
                 episodes=100000,
                 max_steps=1000,
                 epsilon=.9,
                 decay=.99999,
                 alpha=.9,
                 gamma=.9):

        """
        :param env: gym.envs
            OpenAI Gym environment.
        :param mapper: Class
            Mapper class that helps implementing the env.
        :param k: int
            Dimension of the latent space.
        :param episodes: int
            Number of episodes.
        :param max_steps: int
            Maximum number of steps per episode.
        :param epsilon: float
            Probability of taking an exploratory action.
        :param decay: float
            Decayment rate of epsilon.
        :param alpha: float
            Learning rate.
        :param gamma: float
            Discount factor.
        """

        self.env = env
        self.mapper = mapper
        self.k = k
        self.episodes = episodes
        self.max_steps = max_steps
        self.epsilon = epsilon
        self.decay = decay
        self.alpha = alpha
        self.gamma = gamma
        self.actions = [0.0, 1.0, 2.0]

        self.L = np.random.rand(self.mapper.n_rows, k) / 10
        self.R = np.random.rand(k, self.mapper.n_cols) / 10
        self.Q_hat = self.L @ self.R

        self.rewards = []
        self.steps = []
        self.rewards_greedy = []
        self.steps_greedy = []
        self.l_norms = []
        self.r_norms = []

    def get_max_action(self, state):
        """
        :param state: np.array
            Environment state.
        :return int
            Action to take.
        """

        qs = []
        for a in self.actions:
            idx = self.mapper.get_matrix_index(state, a)
            qs.append(self.Q_hat[idx])
        return np.argmax(qs)

    def choose_action(self, state):
        """
        :param state: np.array
            Environment state.
        :return int
            Action to take.
        """

        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()

        return self.get_max_action(state)

    def train(self):

        for episode in range(self.episodes):

            s = self.env.reset()
            cumm_reward = 0

            for step in range(self.max_steps):

                a = self.choose_action(s)

                s_prime, r, done, _ = self.env.step(a)

                theta_0 = np.arctan(s_prime[1] / s_prime[0])
                theta_1 = np.arctan(s_prime[3] / s_prime[2])
                r = -(-1 - np.cos(theta_0) - np.cos(theta_1 + theta_0)) ** 2 / 10
                if done:
                    r += 100

                cumm_reward += r

                mat_idx = mapper.get_matrix_index(s, a)
                mat_idx_prime = mapper.get_matrix_index(s_prime, self.get_max_action(s_prime))

                q_h = self.Q_hat[mat_idx]
                q_b = r + self.gamma * self.Q_hat[mat_idx_prime]

                self.L[mat_idx[0], :] += self.alpha * (q_b - q_h) * self.R[:, mat_idx[
                    1]]/np.linalg.norm(self.R[:, mat_idx[1]])
                self.R[:, mat_idx[1]] += self.alpha * (q_b - q_h) * self.L[mat_idx[0],
                                                                    :]/np.linalg.norm(self.L[mat_idx[0], :])

                s = s_prime

                self.Q_hat[mat_idx[0], :] = self.L[mat_idx[0], :] @ self.R
                self.Q_hat[:, mat_idx[1]] = self.L @ self.R[:, mat_idx[1]]

                self.epsilon *= self.decay

                if done:
                    break

            self.rewards.append(cumm_reward)
            self.steps.append(step)
            self.l_norms.append(np.linalg.norm(self.L, 'fro'))
            self.r_norms.append(np.linalg.norm(self.R, 'fro'))

            if (episode % 10) == 0:
                r, s = self.run_greedy(self.max_steps)
                self.rewards_greedy.append(r)
                self.steps_greedy.append(s)

    def run_greedy(self, n_steps):
        """
        :param n_steps: int
            Number of steps of the greedy episode.
        :return: tuple
            Cumulative reward of the episode and number of steps.
        """

        s = self.env.reset()
        cum_r = 0

        for step in range(n_steps):

            a = self.get_max_action(s)
            s_prime, r, done, info = self.env.step(a)

            theta_0 = np.arctan(s_prime[1] / s_prime[0])
            theta_1 = np.arctan(s_prime[3] / s_prime[2])

            r = -(-1 - np.cos(theta_0) - np.cos(theta_1 + theta_0)) ** 2 / 10
            if done:
                r += 100

            cum_r += r

            if done:
                break

            s = s_prime

        return cum_r, step

    def test(self, n_steps):
        """
        :param n_steps: int
            Number of steps of the greedy episode.
        :return: tuple
            Cumulative reward of the episode and number of steps.
        """

        s = self.env.reset()

        for _ in range(n_steps):

            a = self.get_max_action(s)
            s_prime, r, done, info = self.env.step(a)
            PIL.Image.fromarray(self.env.render(mode='rgb_array')).resize((320, 420))

            if done:
                break

            s = s_prime

        self.env.close()


class Buffer:
    def __init__(self, size):
        """
        :param size: int
            Size of the buffer.
        """

        self.buffer = []
        self.limit = size

    def sample_buffer(self, n):
        """
        :param n: int
            Number of samples to take.
        :return: array
            Array of samples of experiences.
        """

        indices = np.random.randint(0, len(self.buffer), size=n)
        return np.array([self.buffer[index] for index in indices])

    def store_buffer(self, item):
        """
        :param item: tuple
            Tuple of state-action-reward-state.
        """

        if len(self.buffer) > self.limit:
            self.buffer.pop(0)

        self.buffer.append(item)


class Dqn:
    def __init__(self,
                 env,
                 model,
                 buffer,
                 episodes=5000,
                 max_steps=1000,
                 gamma=0.99,
                 epsilon=1.0,
                 decayment_rate=0.99999,
                 batch_size=12):
        """
        :param env: gym.envs
            OpenAI Gym environment.
        :param model: Keras.models
            NN Function approximator.
        :param buffer: Buffer
            Buffer to store experiences.
        :param episodes: int
            Number of episodes.
        :param max_steps: int
            Maximum number of steps per episode.
        :param gamma: float
            Discount factor.
        :param epsilon: float
            Probability of taking an exploratory action.
        :param decayment_rate: float
            Decayment rate of epsilon.
        :param batch_size: int
            Size of the training batch.
        """

        self.model = model
        self.buffer = buffer
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.decayment_rate = decayment_rate
        self.episodes = episodes
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n

        self.rewards = []
        self.steps = []
        self.rewards_greedy = []
        self.steps_greedy = []

    def choose_action(self, st):
        """
        :param state: np.array
            Environment state.
        :return int
            Action to take.
        """

        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()

        return np.argmax(self.model.predict(st)[0])

    def train(self):

        for episode in range(self.episodes):

            cum_r = 0
            s = self.env.reset()
            s = np.reshape(s, [1, self.n_states])

            for step in range(self.max_steps):

                a = self.choose_action(s)
                s_prime, r, done, info = self.env.step(a)
                theta_0 = np.arctan(s_prime[1] / s_prime[0])
                theta_1 = np.arctan(s_prime[3] / s_prime[2])
                r = -(-1 - np.cos(theta_0) - np.cos(theta_1 + theta_0)) ** 2 / 10

                if done:
                    r += 100

                cum_r += r

                s_prime = np.reshape(s_prime, [1, self.n_states])
                self.buffer.store_buffer([s[0], s_prime[0], a, r])

                if episode > 1:
                    snap = self.buffer.sample_buffer(self.batch_size)
                    st, nst, at, rt = np.vstack(snap[:, 0]), np.vstack(snap[:, 1]), snap[:, 2], snap[:, 3]
                    target_action = rt + self.gamma * np.amax(self.model.predict(nst), axis=1)
                    target = self.model.predict(st)
                    for i in range(self.batch_size):
                        target[i, at[i]] = target_action[i]
                    self.model.fit(st, target, verbose=False)

                s = s_prime

                self.epsilon *= self.decayment_rate

                if done:
                    break

            self.rewards.append(cum_r)
            self.steps.append(step)

            if (episode % 10) == 0:
                greedy_r, greedy_steps = self.run_greedy(self.model, self.max_steps)
                self.rewards_greedy.append(greedy_r)
                self.steps_greedy.append(greedy_steps)

    def run_greedy(self, n_steps):
        """
        :param n_steps: int
            Number of steps of the greedy episode.
        :return: tuple
            Cumulative reward of the episode and number of steps.
        """

        s = self.env.reset()
        s = np.reshape(s, [1, self.n_states])
        cum_r = 0

        for i in range(n_steps):
            a = np.argmax(self.model.predict(s)[0])
            s_prime, r, done, info = self.env.step(a)

            theta_0 = np.arctan(s_prime[1] / s_prime[0])
            theta_1 = np.arctan(s_prime[3] / s_prime[2])
            r = -(-1 - np.cos(theta_0) - np.cos(theta_1 + theta_0)) ** 2 / 10

            if done:
                r += 100

            s_prime = np.reshape(s_prime, [1, self.n_states])

            cum_r += r
            s = s_prime

            if done:
                break

        return cum_r, i

    def test(self, n_steps):
        """
        :param n_steps: int
            Number of steps of the greedy episode.
        """

        state = self.env.reset()
        state = np.reshape(state, [1, self.n_states])

        for _ in range(n_steps):
            action = np.argmax(self.model.predict(state)[0])
            new_state, reward, done, info = self.env.step(action)
            PIL.Image.fromarray(self.env.render(mode='rgb_array')).resize((320, 420))
            state = new_state
            state = np.reshape(state, [1, self.n_states])

            if done:
                break

        self.env.close()


class Mapper:
    def __init__(self):

        cos_theta_1 = np.around(np.arange(-1.0, 1.0 + 0.2, 0.2), 1) + 0.
        sin_theta_1 = np.around(np.arange(-1.0, 1.0 + 0.2, 0.2), 1) + 0.
        cos_theta_2 = np.around(np.arange(-1.0, 1.0 + 0.2, 0.2), 1) + 0.
        sin_theta_2 = np.around(np.arange(-1.0, 1.0 + 0.2, 0.2), 1) + 0.
        theta_dot_1 = np.around(np.arange(-14.0, 14.0 + 2.0, 2.0), 1) + 0.
        theta_dot_2 = np.around(np.arange(-30.0, 30.0 + 2.0, 2.0), 1) + 0.
        push = np.around(np.arange(0.0, 2.0 + 1.0, 1.0), 1) + 0.

        grid = [cos_theta_1, sin_theta_1, cos_theta_2, sin_theta_2, theta_dot_1, theta_dot_2, push]

        self.forward_map = [combination for combination in itertools.product(*grid)]
        random.shuffle(self.forward_map)

        self.n_rows = int(np.ceil(np.sqrt(len(self.forward_map))))
        self.n_cols = self.n_rows - 1
        matrix_idx = []
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                matrix_idx.append((i, j))

        self.reverse_map = dict(zip(self.forward_map, matrix_idx))

    def get_matrix_index(self, state, action):
        """
        :param state: np.array
            Array containing the state
        :param action: int
            Action
        :return: tuple
            Index of the matrix
        """

        cos_th_1 = -1.0 + 0.2 * (state[0] + 1.0) // 0.2
        sin_th_1 = -1.0 + 0.2 * (state[1] + 1.0) // 0.2
        cos_th_2 = -1.0 + 0.2 * (state[2] + 1.0) // 0.2
        sin_th_2 = -1.0 + 0.2 * (state[3] + 1.0) // 0.2
        th_dot_1 = -14 + 2 * ((state[4] + 14) // 2)
        th_dot_2 = -30 + 2 * ((state[5] + 30) // 2)

        key = (cos_th_1, sin_th_1, cos_th_2, sin_th_2, th_dot_1, th_dot_2, action)
        return self.reverse_map[key]


class Saver:
    @staticmethod
    def save_to_pickle(path, obj):
        """
        :param path: str
            Path to store the object.
        :param obj
            Object to store.
        """
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

    @staticmethod
    def load_from_pickle(path):
        """
        :param path: str
            Path of the object to load.
        """
        with open(path, 'rb') as f:
            return pickle.load(f)

#################################
class TestUtils:

    @staticmethod
    def smooth_signal(signal, w):
        """
        :param signal: list
            Signal to smooth.
        :param w: int
            Window of the smoothing operation.
        :return: tuple
            Mean and standard deviation of the smoothed signal.
        """

        avg = np.array([np.mean(signal[i:i + w]) for i in range(len(signal) - w)])
        std = np.array([np.std(signal[i:i + w]) for i in range(len(signal) - w)])

        return avg, std

    @staticmethod
    def plot_steps(steps, legend, colors, size):
        """
        :param steps: np.array
            Steps to plot.
        :param legend: list
            List of labels to plot.
        :param colors: list
            List of colors to plot.
        :param size: int
            Size of the x-axis to scale it properly.
        """

        plt.figure(figsize=[6, 4])
        plt.grid()
        for i in range(len(colors)):
            plt.plot(np.arange(0, size, 100), steps[i], c=colors[i])
        plt.xlim(0, 25000)
        plt.legend(legend, prop={"size": 12})
        plt.xlabel("Episodes")
        plt.ylabel("(d) Median nº of steps")
        plt.show()

    @staticmethod
    def plot_smoothed_rewards(medians, stds, legend, colors, size):
        """
        :param medians: list
            Array with the medians of the cum. rewards.
        :param stds: list
            Array with the standard devs. of the cum. rewards.
        :param legend: list
            List of labels to plot.
        :param colors: list
            List of colors to plot.
        :param size: int
            Size of the x-axis to scale it properly.
        :return:
        """

        plt.figure(figsize=[6, 4])
        plt.grid()
        for i in range(len(colors)):
            plt.plot(np.arange(0, size * 100, 100), medians[i], c=colors[i])
            plt.fill_between(np.arange(0, size * 100, 100),
                             medians[i] + stds[i],
                             medians[i] - stds[i],
                             color=colors[i],
                             alpha=.1)
        plt.ylim(-45, -5)
        plt.xlim(0, 25000)
        plt.xlabel("Episodes")
        plt.ylabel("Smoothed cumm. reward")
        plt.legend(legend, prop={"size": 12})
        plt.show()

    @staticmethod
    def plot_final_rewards(rewards, legend, colors):
        """
        :param rewards: list
            List with the mean rewards of the trained agents in several greedy episodes
        :param legend: list
            List of labels to plot.
        :param colors: list
            List of colors to plot.
        """

        plt.figure(figsize=[7, 5])
        plt.grid()
        plt.bar(x=np.arange(len(rewards)), height=np.abs(rewards), alpha=.6, color=colors)
        cs = {legend[i]: colors[i] for i in range(len(colors))}
        labels = list(cs.keys())
        handles = [plt.Rectangle((0, 0), 1, 1, color=cs[label], alpha=.6) for label in labels]
        plt.legend(handles, labels, prop={"size": 12})
        plt.ylabel("(e) Cost (negative reward)")
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.show()

    @staticmethod
    def plot_singular_values(s_q, s_lr, legend):
        plt.figure(figsize=[6, 4])
        plt.grid()
        plt.bar(x=np.arange(len(s_q)), height=s_q, alpha=.6, width=.4, color='b')
        plt.bar(x=np.arange(len(s_lr)) + .4, height=s_lr, alpha=.6, width=.4, color='orange')
        plt.legend(legend, prop={"size": 12})
        plt.ylabel("Normalized singular value")
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.show()


class PendulumEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, g=10.0):
        self.max_speed = 8
        self.max_torque = 2.
        self.dt = .05
        self.g = g
        self.m = 1.
        self.l = 1.
        self.viewer = None

        high = np.array([1., 1., self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque,
            high=self.max_torque, shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        costs = self.angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)

        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        self.state = np.array([newth, newthdot])
        return self._get_obs(), -costs, False, {}

    def reset(self, upright=False):
        high = np.array([np.pi, 1])
        if upright:
            self.state = [np.random.rand()/100, np.random.rand()/100]
        else:
            self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = "resources/clockwise.png"
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


    def angle_normalize(self, x):
        return ((x+np.pi) % (2*np.pi)) - np.pi

