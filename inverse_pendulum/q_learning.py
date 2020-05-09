import gym
from utils import Mapper, QLearning, Saver


STEP_STATE = .2
DECIMAL_STATE = 1
STEP_ACTION = .05
DECIMAL_ACTION = 2
EPISODES = 10000
MAX_STEPS = 10000
DECAY_RATE = .9999999
EPSILON = .9
EPSILON_LOWER_BOUND = .1
EXPLORATION_LIMIT = 0
ALPHA = .9
GAMMA = .9

RESULTS_PATH = "results/"
NAME_FILE = "q_learning_ss02_sa005"

mapping = Mapper()
env = gym.make(mapping.environment)

state_map, state_reverse_map = mapping.get_state_map(STEP_STATE, DECIMAL_STATE)
action_map, action_reverse_map = mapping.get_action_map(STEP_ACTION, DECIMAL_ACTION)

n_states = len(state_map)
n_actions = len(action_map)

q_learner = QLearning(env=env,
                      state_map=state_map,
                      action_map=action_map,
                      state_reverse_map=state_reverse_map,
                      action_reverse_map=action_reverse_map,
                      n_states=n_states,
                      n_actions=n_actions,
                      decimal_state=DECIMAL_STATE,
                      decimal_action=DECIMAL_ACTION,
                      step_state=STEP_STATE,
                      step_action=STEP_ACTION,
                      episodes=EPISODES,
                      max_steps=MAX_STEPS,
                      decayment_rate=DECAY_RATE,
                      epsilon=EPSILON,
                      epsilon_lower_bound=EPSILON_LOWER_BOUND,
                      exploration_limit=EXPLORATION_LIMIT,
                      alpha=ALPHA,
                      gamma=GAMMA)

q_learner.train()

saver = Saver()
saver.save_to_pickle(RESULTS_PATH + NAME_FILE + ".pickle", q_learner)
