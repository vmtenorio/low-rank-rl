import gym
import json
from utils import Mapper, QLearning, Saver


parameters_file = "experiments/exp_18.json"
with open(parameters_file) as j:
    parameters = json.loads(j.read())

mapping = Mapper()
env = gym.make(mapping.environment)

state_map, state_reverse_map = mapping.get_state_map(parameters["step_state"],
                                                     parameters["decimal_state"])

action_map, action_reverse_map = mapping.get_action_map(parameters["step_action"],
                                                        parameters["decimal_action"])

n_states = len(state_map)
n_actions = len(action_map)

q_learner = QLearning(env=env,
                      state_map=state_map,
                      action_map=action_map,
                      state_reverse_map=state_reverse_map,
                      action_reverse_map=action_reverse_map,
                      n_states=n_states,
                      n_actions=n_actions,
                      decimal_state=parameters["decimal_state"],
                      decimal_action=parameters["decimal_action"],
                      step_state=parameters["step_state"],
                      step_action=parameters["step_action"],
                      episodes=parameters["episodes"],
                      max_steps=parameters["max_steps"],
                      decayment_rate=parameters["decay_rate"],
                      epsilon=parameters["epsilon"],
                      epsilon_lower_bound=parameters["epsilon_lower_bound"],
                      exploration_limit=parameters["exploration_limit"],
                      alpha=parameters["alpha"],
                      gamma=parameters["gamma"],
                      action_penalty=parameters["action_penalty"])

q_learner.train()

saver = Saver()
saver.save_to_pickle(parameters["results_path"] + parameters["name_file"] + ".pickle", q_learner)
