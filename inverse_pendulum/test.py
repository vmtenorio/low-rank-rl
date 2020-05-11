from utils import Saver, QLearning

q_learner = Saver.load_from_pickle("results/q_learning_ss02_sa01.pickle")
q_learner.test(1000)