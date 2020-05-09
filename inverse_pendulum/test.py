from utils import Saver

q_learner = Saver.load_from_pickle("results/q_learning_ss02_sa005.pickle")
print(q_learner.epsilon)
