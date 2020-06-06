from matplotlib import rcParams
from utils import Saver, TestUtils

saver = Saver()
test_utils = TestUtils()

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Tahoma']
rcParams['font.size'] = 16

medians_q_learning = saver.load_from_pickle("results/q_learning_medians.pickle")
stds_q_learning = saver.load_from_pickle("results/q_learning_stds.pickle")
frob_errors_q_learning = saver.load_from_pickle("results/q_learning_frob_errors.pickle")

colors = ['b', 'r', 'g', 'y']
epsilons = sorted([float(epsilon) for epsilon in medians_q_learning.keys()])

test_utils.plot_smoothed_steps(epsilons=epsilons,
                               medians=medians_q_learning,
                               stds=stds_q_learning,
                               colors=colors)

test_utils.plot_sfe(epsilons=epsilons,
                    frobenius_errors=frob_errors_q_learning,
                    colors=colors)

medians_lr_learning = saver.load_from_pickle("results/lr_learning_medians.pickle")
stds_lr_learning = saver.load_from_pickle("results/lr_learning_stds.pickle")
frob_errors_lr_learning = saver.load_from_pickle("results/lr_learning_frob_errors.pickle")

test_utils.plot_smoothed_steps(epsilons=epsilons,
                               medians=medians_lr_learning,
                               stds=stds_lr_learning,
                               colors=colors)

test_utils.plot_sfe(epsilons=epsilons,
                    frobenius_errors=frob_errors_lr_learning,
                    colors=colors)