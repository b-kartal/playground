global C, EPSILON_FOR_UCB, DEBUG_MODE, UCT_BUDGET, UCT_PARTIAL_EXPAND, UCT_MIXMAX, \
    UCT_ITERATIVE, UCT_MACRO, UCT_REVERSAL, UCT_PARTIAL_EXPAND_THR, UCT_ROLLOUT_LENGTH

C = 1.414213562
EPSILON_FOR_UCB = 0.2  # for epsilon-UCB
UCT_BUDGET = 250
UCT_PARTIAL_EXPAND = True
UCT_PARTIAL_EXPAND_THR = 0.1

UCT_ROLLOUT_LENGTH = 20 # SET AS one more than blast time


DEBUG_MODE = False


# TODO TODO Lots of improvements on the way
# TODO Can we integrate Cython for performance gain
# TODO generalize this so that it takes as arguments different agent players to simulate
# TODO we might want to aggressively search - integrate Iterative MCTS
# TODO !!!! save logs and other statistics about the game play for offline training - experience replay
# TODO augment backprop different statistics to be used for other algorithms such as variance or running average