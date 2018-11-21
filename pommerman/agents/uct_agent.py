from . import BaseAgent
from .. import constants
import time
import cProfile

from . import uct_search
from . import uct_parameters
from . import uct_helper_utils
from . import game_tracker

import copy

from random import choice, choices, uniform, randint # to obtain multiple randoms with repetition >= Python 3.6

# TODO override run from forward_model to speed up the search - currently too bulky return especially for rollout - likely major bottleneck!
# TODO add epsilon-UCB or other methods
# TODO might enable re-planning based on local environment change - such as woods being destroyed

global PROFILING

PROFILING = False

class UctAgent(BaseAgent):

    def __init__(self, *args, **kwargs):
        super(UctAgent, self).__init__(*args, **kwargs)
        self.turnSize = 0
        self.uctSearcher = None
        self.game_tracker = None
        print(f"uct agent has a budget of {uct_parameters.UCT_BUDGET}")

        if PROFILING:
            self.pr = cProfile.Profile()
            self.pr.enable()

    def act(self, obs, action_space):

        if self.turnSize == 0:
            self.game_tracker = game_tracker.GameTracker(obs) # init the tracker
            self.game_tracker.print_agents()
            self.uctSearcher = uct_search.MCTS(obs, self.game_tracker)  # try to keep useful portion of the tree during game play
            #self.board_prior_safety = uct_helper_utils.board_analyze(self.uctSearcher.game_state.sim_board)


        time_start = time.time()
        self.turnSize += 1

        self.game_tracker.run(obs)
        #self.bombersTracker.print_agents()

        action_to_take = self.uctSearcher.run(uct_parameters.UCT_BUDGET, obs, action_space)  #100 being the number of nodes explored - this will be tuned
        time_finish = time.time()

        if PROFILING and self.turnSize == 25:
            self.pr.disable()
            self.pr.dump_stats('profile.pstat')

        print("Turn ", self.turnSize, "TAKING ACTION", constants.Action(action_to_take).name, " took ", int(1000*(time_finish-time_start)), " miliseconds  \n\n ")

        return constants.Action(action_to_take).value

    def episode_end(self, reward):
        print('reward at the end of episode is ', reward)
