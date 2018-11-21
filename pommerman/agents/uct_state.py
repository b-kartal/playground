from . import uct_helper_utils
from . import uct_parameters

from .. import agents
from .. import characters
from .. import constants

import numpy as np
from random import randint


class StateHelper(object):

    def precompute_possible_actions(self, board):
        listoflistoflists = []
        for i in range(0, constants.BOARD_SIZE):
            sublist = []
            for j in range(0, constants.BOARD_SIZE):
                action_list = [constants.Action.Stop.value]  # stay action
                if i - 1 >= 0 and board[i - 1][j] != constants.Item.Rigid.value:  # north
                    action_list.append(constants.Action.Up.value)
                if i + 1 < constants.BOARD_SIZE and board[i + 1][
                    j] != constants.Item.Rigid.value:  # south
                    action_list.append(constants.Action.Down.value)
                if j - 1 >= 0 and board[i][j - 1] != constants.Item.Rigid.value:  # west
                    action_list.append(constants.Action.Left.value)
                if j + 1 < constants.BOARD_SIZE and board[i][
                    j + 1] != constants.Item.Rigid.value:  # east
                    action_list.append(constants.Action.Right.value)

                sublist.append(action_list)
            listoflistoflists.append(sublist)

        return listoflistoflists

    def __init__(self, observation, game_tracker):

        self.sim_joint_obs = {}

        self.state_game_tracker = game_tracker # pointer to the game tracker to utilize

        self.sim_my_position = tuple(observation['position'])
        self.sim_board = np.array(observation['board'])

        self.lookup_possible_acts = self.precompute_possible_actions(self.sim_board)

        self.sim_enemy_locations = uct_helper_utils._enemies_positions(self.sim_board, tuple(observation['enemies']))
        self._reserve_agents = [agents.RandomAgent(), agents.RandomAgent(), agents.RandomAgent(), agents.RandomAgent()]

        for i in range(4):
            self._reserve_agents[i].init_agent(i, constants.GameType.FFA)

        # TODO populate only alive enemies

        for i in range(len(self.sim_enemy_locations)):
            if uct_parameters.DEBUG_MODE:
                print(i,'th enemy at', self.sim_enemy_locations[i])

        self.sim_bombs_dict = uct_helper_utils.convert_bombs(np.array(observation['bomb_blast_strength']))
        self.sim_enemies = [constants.Item(e) for e in observation['enemies']]

        if uct_parameters.DEBUG_MODE:
            print('enemies are', self.sim_enemies)

        game_tracker_flames = self.state_game_tracker.global_flame_map
        self.sim_flames_ind = np.transpose(np.nonzero(game_tracker_flames)) # get indices of flames

        #if uct_parameters.DEBUG_MODE:
        #    print('flames are',self.sim_flames_dict)

        self.sim_ammo = int(observation['ammo'])

        self.sim_blast_strength = int(observation['blast_strength'])
        self.sim_actions_for_four = [None] * 4 # TODO  set it to the number of remaining agents -  must be overridden

        self.sim_agent_list = [] # starts with uct dummy agent - first agent is our agent indeed
        self.sim_agent_list.append(self._reserve_agents[0])

        for i in range(len(self.sim_enemy_locations)): # go over all enemies EXCLUDING recently dead ones
            self.sim_agent_list.append(self._reserve_agents[i+1])

        self.sim_bombs = []
        for i in range(len(self.sim_bombs_dict)): # TODO associate the bomb with the bomber efficiently
            self.sim_bombs.append(characters.Bomb(self.sim_agent_list[randint(0,len(self.sim_agent_list)-1)], self.sim_bombs_dict[i]['position'],
                                                  observation['bomb_life'][self.sim_bombs_dict[i]['position'][0]][self.sim_bombs_dict[i]['position'][1]],
                                                  self.sim_bombs_dict[i]['blast_strength'], moving_direction=None))

        self.sim_flames = []
        for i in range(np.count_nonzero(game_tracker_flames)):
            self.sim_flames.append(characters.Flame(tuple(self.sim_flames_ind[i]), life=game_tracker_flames[self.sim_flames_ind[i][0]][self.sim_flames_ind[i][1]]))

        self.sim_items, self.sim_dist, self.sim_prev = uct_helper_utils._djikstra(self.sim_board, self.sim_my_position, self.sim_bombs_dict, self.sim_enemies, depth=8)


    def reset_obs(self,observation):

        self.sim_my_position = tuple(observation['position'])
        self.sim_board = np.array(observation['board'])
        self.sim_bombs_dict = uct_helper_utils.convert_bombs(np.array(observation['bomb_blast_strength']))
        self.sim_enemies = [constants.Item(e) for e in observation['enemies']]
        self.sim_enemy_locations = uct_helper_utils._enemies_positions(self.sim_board, tuple(observation['enemies']))

        #self.sim_flames_dict = uct_helper_utils.convert_flames(uct_helper_utils._flame(self.sim_board))
        game_tracker_flames = self.state_game_tracker.global_flame_map
        self.sim_flames_ind = np.transpose(np.nonzero(game_tracker_flames)) # get indices of flames

        self.sim_ammo = int(observation['ammo'])
        self.sim_blast_strength = int(observation['blast_strength'])
        self.sim_items, self.sim_dist, self.sim_prev = uct_helper_utils._djikstra(self.sim_board, self.sim_my_position, self.sim_bombs_dict, self.sim_enemies, depth=8)

        # TODO opponent modeling must fill the information correctly here

        # TODO Tricky - how to track bomb bomber relation to reset these values correctly?
        # Agent Modeling has to update this part
        # TODO : Associate bombs with enemies- correlate bomb lifes with bomb & enemy locations

        self._reserve_agents[0].set_start_position(self.sim_my_position)
        self._reserve_agents[0].reset(self.sim_ammo, True, self.sim_blast_strength, observation['can_kick'])
        self.sim_actions_for_four = [None] * 4
        self.sim_agent_list = [self._reserve_agents[0]]  # first agent is our agent indeed

        for i in range(len(self.sim_enemy_locations)):  # go over all enemies EXCLUDING recently dead ones
            self._reserve_agents[i+1].set_start_position(self.sim_enemy_locations[i])
            self._reserve_agents[i+1].reset(1, is_alive=True, blast_strength=None, can_kick=False)
            self.sim_agent_list.append(self._reserve_agents[i+1])

        self.sim_bombs = []
        for i in range(len(self.sim_bombs_dict)): # TODO currently moving bombs do not transfer to the UCT as moving.
            self.sim_bombs.append(characters.Bomb(self.sim_agent_list[randint(0,len(self.sim_agent_list)-1)], self.sim_bombs_dict[i]['position'],
                                                   observation['bomb_life'][self.sim_bombs_dict[i]['position'][0]][
                                                   self.sim_bombs_dict[i]['position'][1]],
                                                   self.sim_bombs_dict[i]['blast_strength'], moving_direction=None))

        self.sim_flames = []
        for i in range(np.count_nonzero(game_tracker_flames)):
            self.sim_flames.append(characters.Flame(tuple(self.sim_flames_ind[i]), life=game_tracker_flames[self.sim_flames_ind[i][0]][self.sim_flames_ind[i][1]]))  # now flames have correct lifetimes!!!
