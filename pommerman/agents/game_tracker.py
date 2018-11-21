from . import uct_helper_utils

from .. import characters
from .. import constants

from copy import deepcopy
import numpy as np




class GameTracker(object):
    # Given  observation, initialize agents from the game start
    # Given old map, and new agent positions, update their skills, e.g. can kick or maximum number of ammo
    # This keeps track of our agent as well for convenience  - can be removed for speed-up
    # It keeps track of flames as well
    @staticmethod
    def _update_skill_and_pos(prev_board, bomber_object, bomber_new_pos):
        bomber_object.position = bomber_new_pos
        prev_board_item_value = prev_board[bomber_new_pos[0]][bomber_new_pos[1]]
        if prev_board_item_value == constants.Item.ExtraBomb.value:
            bomber_object.ammo = min(bomber_object.ammo + 1, 10)
        elif prev_board_item_value == constants.Item.IncrRange.value:
            bomber_object.blast_strength = min(bomber_object.blast_strength + 1, 10)
        elif prev_board_item_value == constants.Item.Kick.value:
            bomber_object.can_kick = True

    @staticmethod
    def _manhattan_distance(pos_a, pos_b):
        return abs(pos_a[0]-pos_b[0])+abs(pos_a[1]-pos_b[1])


    def _extract_info(self, observation):
        self.current_board = np.array(observation['board'])
        self.sim_my_position = tuple(observation['position'])
        self.sim_agent_locations = uct_helper_utils._agents_positions(self.current_board)

    def _init_flame_map(self):
        self.prev_flames = np.zeros(self.current_board.shape) # assume that game starts with no flames
        self.current_flames = np.zeros(self.current_board.shape) # flames from observation
        self.global_flame_map = np.zeros(self.current_board.shape) # actual map with lifetimes to query, values from 0 to 2

    def _update_flame_map(self, observation):

        # tick the time for the existing flames - set negative ones to zero if no actual flame
        self.global_flame_map = self.global_flame_map - 1
        self.global_flame_map[self.global_flame_map < 0] = 0

        # get the current flames from the observation - add them to global flame with a lifetime of 2
        self.current_flames = uct_helper_utils._flame(self.current_board)

        # get the new flames and set their lifetimes to 2 steps
        new_flames_from_obs = 2 * ( self.current_flames - self.prev_flames )
        new_flames_from_obs[new_flames_from_obs < 0] = 0 # to prevent negative values on the 1st step after flame gone

        self.global_flame_map += new_flames_from_obs

        #print('previous flames \n', self.prev_flames, '\n')
        #print('current flames \n', self.current_flames, '\n')
        #print('global flames \n', self.global_flame_map, '\n')

        self.prev_flames = deepcopy(self.current_flames)

    def __init__(self, observation):

        self._extract_info(observation)

        self._init_flame_map()

        self.bombermen_list = [characters.Bomber(i+10, constants.GameType.FFA) for i in range(4)] # create 4 agents
        self.distances_to_our_agent = [0] * 5

        sim_enemies = [constants.Item(e) for e in observation['enemies']]

        self.enemy_indices = []
        temp_id_list = [10,11,12,13]
        for enemy in sim_enemies:
            temp_id_list.remove(enemy.value)
            enemy_index = enemy.value - 10
            self.enemy_indices.append(enemy_index)
            self.bombermen_list[enemy_index].set_start_position(self.sim_agent_locations[enemy_index])
            self.bombermen_list[enemy_index].reset()

        self.our_agent_index = temp_id_list[0] - 10
        self.bombermen_list[self.our_agent_index].set_start_position(self.sim_my_position)
        self.bombermen_list[self.our_agent_index].reset()

        for i in range(len(sim_enemies)):
            self.distances_to_our_agent[self.enemy_indices[i]] = self._manhattan_distance(self.bombermen_list[self.our_agent_index].position,self.bombermen_list[self.enemy_indices[i]].position)

        self.prev_board = deepcopy(self.current_board) # save this as prev_board for next time step


    def run(self, current_observation):

        #print(f" \n our agent index is {self.our_agent_index} and enemy indices are {self.enemy_indices}\n")

        self._update_flame_map(current_observation)

        self._extract_info(current_observation)
        self._update_skill_and_pos(self.prev_board, self.bombermen_list[self.our_agent_index], self.sim_my_position) # update our agents skills

        self.enemy_indices = []
        for i in range(len(self.sim_agent_locations)):
            if list(self.sim_agent_locations.keys())[i] != self.our_agent_index:
                self.enemy_indices.append(list(self.sim_agent_locations.keys())[i])


        for i in range(4):
            if i != self.our_agent_index:
                self.bombermen_list[i].is_alive = False # set them all as dead

        for i in range(len(self.enemy_indices)):
            self.distances_to_our_agent[self.enemy_indices[i]] = self._manhattan_distance(self.bombermen_list[self.our_agent_index].position,self.bombermen_list[self.enemy_indices[i]].position)
            self.bombermen_list[self.enemy_indices[i]].is_alive = True # resurrect only the actual alive ones
            self._update_skill_and_pos(self.prev_board,self.bombermen_list[self.enemy_indices[i]],self.sim_agent_locations[self.enemy_indices[i]])

            #print(f"values are {self.enemy_indices[i]} and {self.bombermen_list[self.enemy_indices[i]].is_alive} and "
            #      f"locatiuons are {self.sim_agent_locations} ")

        self.prev_board = deepcopy(self.current_board)

    def query_agent(self, agentId): # returns a dictionary of agent properties
        index_enemy = agentId - 10 # from constants agent ids are 10, 11, 12, 13 ...
        ret = dict()
        ret["max_ammo"] = self.bombermen_list[index_enemy].ammo
        ret["blast_strength"] = self.bombermen_list[index_enemy].blast_strength
        ret["can_kick"] = self.bombermen_list[index_enemy].can_kick
        ret["is_alive"] = self.bombermen_list[index_enemy].is_alive
        ret["position"] = self.bombermen_list[index_enemy].position
        ret["distance_to_us"] = self.distances_to_our_agent[index_enemy]
        return ret

    def query_flames(self, position):
        return self.global_flame_map[position]


    def print_agents(self):
        for i in range(4):
            print(f"id is {self.bombermen_list[i].agent_id} - is alive {self.bombermen_list[i].is_alive} - "
                  f"pos is {self.bombermen_list[i].position} - ammo is {self.bombermen_list[i].ammo} - "
                  f"blast is {self.bombermen_list[i].blast_strength} - can kick {self.bombermen_list[i].can_kick} -"
                  f"distance is {self.distances_to_our_agent[i]}  \n")
