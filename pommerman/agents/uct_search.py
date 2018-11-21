# UCT-Search with game forward model integrated

from random import choices, uniform

from . import uct_state
from . import uct_tree
from . import uct_parameters
from . import uct_helper_utils

from .. import forward_model
from .. import constants


class MCTS(object):

    def __init__(self, obs, game_tracker):

        self.game_state = uct_state.StateHelper(obs, game_tracker)  # game state object
        self.game_state.reset_obs(obs)

        if uct_parameters.DEBUG_MODE:
            print('world is reset')

        self.frwd_model = forward_model.ForwardModel()  # to simulate the game dynamics during rollout
        self.root_node = None

    def run(self, search_budget, obs, action_space):

        self.root_node = uct_tree.TreeNode(None, None)
        self.game_state.reset_obs(obs)
        self.root_node.available_actions = self.possible_actions()

        if uct_parameters.DEBUG_MODE:
            print('forward model is generated')
            print('tree node is generated ', self.root_node.depth)
            print('\n\n', self.game_state.sim_agent_list)
            print('\n')
            print('printed bombers and agents ')
            print(self.game_state.sim_bombs, ' just printed bombs ')
            print(self.game_state.sim_joint_obs)
            print('received obs for all agents')
            print('forward model run')
            print(self.game_state.sim_actions_for_four)

        for i in range(search_budget):  # Main MCTS part

            if i % 99 == 0 and uct_parameters.DEBUG_MODE:
                print("\n Iteration ", i, " started and root node stats ", self.root_node.visit_count, " win rate ",
                      self.root_node.win_rate / self.root_node.visit_count, " and children size of ",
                      len(self.root_node.children_nodes))
                for i in range(len(self.root_node.children_nodes)):
                    print(i, "th kid has visit count", self.root_node.children_nodes[i].visit_count, " win rate",
                          self.root_node.children_nodes[i].win_rate, " UCB score of ",
                          uct_tree.ucb_value(self.root_node.children_nodes[i]))

            self.game_state.reset_obs(obs)

            node_to_expand_from = self.selection(self.root_node)  # this already called the forward.step() several times

            if uct_parameters.DEBUG_MODE:
                print('flames are', self.game_state.sim_flames)
                print('actions are ', self.game_state.sim_actions_for_four)
                print('check ')
                print(uct_helper_utils.all_actions[len(node_to_expand_from.children_nodes)].value)
                print('check ')

            # game_state.sim_actions_for_four[0] = all_actions[len(node_to_expand_from.children_nodes)].value # for now assume all actions are possible always
            # game_state.sim_actions_for_four = frwd_model.act(game_state.sim_agent_list, game_state.sim_joint_obs, action_space) # TODO HACK omit this as opponents are random agents

            self.game_state.sim_actions_for_four[1:4] = choices([0, 1, 2, 3, 4, 5], k=3)  # TODO This line replaced the forward model act method

            self.game_state.sim_actions_for_four[0] = node_to_expand_from.available_actions[len(node_to_expand_from.children_nodes)]

            new_added_node = node_to_expand_from.add_child_node(self.game_state.sim_actions_for_four[0])  # TODO refactor sim_action[0] hardcoding



            new_added_node.available_actions = self.possible_actions()  # TODO call this next time lazy fashion

            #TODO Hack - as we do not track bomb-bomber relation - set my agent accordingly

            self.game_state.sim_board, self.game_state.sim_agent_list, self.game_state.sim_bombs, self.game_state.sim_items, self.game_state.sim_flames \
                = self.frwd_model.step(self.game_state.sim_actions_for_four, self.game_state.sim_board, self.game_state.sim_agent_list,
                                  self.game_state.sim_bombs,
                                  self.game_state.sim_items, self.game_state.sim_flames)

            rollout_reward = self.rollout(new_added_node, action_space)

            if uct_parameters.DEBUG_MODE:
                print(' rollout reward is ', rollout_reward, " \n \n")

            self.backprop(new_added_node, rollout_reward)

        #self.root_node = self.root_node.best_visit_child() # update the root to the next best child - TODO do we need to prune the remaining sub-trees?
        #self.root_node.parent_node = None
        return constants.Action(self.root_node.best_visit_child().edge_action).value  # if actions are not turned off, then simply return best_visit_child()

    def possible_actions(self):  # precompute this and lookup
        action_list = list(self.game_state.lookup_possible_acts[self.game_state.sim_my_position[0]][self.game_state.sim_my_position[1]])
        if self.game_state.sim_agent_list[0].ammo > 0:
            action_list.append(constants.Action.Bomb.value)

        return action_list

    def selection(self, node):
        # return the node with highest ucb value and do this recursively
        while node.is_full_explored() or ( uct_parameters.UCT_PARTIAL_EXPAND and len(node.children_nodes) > 0 and uniform(0.0, 1.0) < uct_parameters.UCT_PARTIAL_EXPAND_THR ):  # keep recursively selecting until finding a terminal node or a node that can produce more children nodes.

            # act() will return a random action for our agent, override it with tree action here before passing to step() TODO Hack Alert

            node = node.best_ucb_child()
            self.game_state.sim_joint_obs = self.frwd_model.get_observations(self.game_state.sim_board, self.game_state.sim_agent_list,
                                                                   self.game_state.sim_bombs, False, constants.BOARD_SIZE)

            #game_state.sim_actions_for_four = frwd_model.act(game_state.sim_agent_list, game_state.sim_joint_obs, action_space) # TODO this is removed for FFA assuming random agents as opponents - need this for team message
            self.game_state.sim_actions_for_four[0] = constants.Action(node.edge_action).value # override forward_model action with tree action selection

            if uniform(0.0, 1.0) < 1: # currently leave as vanilla
                self.game_state.sim_actions_for_four[1:4] = choices([0, 1, 2, 3, 4, 5], k=3)
            else:
                self.game_state.sim_actions_for_four[1:4] = choices([0, 1, 2, 3, 4], k=3)


            if uct_parameters.DEBUG_MODE:
                print(self.game_state.sim_actions_for_four)


            self.game_state.sim_board, self.game_state.sim_agent_list, self.game_state.sim_bombs, self.game_state.sim_items, self.game_state.sim_flames = \
                self.frwd_model.step(self.game_state.sim_actions_for_four, self.game_state.sim_board, self.game_state.sim_agent_list, self.game_state.sim_bombs,
                                     self.game_state.sim_items, self.game_state.sim_flames) # tick the time and update board and agents

            # TODO also augment here the action taken for the best strategy generated
        #print('selection ended')
        return node # returned node is either a leaf or needs more children

    def rollout(self, bottom_node, action_space):

        def _rollout_reward(game_engine_reward, game_length):

            def _auxilary_reward():  # To incentivize tune-up collection
                sum = 0.0
                sum += min(0.2, (self.game_state.sim_agent_list[
                                     0].blast_strength - constants.DEFAULT_BLAST_STRENGTH) * 0.025)
                if self.game_state.sim_agent_list[0].can_kick is True:
                    sum += 0.10
                return min(sum, 0.4)

            survival_rev = 0.0
            number_of_enemies_left = 0
            for i in range(len(self.game_state.sim_enemy_locations)):
                if self.game_state.sim_agent_list[i + 1].is_alive:
                    number_of_enemies_left += 1

            enemy_killed_reward = (0.75-0.25*number_of_enemies_left)

            if number_of_enemies_left == 0 and self.game_state.sim_agent_list[0].is_alive:
                survival_rev = 1
            elif number_of_enemies_left == 0 and self.game_state.sim_agent_list[0].is_alive is False:
                survival_rev = 0.5 # if all died together it is a tie
            else:
                survival_rev = (game_engine_reward + 1) / 2

            return min(1, survival_rev + _auxilary_reward() + enemy_killed_reward) # TODO weight these two based on state!



        rollout_length = min( (constants.MAX_STEPS - bottom_node.depth), uct_parameters.UCT_ROLLOUT_LENGTH )
        actual_game_length = 0
        for i in range(0, rollout_length):

            self.game_state.sim_joint_obs = self.frwd_model.get_observations(self.game_state.sim_board, self.game_state.sim_agent_list,
                                                                             self.game_state.sim_bombs, False, constants.BOARD_SIZE)


            #self.game_state.sim_actions_for_four = self.frwd_model.act(self.game_state.sim_agent_list, self.game_state.sim_joint_obs, action_space) # TODO Hackthis is removed for FFA assuming random agents as opponents - need this for team message

            # TODO Bias the rollout to place a bomb not very often ...

            if uniform(0.0, 1.0) < 1: # currently leave as vanilla
                self.game_state.sim_actions_for_four[0:4] = choices([0, 1, 2, 3, 4, 5], k=4)
            else:
                self.game_state.sim_actions_for_four[0:4] = choices([0, 1, 2, 3, 4], k=4)

            # current_possible_actions = possible_actions() # returns possible actions for our agent TODO do not provide extra advantage to our agent
            # game_state.sim_actions_for_four[0] = choice(current_possible_actions) # TODO only possible action for our agent

            self.game_state.sim_board, self.game_state.sim_agent_list, self.game_state.sim_bombs, self.game_state.sim_items, self.game_state.sim_flames = \
                self.frwd_model.step(self.game_state.sim_actions_for_four, self.game_state.sim_board, self.game_state.sim_agent_list,
                                     self.game_state.sim_bombs, self.game_state.sim_items, self.game_state.sim_flames)  # tick the time and update board and agents

            done = self.frwd_model.get_done(self.game_state.sim_agent_list,  i, constants.MAX_STEPS, constants.GameType.FFA, 0)

            if done:
                actual_game_length = i
                break
        rewards_for_all = self.frwd_model.get_rewards(self.game_state.sim_agent_list, constants.GameType.FFA, constants.MAX_STEPS, constants.MAX_STEPS)  # call this way as run() finished the game
        if uct_parameters.DEBUG_MODE:
            print('all rewards are', rewards_for_all)

        return _rollout_reward(rewards_for_all[0], actual_game_length) # pass it to rollout reward function

    def backprop(self, bottom_node, reward):
        while bottom_node != self.root_node:
            bottom_node.win_rate += reward
            bottom_node.visit_count += 1
            bottom_node = bottom_node.parent_node

        bottom_node.win_rate += reward  # For root node update
        bottom_node.visit_count += 1  # For root node update

