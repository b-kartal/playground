from pommerman.agents import BaseAgent
import numpy as np

from . import dqn_agent_utilities

from . import pytorch_dqn_learner

debug_dqn_agent = False


class DqnAgent(BaseAgent):

    def __init__(self, *args, **kwargs):
        super(DqnAgent, self).__init__(*args, **kwargs)
        self.game_agent_id = None # set this from board observation
        #self.boardSize = None
        self.testing = False
        self.currentTimeStep = 0
        self.currentEpisode = 0
        self.dqn_engine = pytorch_dqn_learner.dqn_learning() # this is the interfacing class between dqnAgent and general DqnLearning algorithm (domain agnostic hopefully)

        self.dqn_engine.episode_durations = []
        self.dqn_engine.episode_rewards = []

        try:
            self.dqn_engine.load_trained_model(moreTraining=True)  # if a model file exists
        except:
            print("no model file found")

    def act(self, obs, action_space): # can exploit or explore based on dqn algorithm
        # set current state - action is set from dqn_engine act method

        if self.currentTimeStep == 0: # set agentId from game board
            our_agent_pos = np.array(obs['position'])
            board = np.array(obs['board'])
            self.game_agent_id = board[our_agent_pos[0]][our_agent_pos[1]]
            self.dqn_engine.current_state = dqn_agent_utilities.generate_NN_input(self.game_agent_id, obs, self.currentTimeStep)
        else:
            self.dqn_engine.next_state = dqn_agent_utilities.generate_NN_input(self.game_agent_id, obs, self.currentTimeStep)
            self.dqn_engine.reward = 0
            self.dqn_engine.save_to_buffer_learn(self.currentEpisode)
            # prepare for the next interaction
            self.dqn_engine.current_state = dqn_agent_utilities.generate_NN_input(self.game_agent_id, obs, self.currentTimeStep) # for the next transition

        self.currentTimeStep += 1
        if debug_dqn_agent:
            print("passed to dqn learner to act")
        return self.dqn_engine.act(self.dqn_engine.current_state, action_space.n, self.testing)  # 6 is the action_space for Pommerman

    def episode_end(self, reward):


        self.dqn_engine.next_state = None # To make sure NN approximation for V(terminal) = 0
        self.dqn_engine.reward = reward

        if debug_dqn_agent:
            print(f"episode reward is {reward}")

        self.dqn_engine.save_to_buffer_learn(self.currentEpisode)

        self.dqn_engine.episode_rewards.append(reward)
        self.dqn_engine.episode_durations.append(self.currentTimeStep)

        self.dqn_engine.plot_durations()

        if self.currentEpisode % 1000 == 0: # save model every 1000 time  - might save multiple models in a run
            self.dqn_engine.save_trained_model(self.currentEpisode)


        self.currentTimeStep = 0
        self.currentEpisode += 1