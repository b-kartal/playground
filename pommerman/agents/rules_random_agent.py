'''An agent that preforms a random not suicidal action each step'''
from . import BaseAgent
from pommerman.constants import Action
from pommerman.agents import filter_action
import random


class RulesRandomAgent(BaseAgent):
    """ random with filtered actions"""

    def __init__(self, *args, **kwargs):
        super(RulesRandomAgent, self).__init__(*args, **kwargs)


    def act(self, obs, action_space):
        valid_actions=filter_action.get_filtered_actions(obs)
        if len(valid_actions)==0:
            valid_actions.append(Action.Stop.value)
        return random.choice(valid_actions)
