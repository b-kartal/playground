from . import BaseAgent
from pommerman.constants import Action
from pommerman.agents import filter_action
import random


class RulesRandomAgentNoBomb(BaseAgent):
    """ random with filtered actions but no bomb"""

    def __init__(self, *args, **kwargs):
        super(RulesRandomAgentNoBomb, self).__init__(*args, **kwargs)


    def act(self, obs, action_space):
        valid_actions=filter_action.get_filtered_actions(obs)
        if Action.Bomb.value in valid_actions:
            valid_actions.remove(Action.Bomb.value)
        if len(valid_actions)==0:
            valid_actions.append(Action.Stop.value)
        return random.choice(valid_actions)
