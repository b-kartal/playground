from . import BaseAgent
from .. import constants
import random


class RandomAgentNoBombs(BaseAgent):
    """The Random Agent that returns random actions given an action_space."""

    def __init__(self, *args, **kwargs):
        super(RandomAgentNoBombs, self).__init__(*args, **kwargs)
    
    def act(self, obs, action_space):
       directions=[constants.Action.Left, constants.Action.Right,constants.Action.Up,
                   constants.Action.Down, constants.Action.Stop]

       return random.choice(directions).value
