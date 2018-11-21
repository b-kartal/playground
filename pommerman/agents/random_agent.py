'''An agent that preforms a random action each step'''
from . import BaseAgent


class RandomAgent(BaseAgent):
    """The Random Agent that returns random actions given an action_space."""
    
    def __init__(self, *args, **kwargs):
        super(RandomAgent, self).__init__(*args, **kwargs)

    def act(self, obs, action_space):
        return action_space.sample()
