import random

from agent import Agent
from oxono import Game


class ZeroAgent(Agent):
    def __init__(self, player):
        super().__init__(player)

    def act(self, state, remaining_time):
        actions = list(Game.actions(state))
        return random.choice(actions)
