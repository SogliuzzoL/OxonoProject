import os

import torch

from agent import Agent
from config import DEVICE, MODEL_PATH
from mcts import MCTS
from oxono import State
from oxononet import OxonoNet, all_possible_actions


class ZeroAgent(Agent):
    def __init__(self, player):
        super().__init__(player)
        self.all_actions = all_possible_actions()
        self.input_channels = 7
        self.net = OxonoNet(self.input_channels, len(self.all_actions))
        if os.path.exists(MODEL_PATH):
            self.net.load_state_dict(torch.load(
                MODEL_PATH,
                map_location=DEVICE)
            )
            self.net.eval()
        self.count_moves = 0

    def act(self, state: State, remaining_time):
        time_to_play = remaining_time / (6 * 6 - self.count_moves + 1)

        mcts = MCTS(self.net, self.all_actions)
        best_action = mcts.get_best_action(
            state,
            time_simulation=time_to_play,
            c_puct=1.5
        )

        self.count_moves += 2
        return best_action
