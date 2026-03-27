import os

import torch

from agent import Agent
from config import MODEL_PATH
from mcts import MCTS, export_mcts_to_graph
from oxono import State
from oxononet import OxonoNet, all_possible_actions

DEVICE = "cpu"


class ZeroAgent(Agent):
    def __init__(self, player):
        super().__init__(player)
        self.all_actions = all_possible_actions()

        self.input_channels = 7
        self.net = OxonoNet(self.input_channels, len(self.all_actions))
        if os.path.exists(MODEL_PATH):
            self.net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            self.net.eval()

        self.mcts = MCTS(self.net, self.all_actions)

        self.count_moves = 0

        self.last_totem_O = None
        self.last_totem_X = None

    def act(self, state: State, remaining_time):
        last_action = state.last_move
        if last_action is not None:
            if state.totem_O != self.last_totem_O:
                last_action = ("O", state.totem_O, state.last_move)
            else:
                last_action = ("X", state.totem_X, state.last_move)
            self.mcts.update_root(last_action)

        self.mcts.run(state, num_simulations=1_000, c_puct=1.5, device=DEVICE)

        best_action = self.mcts.get_best_action()
        # export_mcts_to_graph(
        #     self.mcts, f"mcts_graph_move_{self.count_moves}_player_{self.player}"
        # )

        self.mcts.update_root(best_action)

        self.last_totem_O = state.totem_O if best_action[0] != "O" else best_action[1]
        self.last_totem_X = state.totem_X if best_action[0] != "X" else best_action[1]

        self.count_moves += 2

        return best_action
