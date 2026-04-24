import os
import time

import torch

from agent import Agent
from files_99942500_26482510.athenan_net import R2AthenanNet
from files_99942500_26482510.athenan_tree import AthenanTree
from oxono import Game

MODEL_PATH = "files_99942500_26482510/athenan_oxono_model.pth"


class AthenanAgent(Agent):
    def __init__(self, player):
        self.player = player
        self.net = R2AthenanNet()

        if os.path.exists(MODEL_PATH):
            self.net.load_state_dict(
                torch.load(
                    MODEL_PATH,
                    map_location="cpu",
                )
            )
        #     print("Athenan model loaded successfully.")
        # else:
        #     print("Athenan model file not found. Starting with untrained model.")

        self.net.eval()

    def act(self, state, remaining_time):
        t0 = time.time()
        for move in Game.actions(state):
            test_state = state.copy()
            Game.apply(test_state, move)
            if Game.is_terminal(test_state):
                return move

        t1 = time.time()
        elapsed_time = t1 - t0
        remaining_time -= elapsed_time

        pieces_remaining = state.pieces_x[self.player] + state.pieces_o[self.player]
        time_to_act = (
            remaining_time / pieces_remaining
            if pieces_remaining > 1
            else remaining_time / 2
        )

        deadline = time.time() + time_to_act

        tree = AthenanTree(state, self.player, self.net, device="cpu")

        # print(f"Time allocated for move: {time_to_act:.2f} seconds. Starting search...")
        while time.time() < deadline:
            tree.principal_variation_search()

        best_move = None
        max_visits = -1

        for action, child in tree.root.children.items():
            # print(f"Move: {action}, Visits: {child.visits}, Value: {child.value}")
            if child.visits > max_visits:
                max_visits = child.visits
                best_move = action

        if best_move is None:
            best_move = list(Game.actions(state))[0]

        return best_move
