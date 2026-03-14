import os

import torch

from agent import Agent
from config import DEVICE, MODEL_PATH
from oxono import Game, State
from oxononet import OxonoNet, all_possible_actions, predict, probs_to_dict


class NetworkAgent(Agent):
    def __init__(self, player):
        super().__init__(player)
        self.all_actions = all_possible_actions()

        self.input_channels = 7
        self.net = OxonoNet(self.input_channels, len(self.all_actions))
        if os.path.exists(MODEL_PATH):
            self.net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            self.net.eval()

    def act(self, state: State, remaining_time):
        legal_actions = list(Game.actions(state))
        probs, values = predict(state, self.net, self.all_actions, legal_actions)
        action_probs = probs_to_dict(probs, self.all_actions, legal_actions)
        print(f"Action probabilities: {action_probs}")
        best_action = max(action_probs, key=action_probs.get)

        return best_action
