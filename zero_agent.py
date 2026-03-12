import os

import torch

from agent import Agent
from config import DEVICE, MODEL_PATH
from oxononet import OptimalOxonoNet, all_possible_actions, predict, probs_to_dict


class ZeroAgent(Agent):
    def __init__(self, player):
        super().__init__(player)
        self.all_actions = all_possible_actions()
        self.input_channels = 7
        self.net = OptimalOxonoNet(self.input_channels, len(self.all_actions))
        if os.path.exists(MODEL_PATH):
            self.net.load_state_dict(torch.load(
                MODEL_PATH,
                map_location=DEVICE)
            )
            self.net.eval()

    def act(self, state, remaining_time):
        probs, value = predict(state, self.net, self.all_actions)
        probs = probs.cpu().numpy()
        action_probs = probs_to_dict(probs, self.all_actions)

        return list(action_probs.keys())[0]
