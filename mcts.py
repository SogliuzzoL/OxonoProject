import time

import numpy as np

from oxono import Game, State
from oxononet import OxonoNet, predict, probs_to_dict


class Node:
    def __init__(self, prior=0.0):
        self.prior = prior
        self.value = 0.0
        self.visit_count = 0
        self.children = {}
        self.is_terminal = False

    def compute_Q_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value / self.visit_count

    def compute_U_value(self, parent_visit_count, c_puct=1.5):
        return c_puct * self.prior * np.sqrt(parent_visit_count) / (1 + self.visit_count)

    def compute_score(self, parent_visit_count, c_puct=1.5):
        return self.compute_Q_value() + self.compute_U_value(parent_visit_count, c_puct)

    def generate_children(self, action_probs):
        for action, prob in action_probs.items():
            if action not in self.children:
                self.children[action] = Node(prior=prob)


class MCTS:
    def __init__(self, net: OxonoNet, all_actions: list):
        self.root = Node()
        self.net = net
        self.all_actions = all_actions

    def run(self, state: State, time_simulation: int, c_puct: float = 1.5):
        t0 = time.time()
        while time.time() - t0 < time_simulation:
            node = self.root
            current_state = state.copy()
            path = [(node, current_state.current_player)]

            while node.children and not Game.is_terminal(current_state):
                current_player = current_state.current_player
                best_action, best_child = max(
                    node.children.items(),
                    key=lambda item:
                    item[1].compute_score(node.visit_count, c_puct)
                )
                Game.apply(current_state, best_action)
                node = best_child
                path.append((node, current_player))

            if Game.is_terminal(current_state):
                node.is_terminal = True
                value = Game.utility(
                    current_state, current_state.current_player)
            else:
                legal_moves = list(Game.actions(current_state))
                probs, values = predict(
                    current_state,
                    self.net,
                    self.all_actions,
                    legal_moves
                )

                probs = probs.cpu().numpy()
                value = float(values.cpu().numpy()[0])

                action_probs = probs_to_dict(
                    probs,
                    self.all_actions,
                    legal_moves
                )

                node.generate_children(action_probs)

            current_player = current_state.current_player

            for path_node, player in reversed(path):
                path_node.visit_count += 1

                if player == current_player:
                    path_node.value += value
                else:
                    path_node.value -= value

    def get_best_action(self):
        best_action, _ = max(
            self.root.children.items(),
            key=lambda item: item[1].visit_count
        )
        return best_action

    def get_action_probs(self, temperature=1.0):
        if temperature == 0:
            best_action = max(self.root.children.items(),
                              key=lambda item: item[1].visit_count)[0]
            return {action: 1.0 if action == best_action else 0.0
                    for action in self.root.children.keys()}

        action_probs = {action: child.visit_count ** (1 / temperature)
                        for action, child in self.root.children.items()}

        total = sum(action_probs.values())

        action_probs = {action: prob / total
                        for action, prob in action_probs.items()}

        action_probs = dict(sorted(action_probs.items(),
                            key=lambda item: item[1], reverse=True))

        return action_probs
