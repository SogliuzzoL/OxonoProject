import time

import numpy as np

from files_99942500_26482510.oxononet import OxonoNet, predict, probs_to_dict
from files_99942500_26482510.zobrist import compute_zobrist_hash, generate_zobrist_table
from oxono import Game, State

ZOBRIST_TABLE = generate_zobrist_table()


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
        return (
            c_puct * self.prior * np.sqrt(parent_visit_count) / (1 + self.visit_count)
        )

    def compute_score(self, parent_visit_count, c_puct=1.5):
        return -self.compute_Q_value() + self.compute_U_value(
            parent_visit_count, c_puct
        )

    def generate_children(self, action_probs):
        for action, prob in action_probs.items():
            if action not in self.children:
                self.children[action] = Node(prior=prob)


class MCTS:
    def __init__(self, net: OxonoNet, all_actions: list):
        self.root = Node()
        self.net = net
        self.all_actions = all_actions
        self.cache = {}

    def update_root(self, action):
        if action in self.root.children:
            self.root = self.root.children[action]
        else:
            self.root = Node()

    def get_prediction_with_cache(self, state: State, legal_moves: list, device="cpu"):
        state_hash = compute_zobrist_hash(ZOBRIST_TABLE, state)
        if state_hash in self.cache:
            return self.cache[state_hash]

        probs, values = predict(
            state, self.net, self.all_actions, legal_moves, device=device
        )
        self.cache[state_hash] = (
            probs.cpu().detach().numpy(),
            values.cpu().detach().numpy(),
        )
        return self.cache[state_hash]

    def run(
        self,
        state: State,
        time_simulation: float | None = None,
        num_simulations: int = 100,
        c_puct: float = 1.5,
        device="cpu",
    ):
        self.terminal_hits = 0
        self.simulations_done = 0

        t0 = time.time()
        i = 0
        while (
            time.time() - t0 < time_simulation
            if time_simulation is not None
            else i < num_simulations
        ):
            i += 1
            self.simulations_done += 1
            node = self.root
            current_state = state.copy()
            path = [node]

            while node.children and not Game.is_terminal(current_state):
                best_action, best_child = max(
                    node.children.items(),
                    key=lambda x: x[1].compute_score(node.visit_count, c_puct),
                )
                Game.apply(current_state, best_action)
                node = best_child
                path.append(node)

            if Game.is_terminal(current_state):
                node.is_terminal = True
                self.terminal_hits += 1
                value = Game.utility(current_state, current_state.current_player)
            else:
                legal_moves = list(Game.actions(current_state))
                probs, values = self.get_prediction_with_cache(
                    current_state, legal_moves, device=device
                )
                value = float(values[0])

                action_probs = probs_to_dict(probs, self.all_actions, legal_moves)
                node.generate_children(action_probs)

            for path_node in reversed(path):
                path_node.visit_count += 1
                path_node.value += value
                value = -value

    def get_best_action(self):
        best_action, _ = max(self.root.children.items(), key=lambda x: x[1].visit_count)
        return best_action

    def get_action_probs(self, temperature=1.0):
        if temperature == 0:
            best_action = max(
                self.root.children.items(), key=lambda x: x[1].visit_count
            )[0]
            return {
                action: 1.0 if action == best_action else 0.0
                for action in self.root.children.keys()
            }

        action_probs = {
            action: child.visit_count ** (1 / temperature)
            for action, child in self.root.children.items()
        }

        total = sum(action_probs.values())

        action_probs = {action: prob / total for action, prob in action_probs.items()}

        action_probs = dict(
            sorted(action_probs.items(), key=lambda x: x[1], reverse=True)
        )

        return action_probs
