import random
import time

import numpy as np

try:
    from files_99942500_26482510.oxononet import predict, probs_to_dict
except ImportError:
    predict = probs_to_dict = None

from files_99942500_26482510.zobrist import compute_zobrist_hash, generate_zobrist_table
from files_99942500_26482510.heuristic import MAX_HEURISTIC_VALUE
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
    def __init__(self, net=None, all_actions=None, eval_fn=None,
                 use_normalization=True, use_heuristic_priors=True,
                 use_early_playout=False, playout_depth=5):
        self.root = Node()
        self.net = net
        self.all_actions = all_actions
        self.cache = {}
        self.eval_fn = eval_fn
        self.use_normalization = use_normalization
        self.use_heuristic_priors = use_heuristic_priors
        self.use_early_playout = use_early_playout
        self.playout_depth = playout_depth

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

    def _early_playout_value(self, state: State, player: int) -> float:
        """
        Play PLAYOUT_DEPTH random moves from state, then evaluate.
        Gives a smoother value estimate than pure static evaluation by
        sampling a short trajectory into the future.
        """
        current = state.copy()
        for _ in range(self.playout_depth):
            if Game.is_terminal(current):
                return float(Game.utility(current, player) * MAX_HEURISTIC_VALUE)
            move = random.choice(list(Game.actions(current)))
            Game.apply(current, move)
        return float(self.eval_fn(current, player))

    def run(
        self,
        state: State,
        time_simulation: float | None = None,
        num_simulations: int = 100,
        c_puct: float = 1.5,
        device="cpu",
        player=None,
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
                # for zero agent
                if self.net is not None:
                    probs, values = self.get_prediction_with_cache(
                        current_state, legal_moves, device=device
                    )
                    value = float(values[0])
                    action_probs = probs_to_dict(probs, self.all_actions, legal_moves)
                    node.generate_children(action_probs)
                # for mcts agent
                elif self.eval_fn is not None and player is not None:
                    if self.use_early_playout:
                        raw_value = self._early_playout_value(current_state, player)
                    else:
                        raw_value = self.eval_fn(current_state, player)

                    if self.use_normalization:
                        value = max(-1.0, min(1.0, raw_value / MAX_HEURISTIC_VALUE))
                    else:
                        value = raw_value

                    if self.use_heuristic_priors:
                        moves_by_score = {}
                        for move in legal_moves:
                            next_state = current_state.copy()
                            Game.apply(next_state, move)
                            moves_by_score[move] = self.eval_fn(next_state, player)

                        min_score = min(moves_by_score.values())
                        positive_shifted_scores = {
                            move: score - min_score + 1.0
                            for move, score in moves_by_score.items()
                        }
                        total = sum(positive_shifted_scores.values())
                        action_probs = {
                            move: score / total
                            for move, score in positive_shifted_scores.items()
                        }
                    else:
                        uniform_prob = 1.0 / len(legal_moves)
                        action_probs = {move: uniform_prob for move in legal_moves}

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
