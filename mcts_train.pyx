import numpy as np
import torch

from config import DEVICE
from oxono_train import Game, State
from oxononet import OxonoNet, probs_to_dict, state_to_tensor


class TrainNode:
    """Represents a node in the Monte Carlo Tree Search, equipped with virtual loss."""

    def __init__(self, prior: float = 0.0):
        self.prior = prior
        self.value = 0.0
        self.visit_count = 0
        self.virtual_loss = 0.0
        self.children = {}
        self.is_terminal = False

    def compute_Q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return (self.value - self.virtual_loss) / self.visit_count

    def compute_U_value(self, parent_visit_count: int, c_puct: float = 1.5) -> float:
        return (
            c_puct * self.prior * np.sqrt(parent_visit_count) / (1 + self.visit_count)
        )

    def compute_score(self, parent_visit_count: int, c_puct: float = 1.5) -> float:
        return self.compute_Q_value() + self.compute_U_value(parent_visit_count, c_puct)

    def generate_children(self, action_probs: dict):
        for action, prob in action_probs.items():
            if action not in self.children:
                self.children[action] = TrainNode(prior=prob)


class MCTSTrain:
    """MCTS optimized for parallel self-play generation using dynamic batching."""

    def __init__(self, net: OxonoNet, all_actions: list):
        self.root = TrainNode()
        self.net = net
        self.all_actions = all_actions

    def update_root(self, action: tuple):
        if action in self.root.children:
            self.root = self.root.children[action]
        else:
            self.root = TrainNode()

    def get_batch_predictions(self, states: list, device: str) -> tuple:
        """Evaluates multiple leaf states simultaneously."""
        tensors = [state_to_tensor(state) for state in states]
        batch_tensor = torch.stack(tensors).to(device)

        with torch.no_grad():
            probs, values = self.net(batch_tensor)

        return probs, values

    def run(
        self,
        state: State,
        num_simulations: int = 400,
        c_puct: float = 1.5,
        virtual_loss: float = 1.0,
        search_threads: int = 16,
        device: str = DEVICE,
    ):
        """Executes parallel descents through the tree."""
        simulations_completed = 0

        while simulations_completed < num_simulations:
            paths = []
            leaf_states = []
            current_batch_size = min(
                search_threads, num_simulations - simulations_completed
            )

            for _ in range(current_batch_size):
                node = self.root
                current_state = state.copy()
                path = [(node, current_state.current_player)]

                while node.children and not Game.is_terminal(current_state):
                    current_player = current_state.current_player
                    best_action, best_child = max(
                        node.children.items(),
                        key=lambda x: x[1].compute_score(node.visit_count, c_puct),
                    )

                    best_child.virtual_loss += virtual_loss
                    best_child.visit_count += 1

                    Game.apply(current_state, best_action)
                    node = best_child
                    path.append((node, current_player))

                paths.append(path)
                leaf_states.append(current_state)
                simulations_completed += 1

            valid_states = [s for s in leaf_states if not Game.is_terminal(s)]
            probs_batch, values_batch = None, None

            if valid_states:
                probs_batch, values_batch = self.get_batch_predictions(
                    valid_states, device
                )

            pred_idx = 0
            for path, leaf_state in zip(paths, leaf_states):
                node = path[len(path) - 1][0]
                current_player = leaf_state.current_player

                if Game.is_terminal(leaf_state):
                    node.is_terminal = True
                    actual_value = Game.utility(leaf_state, current_player)
                else:
                    probs = probs_batch[pred_idx]
                    value = values_batch[pred_idx]
                    pred_idx += 1

                    legal_moves = list(Game.actions(leaf_state))

                    legal_mask = torch.zeros(probs.size(-1), dtype=torch.bool).to(
                        probs.device
                    )
                    for move in legal_moves:
                        move_tuple = (
                            move[0],
                            (move[1][0], move[1][1]),
                            (move[2][0], move[2][1]),
                        )
                        idx = self.all_actions.index(move_tuple)
                        legal_mask[idx] = True

                    masked_logits = probs.clone()
                    masked_logits[~legal_mask] = float("-inf")
                    masked_probs = torch.nn.functional.softmax(masked_logits, dim=-1)

                    action_probs = probs_to_dict(
                        masked_probs, self.all_actions, legal_moves
                    )
                    node.generate_children(action_probs)
                    actual_value = float(value)

                discount = 0.99
                for path_node, player in reversed(path):
                    if path_node != self.root:
                        path_node.virtual_loss -= virtual_loss

                    if player == current_player:
                        path_node.value += actual_value
                    else:
                        path_node.value -= actual_value

                    actual_value *= discount

    def get_action_probs(self, temperature: float = 1.0) -> dict:
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
        return dict(sorted(action_probs.items(), key=lambda x: x[1], reverse=True))
