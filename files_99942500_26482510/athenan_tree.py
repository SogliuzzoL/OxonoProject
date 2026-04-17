import torch

from files_99942500_26482510.utils import state_to_tensor
from files_99942500_26482510.zobrist import compute_zobrist_hash, generate_zobrist_table
from oxono import Game


class AthenanNode:
    def __init__(
        self, state, current_player, parent=None, move_from_parent=None, depth=0
    ):
        self.state = state
        self.current_player = current_player
        self.parent = parent
        self.move_from_parent = move_from_parent
        self.depth = depth

        self.children = {}
        self.value = None
        self.visits = 0
        self.is_expanded = False
        self.is_terminal = Game.is_terminal(state)


class AthenanTree:
    def __init__(self, root_state, current_player, network, device="cpu"):
        self.root = AthenanNode(root_state, current_player, depth=0)
        self.network = network
        self.device = device
        self.root_player = current_player

        self.zobrist_table = generate_zobrist_table()
        self.transposition_table = {}

    def principal_variation_search(self):
        node = self.root
        path = [node]

        while node.is_expanded and not node.is_terminal:
            best_child = None
            is_maximizing = node.current_player == self.root_player

            if is_maximizing:
                best_val = float("-inf")
                for child in node.children.values():
                    if child.value is not None and child.value > best_val:
                        best_val = child.value
                        best_child = child
            else:
                best_val = float("inf")
                for child in node.children.values():
                    if child.value is not None and child.value < best_val:
                        best_val = child.value
                        best_child = child

            if best_child is None:
                break

            node = best_child
            path.append(node)

        if not node.is_expanded and not node.is_terminal:
            self._expand_and_evaluate(node)

        self._backpropagate(path)

        self.root.visits += 1
        if len(path) > 1:
            path[1].visits += 1

    def descent_search(self):
        node = self.root
        path = []

        while not node.is_terminal:
            if not node.is_expanded:
                self._expand_and_evaluate(node)

            path.append(node)

            best_child = None
            is_maximizing = node.current_player == self.root_player

            if is_maximizing:
                best_val = float("-inf")
                for child in node.children.values():
                    if child.value is not None and child.value > best_val:
                        best_val = child.value
                        best_child = child
            else:
                best_val = float("inf")
                for child in node.children.values():
                    if child.value is not None and child.value < best_val:
                        best_val = child.value
                        best_child = child

            if best_child is None:
                break

            node = best_child

        path.append(node)
        self._backpropagate(path)

        self.root.visits += 1

    def _expand_and_evaluate(self, node):
        actions = list(Game.actions(node.state))
        for action in actions:
            next_state = node.state.copy()
            Game.apply(next_state, action)
            child = AthenanNode(
                next_state,
                next_state.current_player,
                parent=node,
                move_from_parent=action,
                depth=node.depth + 1,
            )
            node.children[action] = child

        node.is_expanded = True
        self._evaluate_children(node)

    def _evaluate_children(self, node):
        states_to_evaluate = []
        children_to_evaluate = []

        for child in node.children.values():
            if child.is_terminal:
                base_score = float(Game.utility(child.state, self.root_player))

                if base_score > 0:
                    child.value = base_score - (child.depth * 0.001)
                elif base_score < 0:
                    child.value = base_score + (child.depth * 0.001)
                else:
                    child.value = 0.0
                continue

            state_hash = compute_zobrist_hash(self.zobrist_table, child.state)
            if state_hash in self.transposition_table:
                child.value = self.transposition_table[state_hash]
            else:
                states_to_evaluate.append(state_to_tensor(child.state))
                children_to_evaluate.append((child, state_hash))

        if states_to_evaluate:
            batch_tensor = torch.stack(states_to_evaluate).to(self.device)

            with torch.no_grad():
                predictions = self.network(batch_tensor).squeeze(-1).tolist()

            if not isinstance(predictions, list):
                predictions = [predictions]

            for (child, state_hash), pred_value in zip(
                children_to_evaluate, predictions
            ):
                child.value = pred_value
                self.transposition_table[state_hash] = pred_value

    def _backpropagate(self, path):
        for node in reversed(path[:-1]):
            is_maximizing = node.current_player == self.root_player
            valid_children = [
                c.value for c in node.children.values() if c.value is not None
            ]

            if not valid_children:
                continue

            if is_maximizing:
                node.value = max(valid_children)
            else:
                node.value = min(valid_children)

            state_hash = compute_zobrist_hash(self.zobrist_table, node.state)
            self.transposition_table[state_hash] = node.value
