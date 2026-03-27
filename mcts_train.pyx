import numpy as np
import torch

from config import DEVICE
from oxono_train import Game, State
from oxononet import OxonoNet, probs_to_dict, state_to_tensor

class TrainNode:
    """Représente un nœud MCTS optimisé pour l'entraînement parallèle."""
    def __init__(self, float prior = 0.0):
        self.prior = prior
        self.value = 0.0
        self.visit_count = 0
        self.virtual_loss = 0.0
        self.children = {}
        self.is_terminal = False

    def compute_Q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        # Correction : on ADDITIONNE la virtual_loss pour pénaliser le score Negamax
        return (self.value + self.virtual_loss) / self.visit_count

    def compute_U_value(self, int parent_visit_count, float c_puct = 1.5) -> float:
        return (
            c_puct * self.prior * np.sqrt(parent_visit_count) / (1 + self.visit_count)
        )

    def compute_score(self, int parent_visit_count, float c_puct = 1.5) -> float:
        # Correction : ajout du signe MOINS pour respecter la logique Negamax
        return -self.compute_Q_value() + self.compute_U_value(parent_visit_count, c_puct)

    def generate_children(self, dict action_probs):
        for action, prob in action_probs.items():
            if action not in self.children:
                self.children[action] = TrainNode(prior=prob)


class MCTSTrain:
    """MCTS optimisé pour la génération de données en auto-jeu[cite: 3]."""
    def __init__(self, net: OxonoNet, list all_actions):
        self.root = TrainNode()
        self.net = net
        self.all_actions = all_actions

    def update_root(self, tuple action):
        if action in self.root.children:
            self.root = self.root.children[action]
        else:
            self.root = TrainNode()

    def get_batch_predictions(self, list states, str device):
        """Évaluation groupée des états pour optimiser l'usage du GPU[cite: 4, 5]."""
        tensors = [state_to_tensor(state) for state in states]
        batch_tensor = torch.stack(tensors).to(device)
        with torch.no_grad():
            probs, values = self.net(batch_tensor)
        return probs, values

    def run(
        self,
        state: State,
        int num_simulations = 400,
        float c_puct = 1.5,
        float virtual_loss = 1.0,
        int search_threads = 16,
        str device = DEVICE,
    ):
        cdef int simulations_completed = 0
        cdef float discount = 0.99
        cdef float v
        cdef int path_len

        while simulations_completed < num_simulations:
            paths = []
            leaf_states = []
            current_batch_size = min(search_threads, num_simulations - simulations_completed)

            # --- PHASE 1 : SÉLECTION (PARALLÈLE) ---
            for _ in range(current_batch_size):
                node = self.root
                current_state = state.copy()
                path = [node]

                # Incrément de la racine pour assurer la validité du PUCT [cite: 2, 9]
                node.visit_count += 1

                while node.children and not Game.is_terminal(current_state):
                    # Sélection via score Q + U [cite: 3, 9]
                    best_action, best_child = max(
                        node.children.items(),
                        key=lambda x: x[1].compute_score(node.visit_count, c_puct),
                    )

                    best_child.virtual_loss += virtual_loss
                    best_child.visit_count += 1

                    Game.apply(current_state, best_action)
                    node = best_child
                    path.append(node)

                paths.append(path)
                leaf_states.append(current_state)
                simulations_completed += 1

            # --- PHASE 2 : ÉVALUATION (BATCHED) ---
            valid_indices = [idx for idx, s in enumerate(leaf_states) if not Game.is_terminal(s)]
            valid_states = [leaf_states[idx] for idx in valid_indices]

            probs_batch, values_batch = None, None
            if valid_states:
                probs_batch, values_batch = self.get_batch_predictions(valid_states, device)

            # --- PHASE 3 : RÉTROPROPAGATION ---
            pred_idx = 0
            for i in range(len(paths)):
                path = paths[i]
                leaf_state = leaf_states[i]
                path_len = len(path)
                leaf_node = path[path_len - 1] # Évite l'indice négatif incompatible [cite: 13]

                if Game.is_terminal(leaf_state):
                    leaf_node.is_terminal = True
                    # Valeur pour le joueur ayant effectué le dernier mouvement [cite: 13]
                    v = float(Game.utility(leaf_state, leaf_state.current_player))
                else:
                    probs = probs_batch[pred_idx]
                    v = float(values_batch[pred_idx])
                    pred_idx += 1

                    # Filtrage des coups légaux [cite: 14, 17]
                    legal_moves = list(Game.actions(leaf_state))
                    legal_mask = torch.zeros(probs.size(-1), dtype=torch.bool).to(probs.device)
                    for move in legal_moves:
                        move_tuple = (move[0], (move[1][0], move[1][1]), (move[2][0], move[2][1]))
                        idx_act = self.all_actions.index(move_tuple)
                        legal_mask[idx_act] = True

                    masked_logits = probs.clone()
                    masked_logits[~legal_mask] = float("-inf")
                    masked_probs = torch.nn.functional.softmax(masked_logits, dim=-1)

                    action_probs = probs_to_dict(masked_probs, self.all_actions, legal_moves)
                    leaf_node.generate_children(action_probs)

                # Backpropagation Negamax : inversion du signe à chaque niveau [cite: 20]
                # v est la valeur pour le joueur dont c'est le tour à la feuille.
                for j in range(path_len - 1, -1, -1):
                    node = path[j]
                    if node != self.root:
                        node.virtual_loss -= virtual_loss

                    node.value += v
                    v = -v * discount

    def get_action_probs(self, float temperature = 1.0) -> dict:
        if not self.root.children: return {}
        if temperature == 0:
            best_action = max(self.root.children.items(), key=lambda x: x[1].visit_count)[0]
            return {action: 1.0 if action == best_action else 0.0 for action in self.root.children.keys()}

        counts = {a: c.visit_count**(1/temperature) for a, c in self.root.children.items()}
        total = sum(counts.values())
        return {a: p / total for a, p in counts.items()}
