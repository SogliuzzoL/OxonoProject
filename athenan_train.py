import logging
import math
import os
import random
import time

import torch
import torch.optim as optim

from files_99942500_26482510.athenan_net import R2AthenanNet
from files_99942500_26482510.athenan_tree import AthenanTree
from files_99942500_26482510.utils import state_to_tensor
from oxono import Game, State

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s, %(levelname)s, %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_PATH = "files_99942500_26482510/athenan_oxono_model.pth"
OPTIMIZER_PATH = "files_99942500_26482510/athenan_optimizer.pth"
BUFFER_PATH = "files_99942500_26482510/athenan_replay_buffer.pth"


def ordinal_action_selection(node):
    valid_children = [
        (action, child)
        for action, child in node.children.items()
        if child.value is not None
    ]

    if not valid_children:
        return list(Game.actions(node.state))[0]

    is_maximizing = node.current_player == 0
    valid_children.sort(key=lambda x: x[1].value, reverse=is_maximizing)

    epsilon = random.uniform(0, 1)

    index = int((1.0 - epsilon) * len(valid_children))
    index = min(index, len(valid_children) - 1)

    return valid_children[index][0]


class AthenanTrainer:
    def __init__(self, device="cpu"):
        self.device = device
        self.net = R2AthenanNet(board_size=6).to(self.device)
        self.optimizer = optim.SGD(
            self.net.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-4
        )
        self.criterion = torch.nn.MSELoss()

        self.replay_buffer = []
        self.batch_size = 3000

        self.mu = 100
        self.delta = 3

        self.load_checkpoint()

    def load_checkpoint(self):
        if os.path.exists(MODEL_PATH):
            self.net.load_state_dict(
                torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
            )
            logger.info("Model loaded successfully from checkpoint.")

        if os.path.exists(OPTIMIZER_PATH):
            self.optimizer.load_state_dict(
                torch.load(OPTIMIZER_PATH, map_location="cpu", weights_only=True)
            )
            logger.info("Optimizer state loaded successfully from checkpoint.")

        if os.path.exists(BUFFER_PATH):
            self.replay_buffer = torch.load(
                BUFFER_PATH, map_location="cpu", weights_only=False
            )
            logger.info(
                f"Replay buffer loaded successfully from checkpoint with {len(self.replay_buffer)} matches."
            )

    def save_checkpoint(self):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        torch.save(self.net.state_dict(), MODEL_PATH)
        torch.save(self.optimizer.state_dict(), OPTIMIZER_PATH)
        torch.save(self.replay_buffer, BUFFER_PATH)
        logger.info("Checkpoint saved successfully.")

    def train_loop(
        self, total_time_seconds, time_per_move=1.0, save_interval_minutes=15
    ):
        start_time = time.time()
        last_save_time = time.time()
        save_interval_seconds = save_interval_minutes * 60
        match_count = len(self.replay_buffer)

        logger.info("Starting training loop...")

        while time.time() - start_time < total_time_seconds:
            match_data = self.play_training_match(time_per_move)

            self.replay_buffer.append({"to_learn": match_data, "archived": []})

            if len(self.replay_buffer) > self.mu:
                self.replay_buffer.pop(0)

            match_count += 1
            logger.info(
                f"Match {match_count} completed, {len(match_data)} states extracted. Total matches in buffer: {len(self.replay_buffer)}."
            )

            self.update_network()

            current_time = time.time()
            if current_time - last_save_time >= save_interval_seconds:
                logger.info("Auto-saving checkpoint...")
                self.save_checkpoint()
                last_save_time = current_time

    def play_training_match(self, time_per_move):
        state = State()
        match_tree_data = []

        while not Game.is_terminal(state):
            tree = AthenanTree(state, state.current_player, self.net, self.device)
            deadline = time.time() + time_per_move

            while time.time() < deadline:
                tree.descent_search()

            self._extract_tree_data(tree.root, match_tree_data)

            best_action = ordinal_action_selection(tree.root)
            Game.apply(state, best_action)

            if Game.is_terminal(state):
                values = [val.item() for _, val in match_tree_data]
                positives = sum(1 for v in values if v > 0)
                negatives = sum(1 for v in values if v < 0)
                logger.info(
                    f"Terminal state reached. Extracted {len(match_tree_data)} states: {positives} positive, {negatives} negative."
                )

        augmented_data = []
        for state_tensor, target_value in match_tree_data:
            for i in range(4):
                rotated = torch.rot90(state_tensor, k=i, dims=[1, 2])
                augmented_data.append((rotated, target_value))

                flipped = torch.flip(rotated, dims=[2])
                augmented_data.append((flipped, target_value))

        return augmented_data

    def _extract_tree_data(self, node, data_list):
        if node.value is not None and (node.is_expanded or node.is_terminal):
            data_list.append(
                (
                    state_to_tensor(node.state),
                    torch.tensor([node.value], dtype=torch.float32),
                )
            )

        for child in node.children.values():
            if child.is_expanded:
                self._extract_tree_data(child, data_list)

    def update_network(self):
        if not self.replay_buffer:
            return

        n = 0
        for match in self.replay_buffer:
            total_data = len(match["to_learn"]) + len(match["archived"])
            n += math.ceil((self.delta / self.mu) * total_data)

        g = max(1, math.floor(n / self.batch_size))

        if abs(math.floor(n / g) - self.batch_size) > abs(
            math.floor(n / (g + 1)) - self.batch_size
        ):
            g += 1

        batches = [[] for _ in range(g)]
        available_batch_indices = list(range(g))

        for match in self.replay_buffer:
            total_data = len(match["to_learn"]) + len(match["archived"])
            l = math.ceil((self.delta / self.mu) * total_data)

            for _ in range(l):
                if not match["to_learn"]:
                    match["to_learn"] = match["archived"]
                    match["archived"] = []

                idx = random.randrange(len(match["to_learn"]))
                sample = match["to_learn"].pop(idx)
                match["archived"].append(sample)

                if not available_batch_indices:
                    available_batch_indices = list(range(g))

                b_idx = random.choice(available_batch_indices)
                available_batch_indices.remove(b_idx)
                batches[b_idx].append(sample)

            if not match["to_learn"]:
                match["to_learn"] = match["archived"]
                match["archived"] = []

        mean_loss = 0.0
        valid_batches = 0

        for i, batch in enumerate(batches):
            if not batch:
                continue

            states, target_values = zip(*batch)
            states_tensor = torch.stack(states).to(self.device)
            targets_tensor = torch.stack(target_values).to(self.device)

            self.optimizer.zero_grad()
            predictions = self.net(states_tensor)
            loss = self.criterion(predictions, targets_tensor)
            loss.backward()
            self.optimizer.step()

            mean_loss += loss.item()
            valid_batches += 1

        if valid_batches > 0:
            logger.info(
                f"Network updated with {n} samples across {valid_batches} batches. Average batch size: {n / valid_batches:.2f}. Loss: {mean_loss / valid_batches:.6f}."
            )


if __name__ == "__main__":
    trainer = AthenanTrainer(device="cuda" if torch.cuda.is_available() else "cpu")

    try:
        trainer.train_loop(
            total_time_seconds=21 * 7 * 3600,
            time_per_move=0.5,
            save_interval_minutes=15,
        )
    except KeyboardInterrupt:
        logger.info(
            "\nInterruption signal received. Saving checkpoint and exiting gracefully..."
        )
    finally:
        trainer.save_checkpoint()
        logger.info("Checkpoint saved. Exiting.")
