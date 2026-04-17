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

    valid_children.sort(key=lambda x: x[1].value, reverse=True)

    epsilon = random.uniform(0, 1)

    index = int((1.0 - epsilon) * len(valid_children))
    index = min(index, len(valid_children) - 1)

    return valid_children[index][0]


class AthenanTrainer:
    def __init__(self, device="cpu"):
        self.device = device
        self.net = R2AthenanNet(board_size=6).to(self.device)
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.01, momentum=0.9)
        self.criterion = torch.nn.MSELoss()

        self.replay_buffer = []
        self.max_matches_in_buffer = 100
        self.batch_size = 3000

        self.mu = 100
        self.delta = 3

        self.load_checkpoint()

    def load_checkpoint(self):
        if os.path.exists(MODEL_PATH):
            self.net.load_state_dict(
                torch.load(MODEL_PATH, map_location=self.device, weights_only=True)
            )
            logger.info("Model loaded successfully from checkpoint.")

        if os.path.exists(OPTIMIZER_PATH):
            self.optimizer.load_state_dict(
                torch.load(OPTIMIZER_PATH, map_location=self.device, weights_only=True)
            )
            logger.info("Optimizer state loaded successfully from checkpoint.")

        if os.path.exists(BUFFER_PATH):
            self.replay_buffer = torch.load(
                BUFFER_PATH, map_location=self.device, weights_only=False
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

            self.replay_buffer.append(match_data)
            if len(self.replay_buffer) > self.max_matches_in_buffer:
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

        return match_tree_data

    def _extract_tree_data(self, node, data_list):
        if node.value is not None:
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
        all_data = []
        weights = []

        for match_index, match_data in enumerate(self.replay_buffer):
            age = len(self.replay_buffer) - match_index
            weight = math.exp(-age / self.mu) + (1 / self.delta)

            all_data.extend(match_data)
            weights.extend([weight] * len(match_data))

        if len(all_data) < self.batch_size:
            logger.info(
                f"Not enough data to train. Current buffer size: {len(all_data)}. Waiting for more matches to be played."
            )
            return

        batch = random.choices(all_data, weights=weights, k=self.batch_size)

        states, target_values = zip(*batch)
        states_tensor = torch.stack(states).to(self.device)
        targets_tensor = torch.stack(target_values).to(self.device)

        self.optimizer.zero_grad()
        predictions = self.net(states_tensor)
        loss = self.criterion(predictions, targets_tensor)
        loss.backward()
        self.optimizer.step()

        logger.info(
            f"Network updated with batch of size {self.batch_size}. Loss: {loss.item():.4f}"
        )


if __name__ == "__main__":
    trainer = AthenanTrainer(device="cuda" if torch.cuda.is_available() else "cpu")

    try:
        trainer.train_loop(
            total_time_seconds=48 * 3600, time_per_move=0.5, save_interval_minutes=15
        )
    except KeyboardInterrupt:
        logger.info(
            "\nInterruption signal received. Saving checkpoint and exiting gracefully..."
        )
    finally:
        trainer.save_checkpoint()
        logger.info("Checkpoint saved. Exiting.")
