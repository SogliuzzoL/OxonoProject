import argparse
import concurrent.futures
import copy
import logging
import os
import pickle
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from config import MODEL_PATH
from mcts import MCTS
from oxono import Game, State
from oxononet import OxonoNet, all_possible_actions, state_to_tensor

# Configuration of the logging system
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s, %(levelname)s, %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BUFFER_PATH = "replay_buffer.pkl"

# Type alias for clarity
TrainingSample = Tuple[np.ndarray, np.ndarray, float]


@dataclass
class AlphaZeroConfig:
    """Encapsulates the hyperparameters for the AlphaZero training pipeline."""

    iterations: int = 50
    games_per_iteration: int = 100
    evaluation_games: int = 20
    win_rate_threshold: float = 0.55
    max_buffer_size: int = 100_000
    mcts_simulations: int = 400
    batch_size: int = 256
    epochs_per_iteration: int = 5
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    max_workers: int = 5


def execute_self_play_episode(
    net: OxonoNet, all_actions: List[Any], num_simulations: int = 100
) -> List[TrainingSample]:
    """
    Executes a single episode of self-play using MCTS to generate training data.

    Args:
        net: The neural network used for policy and value predictions.
        all_actions: A list mapping indices to possible game actions.
        num_simulations: The number of MCTS simulations per move.

    Returns:
        A list of training samples consisting of states, policies, and final values.
    """
    state = State()
    mcts = MCTS(net, all_actions)
    episode_data = []
    round_count = 0

    while not Game.is_terminal(state):
        # High temperature for the first 20 moves to encourage exploration
        temperature = 1.0 if round_count < 20 else 0.1

        mcts.run(state, num_simulations=num_simulations)
        action_probs = mcts.get_action_probs(temperature)

        policy = np.zeros(len(all_actions))
        for action, prob in action_probs.items():
            idx = all_actions.index(action)
            policy[idx] = prob

        tensor_state = state_to_tensor(state)
        episode_data.append([tensor_state, policy, state.current_player])

        actions = list(action_probs.keys())
        probs = list(action_probs.values())

        chosen_action_idx = np.random.choice(len(actions), p=probs)
        action = actions[chosen_action_idx]
        Game.apply(state, action)

        mcts.update_root(action)
        round_count += 1

    winner_utility = Game.utility(state, 1)
    dataset = []

    # Assign the final reward to all states based on the active player
    for tensor_state, policy, player_turn in episode_data:
        value = winner_utility if player_turn == 1 else -winner_utility
        dataset.append((tensor_state, policy, value))

    return dataset


def worker_generate_data(
    state_dict: Dict[str, torch.Tensor], all_actions: List[Any], num_simulations: int
) -> List[TrainingSample]:
    """
    Worker function to run a self-play game in an isolated process.

    Args:
        state_dict: The state dictionary of the neural network to reconstruct.
        all_actions: The list of all legal actions.
        num_simulations: The number of MCTS simulations.

    Returns:
        The generated training dataset ensuring tensors are moved to CPU memory.
    """
    torch.set_num_threads(1)

    worker_net = OxonoNet(7, len(all_actions)).to(DEVICE)
    worker_net.load_state_dict(state_dict)
    worker_net.eval()

    dataset = execute_self_play_episode(
        worker_net, all_actions, num_simulations=num_simulations
    )

    safe_dataset = [
        (state_tensor.cpu().numpy(), policy, value)
        for state_tensor, policy, value in dataset
    ]

    return safe_dataset


def worker_evaluate_game(
    best_model_path: str,
    new_model_path: str,
    all_actions: List[Any],
    num_simulations: int,
    new_is_p1: bool,
) -> str:
    """
    Evaluates the candidate model against the current reference model.
    Models are loaded on the CPU to prevent VRAM overflow during parallel evaluation.

    Args:
        best_model_path: File path to the reference model weights.
        new_model_path: File path to the candidate model weights.
        all_actions: The list of all legal actions.
        num_simulations: MCTS simulations per turn.
        new_is_p1: Boolean indicating if the candidate model plays as player 1.

    Returns:
        A string indicating the winner ('new', 'best', or 'draw').
    """
    torch.set_num_threads(1)
    eval_device = "cpu"

    best_net = OxonoNet(7, len(all_actions)).to(eval_device)
    best_net.load_state_dict(
        torch.load(best_model_path, map_location=eval_device, weights_only=True)
    )
    best_net.eval()

    new_net = OxonoNet(7, len(all_actions)).to(eval_device)
    new_net.load_state_dict(
        torch.load(new_model_path, map_location=eval_device, weights_only=True)
    )
    new_net.eval()

    state = State()
    mcts_new = MCTS(new_net, all_actions)
    mcts_best = MCTS(best_net, all_actions)
    move_count = 0

    while not Game.is_terminal(state):
        is_p1_turn = state.current_player == 1

        if (is_p1_turn and new_is_p1) or (not is_p1_turn and not new_is_p1):
            active_mcts = mcts_new
        else:
            active_mcts = mcts_best

        active_mcts.run(state, num_simulations=num_simulations, device=eval_device)

        eval_temp = 1.0 if move_count < 2 else 0.0
        action_probs = active_mcts.get_action_probs(temperature=eval_temp)

        if eval_temp == 0.0:
            best_action = max(action_probs, key=action_probs.get)
        else:
            actions = list(action_probs.keys())
            probs = list(action_probs.values())
            chosen_idx = np.random.choice(len(actions), p=probs)
            best_action = actions[chosen_idx]

        Game.apply(state, best_action)
        mcts_new.update_root(best_action)
        mcts_best.update_root(best_action)
        move_count += 1

    winner_utility = Game.utility(state, 1)

    if winner_utility == 1:
        return "new" if new_is_p1 else "best"
    elif winner_utility == -1:
        return "best" if new_is_p1 else "new"

    return "draw"


def train_neural_network(
    net: OxonoNet, dataset: List[TrainingSample], config: AlphaZeroConfig
) -> OxonoNet:
    """
    Trains the neural network using the reinforcement learning dataset.

    Args:
        net: The neural network to train.
        dataset: The collection of self-play experiences.
        config: The configuration object containing training hyperparameters.

    Returns:
        The trained neural network.
    """
    states, target_policies, target_values = zip(*dataset)

    states_t = torch.tensor(np.array(states), dtype=torch.float32).to(DEVICE)
    policies_t = torch.tensor(np.array(target_policies), dtype=torch.float32).to(DEVICE)
    values_t = (
        torch.tensor(np.array(target_values), dtype=torch.float32)
        .unsqueeze(1)
        .to(DEVICE)
    )

    train_data = TensorDataset(states_t, policies_t, values_t)
    dataloader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)

    optimizer = optim.Adam(
        net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    # Implement a learning rate scheduler to improve convergence
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    net.train()

    for epoch in range(config.epochs_per_iteration):
        total_loss = 0.0

        for batch_states, batch_target_policies, batch_target_values in tqdm(
            dataloader,
            desc=f"Training Epoch {epoch + 1}/{config.epochs_per_iteration}",
            leave=False,
        ):
            optimizer.zero_grad()

            predicted_policy_logits, predicted_values = net(batch_states)

            value_loss = F.mse_loss(predicted_values, batch_target_values)
            policy_loss = F.cross_entropy(
                predicted_policy_logits, batch_target_policies
            )

            loss = value_loss + policy_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        average_loss = total_loss / len(dataloader)
        logger.info(
            f"Epoch {epoch + 1} completed with average loss: {average_loss:.4f}"
        )

    return net


def conduct_evaluation_tournament(
    best_model_path: str,
    new_model_path: str,
    all_actions: List[Any],
    config: AlphaZeroConfig,
) -> Tuple[int, int, int]:
    """
    Runs a parallel evaluation tournament between the reference and candidate models.

    Args:
        best_model_path: File path to the reference model.
        new_model_path: File path to the candidate model.
        all_actions: All legal actions.
        config: Training configuration object.

    Returns:
        A tuple containing the number of wins for the new model, wins for the best model, and draws.
    """
    new_wins, best_wins, draws = 0, 0, 0

    eval_arguments = [
        (
            best_model_path,
            new_model_path,
            all_actions,
            config.mcts_simulations,
            i % 2 == 0,
        )
        for i in range(config.evaluation_games)
    ]

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=config.max_workers
    ) as executor:
        results = executor.map(
            worker_evaluate_game,
            [arg[0] for arg in eval_arguments],
            [arg[1] for arg in eval_arguments],
            [arg[2] for arg in eval_arguments],
            [arg[3] for arg in eval_arguments],
            [arg[4] for arg in eval_arguments],
        )

        for result in tqdm(
            results, total=config.evaluation_games, desc="Evaluation Tournament"
        ):
            if result == "new":
                new_wins += 1
            elif result == "best":
                best_wins += 1
            else:
                draws += 1

    return new_wins, best_wins, draws


def main():
    """Main execution loop for the AlphaZero training pipeline."""
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="AlphaZero Training for Oxono")
    parser.add_argument(
        "--iterations", type=int, default=100, help="Number of training iterations"
    )
    args = parser.parse_args()

    config = AlphaZeroConfig(iterations=args.iterations)
    all_actions = all_possible_actions()

    best_net = OxonoNet(7, len(all_actions)).to(DEVICE)

    if os.path.exists(MODEL_PATH):
        logger.info(f"Loading reference model from {MODEL_PATH}.")
        best_net.load_state_dict(
            torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        )

    if os.path.exists(BUFFER_PATH):
        logger.info(f"Loading replay buffer from {BUFFER_PATH}.")
        with open(BUFFER_PATH, "rb") as file:
            replay_buffer = pickle.load(file)
    else:
        replay_buffer = deque(maxlen=config.max_buffer_size)

    for iteration in range(config.iterations):
        logger.info(
            f"Starting AlphaZero iteration {iteration + 1} of {config.iterations}."
        )
        best_net.eval()
        iteration_data = []

        logger.info("Generating data through parallel self-play.")
        current_state_dict = {k: v.cpu() for k, v in best_net.state_dict().items()}

        games_arguments = [
            (current_state_dict, all_actions, config.mcts_simulations)
            for _ in range(config.games_per_iteration)
        ]

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=config.max_workers
        ) as executor:
            results = executor.map(
                worker_generate_data,
                [arg[0] for arg in games_arguments],
                [arg[1] for arg in games_arguments],
                [arg[2] for arg in games_arguments],
            )

            for completed_game in tqdm(
                results, total=config.games_per_iteration, desc="Self-Play Generation"
            ):
                iteration_data.extend(completed_game)

        logger.info(f"Successfully generated {config.games_per_iteration} games.")
        replay_buffer.extend(iteration_data)

        with open(BUFFER_PATH, "wb") as file:
            pickle.dump(replay_buffer, file)

        logger.info(f"Replay buffer saved containing {len(replay_buffer)} states.")
        logger.info("Starting candidate model training phase.")

        new_net = copy.deepcopy(best_net)
        train_neural_network(new_net, list(replay_buffer), config)

        logger.info("Training complete, initiating comparative evaluation tournament.")

        temp_best_path = "temp_best.pth"
        temp_new_path = "temp_new.pth"

        torch.save(current_state_dict, temp_best_path)
        torch.save(new_net.state_dict(), temp_new_path)

        new_wins, best_wins, draws = conduct_evaluation_tournament(
            temp_best_path, temp_new_path, all_actions, config
        )

        for path in (temp_best_path, temp_new_path):
            if os.path.exists(path):
                os.remove(path)

        logger.info(
            f"Tournament finished, Candidate: {new_wins} wins, Reference: {best_wins} wins, Draws: {draws}."
        )

        total_decisive_games = new_wins + best_wins
        win_rate = (
            (new_wins / total_decisive_games) if total_decisive_games > 0 else 0.0
        )

        logger.info(f"Candidate model win rate established at {win_rate * 100:.1f}%.")

        if win_rate >= config.win_rate_threshold:
            logger.info("Candidate model accepted, updating reference model.")
            best_net = new_net
            torch.save(best_net.state_dict(), MODEL_PATH)
        else:
            logger.info("Candidate model rejected, retaining reference model.")


if __name__ == "__main__":
    main()
