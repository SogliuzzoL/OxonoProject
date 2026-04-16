import torch
import torch.nn as nn
import torch.nn.functional as F

from oxono import State

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def all_possible_actions() -> list[tuple[str, tuple[int, int], tuple[int, int]]]:
    actions = []

    for totem in ["O", "X"]:
        for totem_row in range(6):
            for totem_col in range(6):
                for piece_row in range(6):
                    for piece_col in range(6):
                        if (totem_row, totem_col) != (piece_row, piece_col):
                            move = (
                                totem,
                                (totem_row, totem_col),
                                (piece_row, piece_col),
                            )
                            actions.append(move)

    return actions


def state_to_tensor(state: State) -> torch.Tensor:
    tensor = torch.zeros((7, 6, 6), dtype=torch.float32)
    for row in range(6):
        for col in range(6):
            square = state.board[row][col]
            if square is None:
                continue

            symbol = square[0]
            color = square[1]
            if symbol == "o":
                if color:
                    tensor[0, row, col] = 1.0
                else:
                    tensor[1, row, col] = 1.0
            elif symbol == "x":
                if color:
                    tensor[2, row, col] = 1.0
                else:
                    tensor[3, row, col] = 1.0

    totem_O = state.totem_O
    totem_X = state.totem_X

    tensor[4, totem_O[0], totem_O[1]] = 1.0
    tensor[5, totem_X[0], totem_X[1]] = 1.0

    current_player = state.current_player
    tensor[6, :, :] = 1.0 if current_player else 0.0

    return tensor


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class OxonoNet(nn.Module):
    def __init__(self, input_channels, num_actions, num_blocks=10, channels=256):
        super(OxonoNet, self).__init__()

        self.conv_input = nn.Conv2d(
            input_channels, channels, kernel_size=3, padding=1, bias=False
        )
        self.bn_input = nn.BatchNorm2d(channels)

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_blocks)]
        )

        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 6 * 6, num_actions)

        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * 6 * 6, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.bn_input(self.conv_input(x)))
        x = self.res_blocks(x)

        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        policy_logits = self.policy_fc(p)

        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value


def predict(
    state: State,
    net: OxonoNet,
    all_actions: list,
    legal_moves: list,
    device: str = DEVICE,
) -> tuple[torch.Tensor, torch.Tensor]:
    tensor = state_to_tensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        policy_logits, value = net(tensor)

        legal_mask = torch.zeros(policy_logits.size(-1), dtype=torch.bool).to(
            policy_logits.device
        )

        for move in legal_moves:
            move_tuple = (move[0], (move[1][0], move[1][1]), (move[2][0], move[2][1]))
            idx = all_actions.index(move_tuple)
            legal_mask[idx] = True

        masked_logits = policy_logits.clone()
        masked_logits[0, ~legal_mask] = float("-inf")
        masked_probs = F.softmax(masked_logits, dim=-1)

    return masked_probs.squeeze(0), value.squeeze(0)


def probs_to_dict(
    probs: torch.Tensor, all_actions: list, legal_moves: list
) -> dict[tuple[str, tuple[int, int], tuple[int, int]], float]:
    action_probs = {}
    for idx, prob in enumerate(probs):
        action = all_actions[idx]
        if action in legal_moves:
            action_probs[action] = prob.item()

    sorted_action_probs = dict(
        sorted(action_probs.items(), key=lambda item: item[1], reverse=True)
    )

    return sorted_action_probs


def values_to_dict(
    value: torch.Tensor, all_actions: list, legal_moves: list
) -> dict[str, float]:
    value_dict = {}
    for idx, action in enumerate(all_actions):
        if action in legal_moves:
            value_dict[action] = value.item()

    sorted_value_dict = dict(
        sorted(value_dict.items(), key=lambda item: item[1], reverse=True)
    )

    return sorted_value_dict
