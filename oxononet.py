import torch
import torch.nn as nn
import torch.nn.functional as F

from oxono import Game, State


def all_possible_actions():
    actions = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for totem in ['O', 'X']:
        for totem_row in range(6):
            for totem_col in range(6):
                for d_row, d_col in directions:
                    player_row, player_col = totem_row + d_row, totem_col + d_col
                    if 0 <= player_row < 6 and 0 <= player_col < 6:
                        actions.append(
                            (totem, (totem_row, totem_col), (player_row, player_col)))

    return actions


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class OptimalOxonoNet(nn.Module):
    def __init__(self, input_channels, num_actions, num_blocks=10, channels=256):
        super(OptimalOxonoNet, self).__init__()

        self.conv_input = nn.Conv2d(
            input_channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(channels)

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_blocks)])

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
