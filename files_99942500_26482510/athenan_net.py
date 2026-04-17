import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, filters):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = F.relu(x)
        out = self.conv1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += residual
        return out


class R2AthenanNet(nn.Module):
    def __init__(self, input_channels=7, filters=132, dense_neurons=155, board_size=6):
        super(R2AthenanNet, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, filters, kernel_size=3, padding=1)
        self.res_blocks = nn.Sequential(*[ResidualBlock(filters) for _ in range(8)])
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=1)
        self.flatten_size = filters * board_size * board_size
        self.fc1 = nn.Linear(self.flatten_size, dense_neurons)
        self.fc2 = nn.Linear(dense_neurons, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.res_blocks(x)
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        value = torch.tanh(self.fc2(x))

        return value
