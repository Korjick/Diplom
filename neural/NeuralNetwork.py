import torch
from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.Linear(10, 32),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm1d(32),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(32, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 4)
        )

    def forward(self, x):
        return self.seq(x)
