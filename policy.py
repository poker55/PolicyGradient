import torch
import torch.nn as nn
import math
from constants import MIN_STD

class PolicyNetwork(nn.Module):
    def __init__(self, input_size):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # Output mean and log_std
        )
    
    def forward(self, x):
        output = self.network(x)
        return torch.cat([
            output[:, 0:1],  # mean
            torch.clamp(output[:, 1:2], min=math.log(MIN_STD))  # log_std
        ], dim=1)