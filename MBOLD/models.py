import torch
from torch import nn

class ForwardDynamics(nn.Module):
    def __init__(self, state_dim=6, action_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, state_dim),
        )
    def forward(self, s, a):
        return s + self.net(torch.cat([s, a], dim=-1))
