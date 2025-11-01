import torch
from torch import nn
class HybridForwardDynamics(nn.Module):
    def __init__(self, hidden_dim=256, dt=0.1):
        super().__init__()
        goal_dim = 6
        self.net = nn.Sequential(
            nn.Linear(6 + goal_dim + 2, hidden_dim),  # 6 + 6 + 2 = 14
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )

        self.dt = dt

    def forward(self, s, g, a):
        x, y, vx, vy, sin_t, cos_t = torch.split(s, 1, dim=-1)
        theta = torch.atan2(sin_t, cos_t)
        dv, dtheta, _ = torch.split(self.net(torch.cat([s, g, a], dim=-1)), 1, dim=-1)
        v = torch.sqrt(vx**2 + vy**2 + 1e-6)
        v_next = v + dv
        theta_next = theta + dtheta
        x_next = x + v_next * torch.cos(theta_next) * self.dt
        y_next = y + v_next * torch.sin(theta_next) * self.dt
        vx_next = v_next * torch.cos(theta_next)
        vy_next = v_next * torch.sin(theta_next)
        sin_next = torch.sin(theta_next)
        cos_next = torch.cos(theta_next)
        s_next = torch.cat([x_next, y_next, vx_next, vy_next, sin_next, cos_next], dim=-1)
        return s_next
