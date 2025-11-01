import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from config import buffer_file, batch_size, epochs, lr, state_dim, action_dim, device, models_dir, seed
from utils import load_buffer, set_seed
from models import HybridForwardDynamics


class StateGoalDataset(Dataset):
    def __init__(self, obs, goals, actions, next_obs, next_goals):
        self.obs = obs
        self.goals = goals
        self.actions = actions
        self.next_obs = next_obs
        self.next_goals = next_goals

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.obs[idx], dtype=torch.float32),
            torch.tensor(self.goals[idx], dtype=torch.float32),
            torch.tensor(self.actions[idx], dtype=torch.float32),
            torch.tensor(self.next_obs[idx], dtype=torch.float32),
            torch.tensor(self.next_goals[idx], dtype=torch.float32),
        )


if __name__ == '__main__':
    set_seed(seed)
    print('Loading buffer:', buffer_file)
    obs, goals, acts, obs2, goals2 = load_buffer(buffer_file)
    loader = DataLoader(StateGoalDataset(obs, goals, acts, obs2, goals2),
                        batch_size=batch_size, shuffle=True, drop_last=True)

    fwd = HybridForwardDynamics().to(device)
    opt = torch.optim.Adam(fwd.parameters(), lr=lr, weight_decay=1e-5)
    mse = nn.MSELoss()

    for ep in range(epochs):
        total = 0.0
        fwd.train()
        for s, g, a, s_next, g_next in loader:
            s, g, a, s_next, g_next = s.to(device), g.to(device), a.to(device), s_next.to(device), g_next.to(device)
            s_pred = fwd(s, g, a)
            loss = mse(s_pred, s_next)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(fwd.parameters(), 1.0)
            opt.step()
            total += loss.item()

    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, 'fwd_state.pth')
    torch.save(fwd.state_dict(), path)
    print('Saved model to', path)
