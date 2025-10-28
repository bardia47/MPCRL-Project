import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from config import buffer_file, batch_size, epochs, lr, state_dim, action_dim, device, models_dir, seed
from utils import load_buffer, set_seed
from models import ForwardDynamics

class StateDataset(Dataset):
    def __init__(self, obs, actions, next_obs):
        self.obs = obs
        self.actions = actions
        self.next_obs = next_obs
    def __len__(self):
        return len(self.obs)
    def __getitem__(self, idx):
        o = torch.tensor(self.obs[idx], dtype=torch.float32)
        a = torch.tensor(self.actions[idx], dtype=torch.float32)
        n = torch.tensor(self.next_obs[idx], dtype=torch.float32)
        return o, a, n

if __name__ == '__main__':
    set_seed(seed)
    print('Loading buffer:', buffer_file)
    obs, acts, obs2 = load_buffer(buffer_file)
    loader = DataLoader(StateDataset(obs, acts, obs2), batch_size=batch_size, shuffle=True, drop_last=True)

    fwd = ForwardDynamics(state_dim=state_dim, action_dim=action_dim).to(device)
    opt = torch.optim.Adam(fwd.parameters(), lr=lr, weight_decay=1e-5)
    mse = nn.MSELoss()

    for ep in range(epochs):
        total = 0.0
        fwd.train()
        for s, a, s_next in loader:
            s, a, s_next = s.to(device), a.to(device), s_next.to(device)
            s_pred = fwd(s, a)
            loss = mse(s_pred, s_next)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(fwd.parameters(), 1.0)
            opt.step()
            total += loss.item()
        print(f'Epoch {ep+1}/{epochs}  loss={total/len(loader):.6f}')

    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, 'fwd_state.pth')
    torch.save(fwd.state_dict(), path)
    print('Saved model to', path)
