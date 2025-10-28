import os
import numpy as np
from tqdm import trange
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from config import buffer_file, batch_size, epochs, lr, latent_dim, action_dim, device
from utils import load_buffer, preprocess_image, set_seed

set_seed(42)

class ParkingDataset(Dataset):
    def __init__(self, obs, actions, next_obs):
        self.obs = obs
        self.actions = actions
        self.next_obs = next_obs
    def __len__(self):
        return len(self.obs)
    def __getitem__(self, idx):
        o = preprocess_image(self.obs[idx])
        n = preprocess_image(self.next_obs[idx])
        a = torch.tensor(self.actions[idx], dtype=torch.float32)
        return o, a, n

class EncoderCNN(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 10 * 10, 512), nn.ReLU(),
            nn.Linear(512, latent_dim)
        )
    def forward(self, x):
        return self.net(x)

class ForwardDynamics(nn.Module):
    def __init__(self, latent_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
    def forward(self, z, a):
        x = torch.cat([z, a], dim=-1)
        return self.net(x)

if __name__ == '__main__':
    print('Loading buffer:', buffer_file)
    obs, acts, obs2 = load_buffer(buffer_file)
    dataset = ParkingDataset(obs, acts, obs2)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    enc = EncoderCNN(latent_dim).to(device)
    fwd = ForwardDynamics(latent_dim, action_dim).to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(fwd.parameters()), lr=lr)
    mse = nn.MSELoss()
    for ep in range(epochs):
        total = 0.0
        for b_o, b_a, b_no in loader:
            b_o, b_a, b_no = b_o.to(device), b_a.to(device), b_no.to(device)
            z = enc(b_o)
            z_next = enc(b_no).detach()
            z_pred = fwd(z, b_a)
            loss = mse(z_pred, z_next)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        print(f'Epoch {ep+1}/{epochs}  loss={total/len(loader):.6f}')
    os.makedirs('models', exist_ok=True)
    torch.save(enc.state_dict(), 'models/encoder.pth')
    torch.save(fwd.state_dict(), 'models/forward.pth')
    print('Saved models to models/encoder.pth and models/forward.pth')
