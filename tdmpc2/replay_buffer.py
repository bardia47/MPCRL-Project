# replay_buffer.py
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim,device,size=int(1e6)):
        self.ptr, self.size, self.full = 0, size, False
        self.device = device
        self.obs = torch.zeros((size, obs_dim), dtype=torch.float32, device=device)
        self.next = torch.zeros((size, obs_dim), dtype=torch.float32, device=device)
        self.act = torch.zeros((size, act_dim), dtype=torch.float32, device=device)
        self.rew = torch.zeros((size, 1), dtype=torch.float32, device=device)
        self.done = torch.zeros((size, 1), dtype=torch.float32, device=device)

    def add(self, o, a, r, d, o2):
        if isinstance(o, np.ndarray):
            o = torch.tensor(o, dtype=torch.float32, device=self.device)
            a = torch.tensor(a, dtype=torch.float32, device=self.device)
            r = torch.tensor([r], dtype=torch.float32, device=self.device)
            d = torch.tensor([d], dtype=torch.float32, device=self.device)
            o2 = torch.tensor(o2, dtype=torch.float32, device=self.device)

        i = self.ptr
        self.obs[i] = o
        self.act[i] = a
        self.rew[i] = r
        self.done[i] = d
        self.next[i] = o2
        self.ptr = (self.ptr + 1) % self.size
        self.full = self.full or self.ptr == 0

    def sample(self, batch):
        maxidx = self.size if self.full else self.ptr
        idx = np.random.randint(0, maxidx, size=batch)
        return (
            torch.from_numpy(self.obs[idx]),
            torch.from_numpy(self.act[idx]),
            torch.from_numpy(self.rew[idx]),
            torch.from_numpy(self.done[idx]),
            torch.from_numpy(self.next[idx]),
        )

