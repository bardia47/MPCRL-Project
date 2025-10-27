import torch, torch.nn as nn, torch.optim as optim
import numpy as np
import torch.nn.functional as F

# ---------- small MLP factory ----------
def mlp(in_dim, out_dim, hid=128, layers=2):
    net = []
    for _ in range(layers):
        net += [nn.Linear(in_dim, hid), nn.LayerNorm(hid), nn.SiLU()]
        in_dim = hid
    net += [nn.Linear(in_dim, out_dim)]
    return nn.Sequential(*net)

# ---------- World Model (latent dynamics) ----------
class WorldModel(nn.Module):
    def __init__(self, obs_dim, act_dim, latent_dim=64, hid=256):
        super().__init__()
        self.encoder = mlp(obs_dim, latent_dim, hid)
        self.transition = mlp(latent_dim + act_dim, latent_dim, hid)
        self.reward = mlp(latent_dim + act_dim, 1, hid)

    def encode(self, obs):
        return self.encoder(obs)

    def predict(self, z, a):
        za = torch.cat([z, a], -1)
        z_next = self.transition(za)
        r_pred = self.reward(za)
        return z_next, r_pred


# ---------- Value Function ----------
class ValueNet(nn.Module):
    def __init__(self, latent_dim, hid=256):
        super().__init__()
        self.v = mlp(latent_dim, 1, hid)

    def forward(self, z):
        return self.v(z)


# ---------- Actor (learned policy for warm-start) ----------
class Actor(nn.Module):
    def __init__(self, latent_dim, act_dim, hid=256):
        super().__init__()
        self.net = mlp(latent_dim, 2 * act_dim, hid)

    def forward(self, z):
        mu_logstd = self.net(z)
        mu, logstd = mu_logstd.chunk(2, -1)
        std = torch.exp(torch.clamp(logstd, -5, 1))
        dist = torch.distributions.Normal(mu, std)
        a = dist.rsample()
        return torch.tanh(a), dist


# ---------- CEM Planning (Model Predictive Control) ----------
@torch.no_grad()
def cem_plan(model, value_fn, z0, act_dim, horizon=10, pop=512, elite_frac=0.1, iters=6):
    elites = int(pop * elite_frac)
    mean = torch.zeros(horizon, act_dim, device=z0.device)
    std = torch.ones_like(mean) * 0.5
    for _ in range(iters):
        actions = torch.normal(mean.expand(pop, -1, -1), std.expand(pop, -1, -1))  # (pop,H,act)
        returns = []
        for i in range(pop):
            z, G, discount = z0.clone(), 0, 1
            for a in actions[i]:
                z, r = model.predict(z, a.unsqueeze(0))
                G += discount * r.squeeze()
                discount *= 0.99
            G += discount * value_fn(z).squeeze()
            returns.append(G)
        returns = torch.stack(returns)
        top = torch.topk(returns, elites).indices
        elite_actions = actions[top]
        mean, std = elite_actions.mean(0), elite_actions.std(0) + 1e-4
    return mean[0].clamp(-1, 1)


# ---------- TD-MPC2 Agent ----------
class TD_MPC2_Agent:
    def __init__(self, obs_dim, act_dim, device="cpu"):
        self.device = torch.device(device)
        self.wm = WorldModel(obs_dim, act_dim, latent_dim=32, hid=128).to(self.device)
        self.val = ValueNet(32, hid=128).to(self.device)
        self.actor = Actor(32, act_dim, hid=128).to(self.device)
        self.opt = optim.Adam(
            list(self.wm.parameters()) + list(self.val.parameters()) + list(self.actor.parameters()),
            lr=1e-4
        )
        self.gamma = 0.90
    def plan(self, obs):
        z0 = self.wm.encode(torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0))
        return cem_plan(self.wm, self.val, z0, act_dim=self.actor.net[-1].out_features // 2).cpu().numpy()

    def update(self, batch):
        o, a, r, d, o2 = [x.to(self.device).float() for x in batch]
        z, z2 = self.wm.encode(o), self.wm.encode(o2)
        z_pred, r_pred = self.wm.predict(z, a)

        r_clipped = torch.clamp(r, -1.0, 1.0)

        z_mse = F.mse_loss(z_pred, z2.detach())
        r_mse = F.mse_loss(r_pred, r)
        model_loss = z_mse + 0.5 * r_mse
        with torch.no_grad():
            v_target = r + self.gamma * (1 - d) * self.val(z2)
            v_target = torch.clamp(v_target, -5.0, 5.0)
        v_loss = F.smooth_l1_loss(self.val(z), v_target)

        loss = model_loss + v_loss

        self.opt.zero_grad(set_to_none=True)
        loss.backward()

        self.opt.step()

        metrics = {
            "loss": float(loss.item()),
            "model_loss": float(model_loss.item()),
            "z_mse": float(z_mse.item()),
            "r_mse": float(r_mse.item()),
            "v_loss": float(v_loss.item()),
            "val_mean": float(self.val(z).mean().item()),
        }
        return metrics
    # --- Checkpoint helpers ---
    def state_dict(self):
        """Collect all model and optimizer weights for checkpointing."""
        return {
            "wm": self.wm.state_dict(),
            "val": self.val.state_dict(),
            "actor": self.actor.state_dict(),
            "opt": self.opt.state_dict(),
            "gamma": self.gamma,
        }

    def load_state_dict(self, state_dict):
        """Reload all model and optimizer weights from checkpoint."""
        self.wm.load_state_dict(state_dict["wm"])
        self.val.load_state_dict(state_dict["val"])
        self.actor.load_state_dict(state_dict["actor"])
        if "opt" in state_dict:
            self.opt.load_state_dict(state_dict["opt"])
        self.gamma = state_dict.get("gamma", self.gamma)
