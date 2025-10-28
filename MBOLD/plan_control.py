import numpy as np
import torch
import gymnasium as gym
import highway_env
from config import device, planning_horizon, cem_iters, cem_pop, cem_elite_frac, render, buffer_file
from utils import preprocess_image, load_buffer
from train_distance import EncoderCNN, ForwardDynamics

class CEMPlanner:
    def __init__(self, forward_model, encoder, action_low, action_high, action_dim):
        self.fwd = forward_model
        self.enc = encoder
        self.action_dim = action_dim
        self.low = action_low
        self.high = action_high
    def plan(self, z0, zg, horizon=12, iters=5, pop=512, elite_frac=0.05):
        mean = np.zeros((horizon, self.action_dim), dtype=np.float32)
        std = np.ones_like(mean) * (self.high - self.low) / 2.0
        n_elite = max(1, int(pop * elite_frac))
        for _ in range(iters):
            samples = np.random.randn(pop, horizon, self.action_dim).astype(np.float32) * std + mean
            samples = np.clip(samples, self.low, self.high)
            zs = torch.tensor(z0, dtype=torch.float32, device=device).unsqueeze(0).repeat(pop,1)
            zg_t = torch.tensor(zg, dtype=torch.float32, device=device).unsqueeze(0).repeat(pop,1)
            with torch.no_grad():
                for t in range(horizon):
                    a_t = torch.tensor(samples[:,t,:], dtype=torch.float32, device=device)
                    zs = self.fwd(zs, a_t)
                d = torch.norm(zs - zg_t, dim=-1).cpu().numpy()
            elite_idx = d.argsort()[:n_elite]
            elites = samples[elite_idx]
            mean, std = elites.mean(axis=0), elites.std(axis=0) + 1e-6
        return mean[0]

if __name__ == '__main__':
    enc = EncoderCNN().to(device)
    fwd = ForwardDynamics(latent_dim=64, action_dim=2).to(device)
    enc.load_state_dict(torch.load('models/encoder.pth', map_location=device))
    fwd.load_state_dict(torch.load('models/forward.pth', map_location=device))
    enc.eval(); fwd.eval()
    env = gym.make('parking-v0', render_mode="rgb_array"); env.unwrapped.configure({"simulation_frequency":15})
    act_low, act_high = env.action_space.low, env.action_space.high
    planner = CEMPlanner(fwd, enc, act_low, act_high, env.action_space.shape[0])
    obs, acts, obs2 = load_buffer(buffer_file)
    goal_img = obs[np.random.randint(0, len(obs))]
    for ep in range(20):
        o = env.reset(); raw = env.render()
        cur_img = preprocess_image(raw).unsqueeze(0).to(device)
        with torch.no_grad():
            z_cur = enc(cur_img).squeeze(0).cpu().numpy()
            z_goal = enc(preprocess_image(goal_img).unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
        for t in range(200):
            a = planner.plan(z_cur, z_goal, horizon=planning_horizon, iters=cem_iters, pop=cem_pop, elite_frac=cem_elite_frac)
            obs_step = env.step(a)
            try: _, _, done, _ = obs_step
            except: done = False
            raw = env.render()
            cur_img = preprocess_image(raw).unsqueeze(0).to(device)
            with torch.no_grad():
                z_cur = enc(cur_img).squeeze(0).cpu().numpy()
            if render: env.render()
            if done: break
        print(f'Episode {ep+1} done')
    env.close()
