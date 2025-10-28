import os
import torch

seed = 42
device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_dir = 'data'
models_dir = 'models'
os.makedirs(data_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

buffer_file = os.path.join(data_dir, 'parking_kinematics.npz')

# data collection
num_episodes = 200
max_episode_steps = 100

# training
state_dim = 6
action_dim = 2
batch_size = 256
epochs = 50
lr = 1e-3

# planning
planning_horizon = 24
cem_iters = 8
cem_pop = 1024
cem_elite_frac = 0.1
