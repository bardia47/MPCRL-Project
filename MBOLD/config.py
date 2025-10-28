import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 42

num_episodes = 400
max_episode_steps = 200
data_dir = "data"
buffer_file = data_dir + "/parking_buffer.npz"

img_size = (84, 84)
latent_dim = 64
action_dim = 2

batch_size = 128
epochs = 40
lr = 1e-4

planning_horizon = 12
cem_iters = 5
cem_pop = 512
cem_elite_frac = 0.05

render = False
