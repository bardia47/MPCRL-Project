import numpy as np
import torch
import os

def save_buffer(path, observations, actions, next_observations):
    np.savez_compressed(path,
                        observations=np.array(observations, dtype=np.float32),
                        actions=np.array(actions, dtype=np.float32),
                        next_observations=np.array(next_observations, dtype=np.float32))

def load_buffer(path):
    d = np.load(path)
    return d['observations'], d['actions'], d['next_observations']

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
