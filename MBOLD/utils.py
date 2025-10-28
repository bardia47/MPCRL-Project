import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from config import img_size

resize_and_norm = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(img_size),
    transforms.ToTensor(),
])

def preprocess_image(img):
    if isinstance(img, np.ndarray):
        img = img[:, :, :3]
        return resize_and_norm(img)
    elif isinstance(img, Image.Image):
        return resize_and_norm(np.array(img))
    else:
        raise ValueError("Unsupported image type")

def save_buffer(path, observations, actions, next_observations):
    np.savez_compressed(path, observations=np.array(observations),
                        actions=np.array(actions),
                        next_observations=np.array(next_observations))

def load_buffer(path):
    d = np.load(path)
    return d['observations'], d['actions'], d['next_observations']

def set_seed(seed):
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
