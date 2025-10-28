import os
import numpy as np
import gymnasium as gym
from tqdm import trange
from config import num_episodes, max_episode_steps, buffer_file, data_dir
from utils import preprocess_image, save_buffer, set_seed
import highway_env
set_seed(42)
os.makedirs(data_dir, exist_ok=True)

def collect_random_data(env_name='parking-v0', num_episodes=num_episodes):
    env = gym.make(env_name, render_mode="rgb_array")
    env.unwrapped.configure({"simulation_frequency": 15})
    observations, actions, next_observations = [], [], []

    for ep in trange(num_episodes, desc='Collecting'):
        obs, _ = env.reset()
        raw = env.render()
        if raw is None:
            raw = env.render(mode="rgb_array")
        img = preprocess_image(raw).permute(1,2,0).numpy()*255.0

        for t in range(max_episode_steps):
            a = env.action_space.sample()
            obs_step = env.step(a)
            try:
                obs_next, reward, terminated, truncated, info = obs_step
                done = terminated or truncated
            except:
                _, _, done, _ = obs_step
            raw_next = env.render()
            if raw_next is None:
                raw_next = env.render(mode="rgb_array")
            img_next = preprocess_image(raw_next).permute(1,2,0).numpy()*255.0

            observations.append(img.astype(np.uint8))
            actions.append(np.array(a, dtype=np.float32))
            next_observations.append(img_next.astype(np.uint8))
            img = img_next
            if done:
                break
    env.close()
    return observations, actions, next_observations


if __name__ == '__main__':
    print('Collecting data from parking-v0...')


    obs, acts, obs2 = collect_random_data()
    print('Saving to', buffer_file)
    save_buffer(buffer_file, obs, acts, obs2)
    print('Done')
