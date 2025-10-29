from config import num_episodes, max_episode_steps, buffer_file, seed, FIXED_ACCEL
from env_utils import make_env, obs_to_state
from utils import save_buffer, set_seed
import numpy as np

def collect_random_data(env, num_episodes, max_steps):
    observations, actions, next_observations = [], [], []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        s = obs_to_state(obs)
        for _ in range(max_steps):
            steering = np.random.uniform(low=-1, high=1)
            accel = FIXED_ACCEL
            a = np.array([steering, accel], dtype=np.float32)
            obs_next, _, terminated, truncated, _ = env.step(a)
            obs_next, _, terminated, truncated, _ = env.step(a)
            s_next = obs_to_state(obs_next)
            observations.append(s)
            actions.append(np.array(a, dtype=np.float32))
            next_observations.append(s_next)
            s = s_next
            if terminated or truncated:
                break
    return observations, actions, next_observations

if __name__ == '__main__':
    set_seed(seed)
    env = make_env()

    print(f'Collecting data from {env.unwrapped.spec.name}...')
    env.reset(seed=seed)
    obs, acts, obs2 = collect_random_data(env, num_episodes, max_episode_steps)
    print('Saving to', buffer_file)
    save_buffer(buffer_file, obs, acts, obs2)
    env.close()
    print('Done')
