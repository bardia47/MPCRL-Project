from config import num_episodes, max_episode_steps, buffer_file, seed
from env_utils import make_env, obs_to_state
from utils import save_buffer, set_seed
import numpy as np
import random


def collect_random_data_with_HER(env, num_episodes, max_steps, her_ratio=0.8):
    """Collect random rollouts and augment them with Hindsight Experience Replay (HER)."""
    all_obs, all_goals, all_actions, all_next_obs, all_next_goals = [], [], [], [], []

    for ep in range(num_episodes):
        obs, info = env.reset()
        episode_transitions = []
        for _ in range(max_steps):
            steering = np.random.uniform(low=-1, high=1)
            accel = np.random.uniform(low=-1, high=1)
            a = np.array([steering, accel], dtype=np.float32)

            obs_next, _, terminated, truncated, _ = env.step(a)

            transition = {
                "obs": obs_to_state(obs["observation"]),
                "goal": obs["desired_goal"],
                "action": a,
                "next_obs": obs_to_state(obs_next["observation"]),
                "achieved_goal": obs_next["achieved_goal"],
            }
            episode_transitions.append(transition)

            obs = obs_next
            if terminated or truncated:
                break

        for i, trans in enumerate(episode_transitions):
            remaining = len(episode_transitions) - i
            sample_size = min(2, remaining)
            future_idxs = np.random.choice(range(i, len(episode_transitions)), size=sample_size, replace=False)

            for idx in future_idxs:
                her_goal = episode_transitions[idx]["achieved_goal"]

                all_obs.append(trans["obs"])
                all_goals.append(trans["goal"])
                all_actions.append(trans["action"])
                all_next_obs.append(trans["next_obs"])
                all_next_goals.append(trans["goal"])

                all_obs.append(trans["obs"])
                all_goals.append(her_goal)
                all_actions.append(trans["action"])
                all_next_obs.append(trans["next_obs"])
                all_next_goals.append(her_goal)

    return all_obs, all_goals, all_actions, all_next_obs, all_next_goals


if __name__ == "__main__":
    set_seed(seed)
    env = make_env(render=True)

    print(f"Collecting HER data from {env.unwrapped.spec.name} ...")
    obs, goals, acts, obs2, goals2 = collect_random_data_with_HER(
        env, num_episodes, max_episode_steps, her_ratio=0.8
    )

    print(f"Saving {len(obs)} samples to {buffer_file} ...")
    save_buffer(buffer_file, obs, goals, acts, obs2, goals2)
    env.close()
