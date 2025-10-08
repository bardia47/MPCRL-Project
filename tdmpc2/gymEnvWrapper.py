import dm_env
import numpy as np
from gymnasium import Env

class GymToDmEnv(dm_env.Environment):
    """
    تبدیل gymnasium env به dm_env برای استفاده در TDMPC2
    """
    def __init__(self, gym_env: Env):
        self._env = gym_env
        self._reset_next_step = True

    def reset(self):
        obs, info = self._env.reset()
        self._reset_next_step = False
        return dm_env.restart(self._convert_obs(obs))

    def step(self, action):
        if self._reset_next_step:
            return self.reset()
        obs, reward, terminated, truncated, info = self._env.step(action)
        done = terminated or truncated
        if done:
            self._reset_next_step = True
            return dm_env.termination(reward=reward, observation=self._convert_obs(obs))
        else:
            return dm_env.transition(reward=reward, observation=self._convert_obs(obs))

    def observation_spec(self):
        shape = self._env.observation_space.shape
        dtype = self._env.observation_space.dtype
        return dm_env.specs.Array(shape, dtype, name='observation')

    def action_spec(self):
        low = self._env.action_space.low
        high = self._env.action_space.high
        shape = self._env.action_space.shape
        return dm_env.specs.BoundedArray(shape, np.float32, low, high, name='action')

    def _convert_obs(self, obs):
        # در صورت dict بودن observation آن را تبدیل کن
        if isinstance(obs, dict):
            return np.concatenate([v.flatten() for v in obs.values()])
        return obs
