import math
import numpy as np
import gymnasium as gym
import highway_env

class FixedParkingGoal(gym.Wrapper):
    def __init__(self, env, goal_pos=(0.0, 0.0), goal_heading=0.0):
        super().__init__(env)
        self.goal_pos = np.array(goal_pos, dtype=float)
        self.goal_heading = float(goal_heading)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        goal_vec = np.array(
            [self.goal_pos[0], self.goal_pos[1], math.cos(self.goal_heading), math.sin(self.goal_heading)],
            dtype=float
        )
        self.env.unwrapped.goal = goal_vec

        if hasattr(self.env.unwrapped, "observation_type"):
            obs = self.env.unwrapped.observation_type.observe()

        info = dict(info)
        info["desired_goal"] = goal_vec
        return obs, info