import highway_env
import gymnasium as gym

from tdmpc2.gymEnvWrapper import GymToDmEnv

gym_env = gym.make("parking-v0")
env = GymToDmEnv(gym_env)