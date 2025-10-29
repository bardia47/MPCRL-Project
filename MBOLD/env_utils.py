import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
from gymnasium.envs.registration import register
from simple_parking_no_lane_env import SimpleParkingNoLaneEnv

register(
    id="SimpleParkingNoLane-v0",
    entry_point="simple_parking_no_lane_env:SimpleParkingNoLaneEnv",
)

def make_env(render= False):
    env = gym.make('SimpleParkingNoLane-v0') if not render else gym.make('SimpleParkingNoLane-v0', render_mode='human')
    env.unwrapped.configure({
        'action_type': 'ContinuousAction',
        'simulation_frequency': 15,
        'observation': {
            'type': 'Kinematics',
            'features': ['x', 'y', 'vx', 'vy', 'heading'],
            'absolute': True,
            'normalize': False,
            'vehicles_count': 1
        },
        "add_walls" : False,
        "collision_reward" : 0,


    })
    return env

def obs_to_state(obs):
    arr = np.asarray(obs, dtype=np.float32)
    v = arr[0] if arr.ndim == 2 else arr
    if v.shape[0] != 5:
        raise ValueError(f'Expected 5 features, got shape {v.shape}')
    x, y, vx, vy, heading = v
    return np.array([x, y, vx, vy, np.sin(heading), np.cos(heading)], dtype=np.float32)
