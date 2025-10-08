# train_parking_td3.py
import gymnasium as gym
import highway_env            # registers envs with gym
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.her import HER
import os

# --- Helper to build environment ---
def make_env(render_mode=None):
    def _init():
        env = gym.make('parking-v0', render_mode=render_mode)  # goal-conditioned
        # You can customize config like env.configure({"observation": {"type": "Kinematics"} , ...})
        env = Monitor(env)   # record episode reward / length for SB3
        return env
    return _init

# --- Create vectorized envs and normalization ---
train_env = DummyVecEnv([make_env() for _ in range(4)])   # parallelize 4 envs for sample throughput
# Normalize obs/rewards (helps with learning stability)
train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.)

# Evaluation env (no VecNormalize wrapper state leak)
eval_env = DummyVecEnv([make_env()])
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.)
# If using VecNormalize, remember to save/load its statistics with the model for resuming

# --- Action noise for TD3 (continuous actions) ---
n_actions = train_env.action_space.shape[-1]
# start with small Gaussian noise for exploration of continuous actions
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# --- Callbacks: evaluation + early stop (optional) ---
eval_callback = EvalCallback(eval_env,
                             best_model_save_path='./logs/best_model',
                             log_path='./logs/results',
                             eval_freq=5_000,  # every 5000 steps
                             deterministic=True,
                             render=False)

# ---------- Option A: plain TD3 ----------
model = TD3('MlpPolicy',
            train_env,
            action_noise=action_noise,
            verbose=1,
            learning_rate=1e-4,
            buffer_size=200_000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            policy_delay=2,
            train_freq=(1, "episode"))  # or (1, "step") depending on preference

# Train
model.learn(total_timesteps=400_000, callback=eval_callback)

# Save
model.save("td3_parking_plain")
# Save VecNormalize stats
train_env.save('./logs/vecnormalize_train.pkl')

# ---------- Option B: TD3 + HER (goal-conditioned replay) ----------
# Useful when env is sparse and follows GoalEnv interface (parking-v0 does).
# HER wraps the off-policy algorithm and expects the environment to provide
# 'observation', 'desired_goal', 'achieved_goal' in the observation dict.

model_her = HER(
    "MlpPolicy",
    train_env,        # env must be a GoalEnv or provide the appropriate dict obs
    TD3,              # base algorithm
    n_sampled_goal=4, # how many HER samples per transition
    goal_selection_strategy='future',  # 'future', 'final' or 'episode'
    verbose=1,
    policy_kwargs=dict(net_arch=[256,256]),
    buffer_size=200_000,
    learning_rate=1e-4,
    batch_size=256,
    gamma=0.98,
)

model_her.learn(total_timesteps=600_000, callback=eval_callback)
model_her.save("td3_parking_her")

# When evaluating, remember to apply the same VecNormalize statistics
# and call env.reset() with deterministic seeds where needed.
