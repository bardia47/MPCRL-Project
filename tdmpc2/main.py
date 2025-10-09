# main.py
import os
import time
import json
import threading
import tempfile
import numpy as np
import torch
import gymnasium as gym
import highway_env
from collections import deque
from agent import TD_MPC2_Agent
from replay_buffer import ReplayBuffer

# --- WSL headless setup ---
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

# --- Training parameters ---
TOTAL_STEPS       = 200_000
RANDOM_STEPS      = 5_000
UPDATES_START     = 1_000
UPDATES_PER_STEP  = 2
BATCH_SIZE        = 128
LOG_EVERY         = 1_000
SAVE_EVERY        = 5_000

# --- Folders ---
CKPT_DIR   = os.path.expanduser("~/checkpoints")
REPORTS_DIR= os.path.expanduser("~/reports")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
REPORT_PATH = os.path.join(REPORTS_DIR, "training_log.json")

# --- Environment ---
def make_env(render=False):
    render_mode = "human" if render else "rgb_array"
    env = gym.make("parking-v0", render_mode=render_mode)
    env.unwrapped.configure({
        "observation": {
            "type": "KinematicsGoal",
            "features": ["x","y","vx","vy","cos_h","sin_h"],
            "scales": [100,100,5,5,1,1],
            "normalize": True,
        },
        "action": {"type": "ContinuousAction"},
        "simulation_frequency": 15,
        "policy_frequency": 5,
        "offscreen_rendering": True,
    })
    env.reset()
    return env


def flatten_obs(obs):
    if isinstance(obs, dict):
        return np.concatenate([obs["observation"], obs["desired_goal"]]).astype(np.float32)
    return np.asarray(obs, dtype=np.float32)


# --- Fully async safe checkpoint ---
def safe_checkpoint(path, step, agent):
    """Snapshot and save asynchronously to avoid any CUDA sync or freeze."""
    final_dir = os.path.dirname(path)
    os.makedirs(final_dir, exist_ok=True)

    def _snapshot_and_save():
        try:
            # Lower thread priority to avoid CPU contention
            try:
                os.nice(10)
            except Exception:
                pass

            # --- Snapshot all weights on background thread ---
            weights = {}
            for name, param in agent.wm.state_dict().items():
                weights[f"wm.{name}"] = param.detach().cpu().clone().numpy()
            for name, param in agent.val.state_dict().items():
                weights[f"val.{name}"] = param.detach().cpu().clone().numpy()
            for name, param in agent.actor.state_dict().items():
                weights[f"actor.{name}"] = param.detach().cpu().clone().numpy()

            # --- Write atomically ---
            fd, tmp_path = tempfile.mkstemp(dir=final_dir, prefix=".ckpt_", suffix=".tmp")
            os.close(fd)
            with open(tmp_path, "wb") as f:
                np.savez(f, step=step, **weights)
            os.replace(tmp_path, path)
            print(f"[background] saved checkpoint → {os.path.basename(path)}")

        except Exception as e:
            print(f"[checkpoint error] {e}")
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    threading.Thread(target=_snapshot_and_save, daemon=True).start()


# --- JSON log writer ---
def save_report(step, loss, avg_return):
    entry = {
        "step": step,
        "loss": float(loss) if loss is not None else None,
        "avg_return": float(avg_return),
        "timestamp": time.time(),
    }
    with open(REPORT_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


# --- Main training ---
def main():
    env = make_env(render=False)
    obs_dim, act_dim = 12, 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = TD_MPC2_Agent(obs_dim, act_dim, device=device)
    buffer = ReplayBuffer(obs_dim, act_dim)

    o, _ = env.reset()
    o = flatten_obs(o)
    last_loss = None
    episode_return, episode_len = 0.0, 0
    recent_returns = deque(maxlen=10)

    print(f"[info] Training started. Saving every {SAVE_EVERY} steps...")

    try:
        for step in range(1, TOTAL_STEPS + 1):
            # --- Act ---
            if step < RANDOM_STEPS:
                a = np.random.uniform(-1, 1, act_dim)
            else:
                a = np.asarray(agent.plan(o), dtype=np.float32).reshape(-1)
            a = np.clip(a, env.action_space.low, env.action_space.high)

            # --- Environment step ---
            o2, r, term, trunc, _ = env.step(a)
            done = term or trunc
            episode_return += r
            episode_len += 1

            o2 = flatten_obs(o2)
            buffer.add(o, a, r, done, o2)
            o = o2 if not done else flatten_obs(env.reset()[0])

            # --- End of episode ---
            if done:
                recent_returns.append(episode_return)
                avg_ret = float(np.mean(recent_returns)) if recent_returns else 0.0
                print(f"[episode] return={episode_return:.2f} len={episode_len} avg(10)={avg_ret:.2f}")
                episode_return, episode_len = 0.0, 0

            # --- Learning ---
            if step > UPDATES_START:
                for _ in range(UPDATES_PER_STEP):
                    batch = buffer.sample(BATCH_SIZE)
                    last_loss = agent.update(batch)

            # --- Logging ---
            if step % LOG_EVERY == 0:
                avg_ret = float(np.mean(recent_returns)) if recent_returns else 0.0
                loss_str = f"{last_loss:.4f}" if last_loss is not None else "N/A"
                print(f"[step {step}] loss={loss_str}, avg_return={avg_ret:.2f}")
                save_report(step, last_loss, avg_ret)

            # --- Checkpoint saving ---
            if step % SAVE_EVERY == 0:
                ckpt_path = os.path.join(CKPT_DIR, f"tdmpc2_step_{step}.npz")
                safe_checkpoint(ckpt_path, step, agent)
                print(f"[checkpoint] queued save → {ckpt_path}")

        print(f"[info] Training complete. Total steps: {TOTAL_STEPS}")

    except KeyboardInterrupt:
        print("\n[info] Interrupted — saving final checkpoint...")
        ckpt_path = os.path.join(CKPT_DIR, f"tdmpc2_step_{step}_INTERRUPTED.npz")
        safe_checkpoint(ckpt_path, step, agent)

    finally:
        env.close()
        print(f"[info] Environment closed. Logs at {REPORT_PATH}")


if __name__ == "__main__":
    main()
