# train_tdmpc2_cartpole.py
import os, time, json
import numpy as np
import torch
import gymnasium as gym
from collections import deque
import matplotlib.pyplot as plt

from agent import TD_MPC2_Agent
from replay_buffer import ReplayBuffer

# ---------- Paths ----------
CKPT_DIR    = "checkpoints"
REPORTS_DIR = "reports"
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
REPORT_PATH = os.path.join(REPORTS_DIR, "training_log.jsonl")

# ---------- Training knobs ----------
TOTAL_STEPS       = 10_000
RANDOM_STEPS      = 1_000
USE_MPC_UNTIL     = 5_000
UPDATES_START     = 200
UPDATES_PER_STEP  = 2
BATCH_SIZE        = 256
LOG_EVERY         = 1_000
SAVE_EVERY        = 5_000
EVAL_EVERY        = 10_000
PLOT_AT_END       = True

# ---------- Light CEM params ----------
CEM_HORIZON = 8
CEM_POP     = 64
CEM_ITERS   = 2
CEM_ELITE_FR  = 0.1
CEM_DISCOUNT  = 0.99

# ---------- CartPole continuous-action wrapper ----------
class DiscreteActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def action(self, act):
        a = float(np.asarray(act).reshape(-1)[0])
        return 1 if a >= 0.0 else 0

    def reverse_action(self, a):
        return np.array([1.0 if int(a) == 1 else -1.0], dtype=np.float32)

# ---------- Env ----------
def make_env():
    env = gym.make("CartPole-v1",render_mode=None)
    env = DiscreteActionWrapper(env)
    env.reset()
    return env

def flatten_obs(obs):
    return np.asarray(obs, dtype=np.float32)

# ---------- Actor action helper ----------
@torch.no_grad()
def policy_action(agent: TD_MPC2_Agent, obs, sample=True):
    device = agent.device
    z = agent.wm.encode(torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
    a, dist = agent.actor(z)
    if not sample:
        a = torch.tanh(dist.mean)
    a = a.squeeze(0).detach().cpu().numpy()
    # ensure shape (1,)
    return np.atleast_1d(a).astype(np.float32)

def _get_act_dim(agent: TD_MPC2_Agent, default=1):
    if hasattr(agent, "act_dim"):
        return int(agent.act_dim)
    if hasattr(agent, "actor") and hasattr(agent.actor, "act_dim"):
        return int(agent.actor.act_dim)
    if hasattr(agent, "actor") and hasattr(agent.actor, "net"):
        return int(getattr(agent.actor.net[-1], "out_features", default*2) // 2)
    return default

# ---------- Lightweight CEM ----------
@torch.no_grad()
def plan_cem_vectorized(agent: TD_MPC2_Agent, obs,
                        horizon=CEM_HORIZON, pop=CEM_POP,
                        iters=CEM_ITERS, elite_frac=CEM_ELITE_FR,
                        discount=CEM_DISCOUNT):
    device = agent.device
    act_dim = _get_act_dim(agent, default=1)
    z0 = agent.wm.encode(torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
    z0 = z0.expand(pop, -1)

    elites = max(1, int(pop * elite_frac))
    mean = torch.zeros(horizon, act_dim, device=device)
    std  = torch.ones_like(mean) * 0.5

    for _ in range(iters):
        actions = torch.normal(mean.expand(pop, -1, -1), std.expand(pop, -1, -1))
        returns = torch.zeros(pop, device=device)
        gammas = torch.ones(pop, device=device)

        z = z0.clone()
        for t in range(horizon):
            a_t = actions[:, t, :]
            z, r = agent.wm.predict(z, a_t)
            returns += gammas * r.squeeze(-1)
            gammas *= discount

        returns += gammas * agent.val(z).squeeze(-1)
        top_idx = torch.topk(returns, elites).indices
        elite_actions = actions[top_idx]
        mean, std = elite_actions.mean(0), elite_actions.std(0) + 1e-4

    a0 = mean[0].clamp(-1, 1)
    a0 = a0.detach().cpu().numpy()
    return np.atleast_1d(a0).astype(np.float32)

# ---------- Checkpointing ----------
def save_checkpoint(step, agent: TD_MPC2_Agent):
    path = os.path.join(CKPT_DIR, f"tdmpc2_step_{step}.pt")
    torch.save({
        "step": step,
        "wm": agent.wm.state_dict(),
        "val": agent.val.state_dict(),
        "actor": agent.actor.state_dict(),
        "opt": agent.opt.state_dict(),
    }, path)
    print(f"[checkpoint] saved → {path}")

def load_latest_checkpoint(agent: TD_MPC2_Agent):
    ckpts = [f for f in os.listdir(CKPT_DIR) if f.endswith(".pt")]
    if not ckpts:
        print("[checkpoint] none found, starting fresh")
        return 1
    ckpts.sort(key=lambda x: int(x.split("_")[2].split(".")[0]))
    latest = os.path.join(CKPT_DIR, ckpts[-1])
    print(f"[resume] loading {latest}")
    ckpt = torch.load(latest, map_location="cpu")
    agent.wm.load_state_dict(ckpt["wm"])
    agent.val.load_state_dict(ckpt["val"])
    agent.actor.load_state_dict(ckpt["actor"])
    agent.opt.load_state_dict(ckpt["opt"])
    start = int(ckpt["step"]) + 1
    print(f"[resume] resumed from step {start-1}")
    return start

# ---------- JSON logging ----------
def log_jsonl(obj):
    obj["ts"] = time.time()
    with open(REPORT_PATH, "a") as f:
        f.write(json.dumps(obj) + "\n")

def log_step(step, losses, avg_return):
    log_jsonl({
        "type": "step",
        "step": int(step),
        "model_loss": float(losses["model_loss"]),
        "value_loss": float(losses["v_loss"]),
        "actor_loss": float(losses["actor_loss"]),
        "total_loss": float(losses["total_loss"]),
        "avg_return": float(avg_return)
    })

def log_episode(ep_idx, step, ep_return, ep_len, avg_return):
    log_jsonl({
        "type": "episode",
        "episode": int(ep_idx),
        "step_end": int(step),
        "return": float(ep_return),
        "length": int(ep_len),
        "avg_return": float(avg_return)
    })

def log_eval(step, ret_mpc, len_mpc, ret_actor, len_actor):
    log_jsonl({
        "type": "eval",
        "step": int(step),
        "mpc_return": float(ret_mpc),
        "mpc_length": int(len_mpc),
        "actor_return": float(ret_actor),
        "actor_length": int(len_actor)
    })

# ---------- Eval ----------
def run_eval_episode(env, agent, use_mpc=True, max_steps=500):
    ob, _ = env.reset()
    ob = flatten_obs(ob)
    ret, t, done = 0.0, 0, False
    while not done and t < max_steps:
        a = plan_cem_vectorized(agent, ob) if use_mpc else policy_action(agent, ob, sample=False)
        ob, r, term, trunc, _ = env.step(a)
        ret += r
        ob = flatten_obs(ob)
        done = term or trunc
        t += 1
    return ret, t

# ---------- Plotting ----------
def make_plots():
    steps, losses, avg_steps = [], [], []
    ep_idx, ep_returns, ep_avg, ep_steps = [], [], [], []
    eval_steps, mpc_ret, actor_ret = [], [], []

    with open(REPORT_PATH, "r") as f:
        for line in f:
            rec = json.loads(line)
            t = rec.get("type", "")
            if t == "step":
                steps.append(rec["step"])
                losses.append(rec.get("total_loss"))
                avg_steps.append(rec["avg_return"])
            elif t == "episode":
                ep_idx.append(rec["episode"])
                ep_returns.append(rec["return"])
                ep_avg.append(rec["avg_return"])
                ep_steps.append(rec["step_end"])
            elif t == "eval":
                eval_steps.append(rec["step"])
                mpc_ret.append(rec["mpc_return"])
                actor_ret.append(rec["actor_return"])

    def _savefig(name):
        path = os.path.join(REPORTS_DIR, name)
        plt.savefig(path, bbox_inches="tight")
        print(f"[plot] saved → {path}")
        plt.clf()

    plt.figure(figsize=(8,4))
    ys = [v for v in losses if v is not None]
    xs = [s for s,v in zip(steps, losses) if v is not None]
    if xs:
        plt.plot(xs, ys)
    plt.title("Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    _savefig("loss_over_steps.png")

    plt.figure(figsize=(8,4))
    if ep_idx:
        plt.scatter(ep_steps, ep_returns, s=8, alpha=0.5, label="episode return")
    if ep_steps:
        win = 100
        if len(ep_returns) >= 2:
            mov = []
            for i in range(len(ep_returns)):
                lo = max(0, i - win + 1)
                mov.append(np.mean(ep_returns[lo:i+1]))
            plt.plot(ep_steps, mov, linewidth=2, label=f"{win}-episode moving avg")
    plt.title("Episode Returns")
    plt.xlabel("Env Step (end of episode)")
    plt.ylabel("Return")
    plt.legend()
    _savefig("episode_returns.png")

    plt.figure(figsize=(8,4))
    if eval_steps:
        plt.plot(eval_steps, mpc_ret, label="MPC eval return")
        plt.plot(eval_steps, actor_ret, label="Actor eval return")
        plt.legend()
    plt.title("Eval: MPC vs Actor")
    plt.xlabel("Step")
    plt.ylabel("Return")
    _savefig("eval_returns.png")

# ---------- Main ----------
def main():
    env = make_env()
    obs_dim, act_dim = 4, 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    agent = TD_MPC2_Agent(obs_dim, act_dim, device=device)
    buffer = ReplayBuffer(obs_dim, act_dim)
    start_step = load_latest_checkpoint(agent)

    o, _ = env.reset()
    o = flatten_obs(o)
    last_loss = None
    episode_return, episode_len = 0.0, 0
    recent_returns = deque(maxlen=10)
    episodes_seen = 0
    avg_ret = 0.0

    print(f"[info] Training started. random={RANDOM_STEPS}, mpc_until={USE_MPC_UNTIL}, save_every={SAVE_EVERY}")

    try:
        for step in range(start_step, TOTAL_STEPS + 1):
            if step < RANDOM_STEPS:
                a = np.random.uniform(-1, 1, act_dim).astype(np.float32)
            elif step < USE_MPC_UNTIL:
                a = plan_cem_vectorized(agent, o)
            else:
                a = policy_action(agent, o, sample=True)

            a = np.clip(a, -1.0, 1.0).astype(np.float32)

            o2, r, term, trunc, _ = env.step(a)
            done = term or trunc
            episode_return += float(r)
            episode_len += 1

            o2 = flatten_obs(o2)
            buffer.add(o, a, r, done, o2)
            o = o2 if not done else flatten_obs(env.reset()[0])

            if step > UPDATES_START:
                for _ in range(UPDATES_PER_STEP):
                    batch = buffer.sample(BATCH_SIZE)
                    res = agent.update(batch)
                last_loss = res["total_loss"]
                avg_ret = float(np.mean(recent_returns)) if recent_returns else 0.0
                log_step(step, res, avg_ret)

            if done:
                episodes_seen += 1
                recent_returns.append(episode_return)
                avg_ret = float(np.mean(recent_returns))
                print(f"[episode {episodes_seen}] return={episode_return:.2f} len={episode_len} avg(10)={avg_ret:.2f}")
                log_episode(episodes_seen, step, episode_return, episode_len, avg_ret)
                episode_return, episode_len = 0.0, 0

            if step % LOG_EVERY == 0:
                loss_str = f"{last_loss:.4f}" if last_loss is not None else "N/A"
                print(f"[step {step}] loss={loss_str}, avg_return={avg_ret:.2f}")

            if step % SAVE_EVERY == 0:
                save_checkpoint(step, agent)

            if step % EVAL_EVERY == 0:
                ret_mpc, len_mpc     = run_eval_episode(env, agent, use_mpc=True)
                ret_actor, len_actor = run_eval_episode(env, agent, use_mpc=False)
                print(f"[eval @ {step}] MPC  return={ret_mpc:.2f} len={len_mpc} | "
                      f"Actor return={ret_actor:.2f} len={len_actor}")
                log_eval(step, ret_mpc, len_mpc, ret_actor, len_actor)

        print(f"[info] Training complete. Total steps: {TOTAL_STEPS}")

    except KeyboardInterrupt:
        print("\n[info] Interrupted — saving checkpoint...")
        save_checkpoint(step, agent)

    finally:
        env.close()
        print(f"[info] Environment closed. Logs at {REPORT_PATH}")
        if PLOT_AT_END:
            print("[plot] generating plots...")
            make_plots()

if __name__ == "__main__":
    main()