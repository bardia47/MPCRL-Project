# nmpc_parking_casadi.py
import math
import numpy as np
import casadi as ca
import gymnasium as gym
import highway_env

from MPC.env import FixedParkingGoal

# Model params
L = 2.5
N = 15  # horizon

# Bounds (match env if needed)


Qx, Qy, Qpsi, Qv = 10.0, 10.0, 6.0, 1.0
Rsteer, Racc = 1.0, 0.2
QxT, QyT, QpsiT, QvT = 400.0, 400.0, 50.0, 5.0
Rdu_steer, Rdu_acc = 8.0, 0.8


def build_solver(u_min, u_max, dt):
    x = ca.SX.sym('x');
    y = ca.SX.sym('y');
    psi = ca.SX.sym('psi');
    v = ca.SX.sym('v')
    steer = ca.SX.sym('steer');
    acc = ca.SX.sym('acc')

    f = ca.vertcat(
        v * ca.cos(psi),
        v * ca.sin(psi),
        (v / L) * steer,
        acc
    )

    X = ca.SX.sym('X', 4, N + 1)
    U = ca.SX.sym('U', 2, N)
    X0 = ca.SX.sym('X0', 4)
    goal = ca.SX.sym('goal', 3)  # [gx, gy, gpsi]

    J = 0
    g = []

    # dynamics constraints
    g += [X[:, 0] - X0]
    for k in range(N):
        xk = X[:, k]
        uk = U[:, k]
        x_next = xk + dt * ca.Function('f', [ca.vertcat(x, y, psi, v), ca.vertcat(steer, acc)], [f])(xk, uk)
        g += [X[:, k + 1] - x_next]

        pos_err = ca.vertcat(xk[0] - goal[0], xk[1] - goal[1])
        psi_err = ca.atan2(ca.sin(xk[2] - goal[2]), ca.cos(xk[2] - goal[2]))
        J += Qx * pos_err[0] ** 2 + Qy * pos_err[1] ** 2 + Qpsi * psi_err ** 2 + Qv * (xk[3]) ** 2 \
             + Rsteer * uk[0] ** 2 + Racc * uk[1] ** 2
        if k > 0:
            J += Rdu_steer * (U[0, k] - U[0, k - 1]) ** 2 + Rdu_acc * (U[1, k] - U[1, k - 1]) ** 2

    # terminal cost
    xT = X[:, N]
    pos_errT = ca.vertcat(xT[0] - goal[0], xT[1] - goal[1])
    psi_errT = ca.atan2(ca.sin(xT[2] - goal[2]), ca.cos(xT[2] - goal[2]))
    J += QxT * pos_errT[0] ** 2 + QyT * pos_errT[1] ** 2 + QpsiT * psi_errT ** 2 + QvT * (xT[3]) ** 2

    # decision vars and bounds
    vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
    lbg = np.zeros(4 * (N + 1))  # dynamics eq
    ubg = np.zeros(4 * (N + 1))
    lbx = -ca.inf * ca.DM.ones(4 * (N + 1) + 2 * N, 1)
    ubx = ca.inf * ca.DM.ones(4 * (N + 1) + 2 * N, 1)

    # input bounds
    for k in range(N):
        idx = 4 * (N + 1) + 2 * k
        lbx[idx:idx + 2] = ca.DM(u_min)
        ubx[idx:idx + 2] = ca.DM(u_max)

    nlp = {'x': vars, 'f': J, 'g': ca.vertcat(*g), 'p': ca.vertcat(X0, goal)}
    solver = ca.nlpsol('solver', 'ipopt', nlp, {
        'ipopt.print_level': 0,
        'print_time': 0,
        'ipopt.max_iter': 500,
        'ipopt.tol': 1e-3,
        'ipopt.acceptable_tol': 1e-2,
        'ipopt.linear_solver': 'mumps'
    })
    return solver


def pack_vars(X_guess, U_guess):
    return np.concatenate([X_guess.reshape(-1), U_guess.reshape(-1)])


def run_episode(N=15):
    env = gym.make('parking-v0')
    env.unwrapped.configure({
        "observation": {
            "type": "Kinematics",
            "features": ["x", "y", "vx", "vy", "heading"],
            "scales": [100, 100, 5, 5, 1],
            "normalize": True
        },
        "action": {"type": "ContinuousAction"},
        "simulation_frequency": 10,
        "policy_frequency": 10
    })
    env = FixedParkingGoal(env, goal_pos=(10.0, -2.0), goal_heading=0.0)
    dt = 1.0 / env.unwrapped.config["policy_frequency"]  # dt = 0.1
    obs, info = env.reset()
    act_low = np.array(env.action_space.low, dtype=float)
    act_high = np.array(env.action_space.high, dtype=float)
    # align env timing with NMPC

    u_min = act_low.copy()
    u_max = act_high.copy()

    def obs_to_state(env, o):
        if isinstance(o, dict) and 'observation' in o:
            o = o['observation']
        o = np.asarray(o, dtype=float).ravel()

        feats = getattr(env.unwrapped.observation_type, "features", None)
        if feats is None:
            raise ValueError("env observation_type.features not found")

        idx = {name: i for i, name in enumerate(feats)}

        # required keys must exist
        x = float(o[idx["x"]])
        y = float(o[idx["y"]])

        if "vx" in idx and "vy" in idx:
            vx = float(o[idx["vx"]])
            vy = float(o[idx["vy"]])
            v = math.hypot(vx, vy)
        elif "speed" in idx:
            v = float(o[idx["speed"]])
        else:
            v = 0.0

        if "heading" in idx:
            psi = float(o[idx["heading"]])
        elif "cos_h" in idx and "sin_h" in idx:
            psi = math.atan2(float(o[idx["sin_h"]]), float(o[idx["cos_h"]]))
        else:
            raise ValueError(f"Cannot infer heading from features: {feats}")

        return np.array([x, y, psi, v], dtype=float)
    x0 = obs_to_state(env,obs)
    if 'desired_goal' in info:
        dg = info['desired_goal']
        gx, gy = float(dg[0]), float(dg[1])
        gpsi = 0.0 if len(dg) < 4 else math.atan2(float(dg[3]), float(dg[2]))
    else:
        gx, gy, gpsi = 0.0, 0.0, 0.0

    solver = build_solver(u_min, u_max, dt)
    X_guess = np.tile(x0.reshape(4, 1), (1, N + 1))
    U_guess = np.zeros((2, N))
    u_prev = None

    total = 0.0
    for step in range(300):
        p = np.concatenate([x0, np.array([gx, gy, gpsi])])
        sol = solver(x0=pack_vars(X_guess, U_guess), p=p, lbg=0, ubg=0)
        w = np.array(sol['x']).squeeze()
        X_opt = w[:4 * (N + 1)].reshape(4, N + 1)
        U_opt = w[4 * (N + 1):].reshape(2, N)

        u0 = U_opt[:, 0]
        u0 = np.clip(u0, u_min, u_max)

        obs, r, terminated, truncated, info = env.step(u0)
        total += r
        x0 = obs_to_state(env,obs)

        # warm-start
        X_guess = np.hstack([X_opt[:, 1:], X_opt[:, -1:]])
        U_guess = np.hstack([U_opt[:, 1:], U_opt[:, -1:] * 0.5])

        if terminated or truncated:
            break

    print(f"Total reward: {total:.3f}")
    env.close()


if __name__ == "__main__":
    run_episode()
