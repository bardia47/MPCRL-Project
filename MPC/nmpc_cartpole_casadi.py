# nmpc_cartpole_casadi.py
import math
import numpy as np
import casadi as ca
import gymnasium as gym

# Horizon and weights
N = 25
Qx, Qxd, Qth, Qthd = 1.0, 0.1, 20.0, 0.1
R = 0.01
QxT, QxdT, QthT, QthdT = 20.0, 1.0, 200.0, 1.0
Rdu = 0.1

# Physical params (match Gym CartPole)
M = 1.0     # cart mass
m = 0.1     # pole mass
l = 0.5     # half-length of pole
g = 9.8

def build_solver(u_min, u_max, dt):
    # State and input symbols
    x = ca.SX.sym('x')
    xd = ca.SX.sym('xd')
    th = ca.SX.sym('th')
    thd = ca.SX.sym('thd')
    u = ca.SX.sym('u')

    total = M + m
    poleml = m * l

    # Gym dynamics discretized by Euler
    temp = (u + poleml * thd**2 * ca.sin(th)) / total
    thacc = (g * ca.sin(th) - ca.cos(th) * temp) / (l * (4.0/3.0 - m * ca.cos(th)**2 / total))
    xacc = temp - poleml * thacc * ca.cos(th) / total

    x_next = x + dt * xd
    xd_next = xd + dt * xacc
    th_next = th + dt * thd
    thd_next = thd + dt * thacc

    f = ca.vertcat(x_next, xd_next, th_next, thd_next)

    # Decision variables
    X = ca.SX.sym('X', 4, N+1)
    U = ca.SX.sym('U', 1, N)
    X0 = ca.SX.sym('X0', 4)         # [x, xd, th, thd]
    goal = ca.SX.sym('goal', 4)     # desired state (usually zeros)

    # Objective and constraints
    J = 0
    cons = []
    cons += [X[:, 0] - X0]

    f_fun = ca.Function('f_fun', [ca.vertcat(x, xd, th, thd), u], [f])

    for k in range(N):
        xk = X[:, k]
        uk = U[:, k]

        # Angle error with wrapping
        th_err = ca.atan2(ca.sin(xk[2] - goal[2]), ca.cos(xk[2] - goal[2]))

        # Stage cost
        J += Qx*(xk[0]-goal[0])**2 + Qxd*(xk[1]-goal[1])**2 + Qth*(th_err)**2 + Qthd*(xk[3]-goal[3])**2 \
             + R*(uk**2)
        if k > 0:
            J += Rdu * (U[0, k] - U[0, k-1])**2

        # Dynamics equality
        x_next_k = f_fun(xk, uk)
        cons += [X[:, k+1] - x_next_k]

    # Terminal cost
    xT = X[:, N]
    th_err_T = ca.atan2(ca.sin(xT[2] - goal[2]), ca.cos(xT[2] - goal[2]))
    J += QxT*(xT[0]-goal[0])**2 + QxdT*(xT[1]-goal[1])**2 + QthT*(th_err_T)**2 + QthdT*(xT[3]-goal[3])**2

    # Flatten decision vars
    vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))

    # Constraints bounds
    lbg = np.zeros(4*(N+1))
    ubg = np.zeros(4*(N+1))

    # Variable bounds
    lbx = -ca.inf*ca.DM.ones(4*(N+1) + 1*N, 1)
    ubx =  ca.inf*ca.DM.ones(4*(N+1) + 1*N, 1)

    # Input bounds
    for k in range(N):
        idx = 4*(N+1) + k  # single input per stage
        lbx[idx] = u_min
        ubx[idx] = u_max

    nlp = {'x': vars, 'f': J, 'g': ca.vertcat(*cons), 'p': ca.vertcat(X0, goal)}
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

def run_episode():
    env = gym.make('CartPole-v1')
    obs, info = env.reset(seed=0)

    # Time step and force magnitude from env
    dt = getattr(env.unwrapped, "tau", 0.02)
    force_mag = getattr(env.unwrapped, "force_mag", 10.0)
    u_min, u_max = -force_mag, force_mag

    # Initial state and goal
    x0 = np.array(obs, dtype=float)  # [x, x_dot, theta, theta_dot]
    goal = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)

    # NMPC setup
    solver = build_solver(u_min, u_max, dt)
    X_guess = np.tile(x0.reshape(4,1), (1, N+1))
    U_guess = np.zeros((1, N))

    total = 0.0
    max_steps = 500

    for step in range(max_steps):
        p = np.concatenate([x0, goal])
        sol = solver(x0=pack_vars(X_guess, U_guess), p=p, lbg=0, ubg=0)
        w = np.array(sol['x']).squeeze()

        X_opt = w[:4*(N+1)].reshape(4, N+1)
        U_opt = w[4*(N+1):].reshape(1, N)
        u0 = float(np.clip(U_opt[0, 0], u_min, u_max))

        # Map continuous u to discrete action
        action = 1 if u0 >= 0.0 else 0

        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        x0 = np.array(obs, dtype=float)

        # Warm-start
        X_guess = np.hstack([X_opt[:, 1:], X_opt[:, -1:]])
        U_guess = np.hstack([U_opt[:, 1:], U_opt[:, -1:]*0.5])

        if terminated or truncated:
            break

    print(f"Total reward: {total:.1f}")
    env.close()

if __name__ == "__main__":
    run_episode()