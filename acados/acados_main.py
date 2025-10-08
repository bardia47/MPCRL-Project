import os
import gymnasium as gym
import highway_env
os.environ['LD_LIBRARY_PATH'] = '/home/bardia/acados/lib'
import ctypes
ctypes.CDLL("/home/bardia/acados/lib/libqpOASES_e.so")
ctypes.CDLL("/home/bardia/acados/lib/libblasfeo.so")
ctypes.CDLL("/home/bardia/acados/lib/libhpipm.so")

import numpy as np
import matplotlib.pyplot as plt
from acados_template import AcadosOcp, AcadosModel, AcadosOcpSolver
from casadi import SX, vertcat, cos, sin, tan

# Environment
env = gym.make("parking-v0", render_mode="human")
obs, info = env.reset()
done = False

nx, nu = 4, 2  # [x, y, phi, v], [acceleration, steering]
ny = nx + nu
# Define model
model = AcadosModel()
model.name = "parking_car"
x = SX.sym("x", nx)
u = SX.sym("u", nu)
dt = 0.1
x_next = x + vertcat(
    x[3] * cos(x[2]) * dt,
    x[3] * sin(x[2]) * dt,
    x[3] / 2.0 * tan(u[1]) * dt,
    u[0] * dt
)
model.x = x
model.u = u
model.f_expl_expr = x_next

# OCP setup
ocp = AcadosOcp()
ocp.model = model

x_target = np.array([0.0, 0.0, 0.0, 0.0])
Q = np.diag([10.0, 10.0, 1.0, 1.0])
R = np.diag([1.0, 1.0])
ny = nx + nu

ocp.cost.cost_type = "LINEAR_LS"
ocp.cost.cost_type_e = "LINEAR_LS"

ocp.cost.Vx = np.zeros((ny, nx))
ocp.cost.Vx[:nx, :] = np.eye(nx)
ocp.cost.Vu = np.zeros((ny, nu))
ocp.cost.Vu[nx:, :] = np.eye(nu)
ocp.cost.W = np.zeros((ny, ny))
ocp.cost.W[:nx, :nx] = Q
ocp.cost.W[nx:, nx:] = R
ocp.cost.yref = np.zeros(ny)
ocp.cost.yref[:nx] = x_target

N = 20
ocp.dims.N = N
ocp.solver_options.tf = N * dt
ocp.solver_options.qp_solver = "FULL_CONDENSING_QPOASES"
ocp.solver_options.integrator_type = "ERK"
ocp.solver_options.nlp_solver_type = "SQP"

solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")

def obs_to_state(obs_input):
    if isinstance(obs_input, dict):
        x = obs_input.get("x", 0.0)
        y = obs_input.get("y", 0.0)
        phi = obs_input.get("heading", 0.0)
        v = obs_input.get("velocity", 0.0)
        return np.array([x, y, phi, v], dtype=np.float64)
    return np.array(obs_input, dtype=np.float64)

def solve_mpc(x):
    solver.set(0, "x", x)
    for i in range(N):
        solver.set(i, "yref", np.concatenate([x_target, np.zeros(nu)]))
    status = solver.solve()
    if status != 0:
        print(f"Solver failed with status {status}")
    return solver.get(0, "u")

trajectory = []

while not done:
    x0 = obs_to_state(obs)
    u = solve_mpc(x0)
    obs, reward, terminated, truncated, info = env.step(u)
    done = terminated or truncated
    trajectory.append(x0)

trajectory = np.array(trajectory)
plt.plot(trajectory[:, 0], trajectory[:, 1], 'b.-', label="MPC path")
plt.scatter(x_target[0], x_target[1], c='r', label="Parking spot")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
