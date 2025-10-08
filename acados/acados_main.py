# acados_parking.py
import gymnasium as gym
import highway_env
import numpy as np
import matplotlib.pyplot as plt

from acados_template import AcadosOcp, AcadosOcpSolver
from casadi import SX

env = gym.make("parking-v0", render_mode="human")
obs = env.reset()
done = False


nx = 4  # [x, y, phi, v]
nu = 2  # [acceleration, steering]

ocp = AcadosOcp()
ocp.model = type('model', (), {})()  # dummy model

ocp.model.x = SX.sym("x", nx)
ocp.model.u = SX.sym("u", nu)
x = ocp.model.x
u = ocp.model.u

dt = 0.1
x_next = x + SX.vertcat(
    x[3] * np.cos(x[2]) * dt,  # x
    x[3] * np.sin(x[2]) * dt,  # y
    x[3] / 2.0 * np.tan(u[1]) * dt,  # phi (simple bicycle)
    u[0] * dt  # v
)
ocp.model.f_expl_expr = x_next


x_target = np.array([0.0, 0.0, 0.0, 0.0])  # parking spot
Q = np.diag([10.0, 10.0, 1.0, 1.0])
R = np.diag([1.0, 1.0])
ocp.cost_Vx = np.eye(nx)
ocp.cost_Vu = np.eye(nu)
ocp.cost_W = np.block([
    [Q, np.zeros((nx, nu))],
    [np.zeros((nu, nx)), R]
])
ocp.cost_yref = np.zeros(nx + nu)

N = 20
ocp.dims.N = N
ocp.solver_options.tf = N * dt
ocp.solver_options.qp_solver = "FULL_CONDENSING_QPOASES"
ocp.solver_options.integrator_type = "ERK"
ocp.solver_options.nlp_solver_type = "SQP"

solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")

def obs_to_state(obs_dict):
    """Convert highway-env observation to acados state vector"""
    if isinstance(obs_dict, dict):
        # Use vehicle kinematics: x, y, heading, speed
        x = obs_dict.get("x", 0.0)
        y = obs_dict.get("y", 0.0)
        phi = obs_dict.get("heading", 0.0)
        v = obs_dict.get("velocity", 0.0)
        return np.array([x, y, phi, v])
    else:
        return np.zeros(nx)

def solve_mpc(x0):
    """Set initial state and solve MPC"""
    solver.set(0, "x0", x0)
    for i in range(N):
        solver.set(i, "yref", np.concatenate([x_target, np.zeros(nu)]))
    status = solver.solve()
    u_opt = solver.get(0, "u")
    return u_opt

obs = env.reset()
done = False
trajectory = []

while not done:
    x0 = obs_to_state(obs)
    u = solve_mpc(x0)
    obs, reward, terminated, truncated, info = env.step(u)
    done = terminated or truncated
    trajectory.append(x0)

trajectory = np.array(trajectory)
plt.plot(trajectory[:,0], trajectory[:,1], 'b.-', label="MPC path")
plt.scatter(x_target[0], x_target[1], c='r', label="Parking spot")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
