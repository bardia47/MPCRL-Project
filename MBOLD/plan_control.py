import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from config import device, planning_horizon, cem_iters, cem_pop, cem_elite_frac, seed, models_dir
from env_utils import make_env, obs_to_state
from models import ForwardDynamics
from planning import CEMPlannerState, success_state


def create_goal_state(env):
    obs, _ = env.reset()
    sequence = [
        [-0.5, 1.0],
        [-0.5, 1.0],
        [0.8, 0.5],
        [0.8, 0.5],
        [0.0, -0.5],
        [0.0, -0.5],
        [0.0, 0.0],
    ]

    valid_goal = False
    for a in sequence:
        obs, _, terminated, truncated, info = env.step(a)
        if terminated:
            if 'success' in info and info['success']:
                valid_goal = True
            break
        if truncated:
            break

    goal_state = obs_to_state(obs)
    print(f"Goal generation {'SUCCESS' if valid_goal else 'FAILED'}")
    return goal_state


def debug_environment(env):
    """Debug environment details"""
    print("\n=== ENVIRONMENT DEBUG ===")
    print(f"Action space: {env.action_space}")
    print(f"Action bounds: low={env.action_space.low}, high={env.action_space.high}")
    print(f"Observation space: {env.observation_space}")

    # Test a big action
    obs, _ = env.reset()
    s_before = obs_to_state(obs)
    print(f"State before big action: {np.round(s_before, 3)}")

    big_action = np.array([1.0, 1.0])  # Max action
    obs, _, _, _, _ = env.step(big_action)
    s_after = obs_to_state(obs)
    print(f"State after big action [1,1]: {np.round(s_after, 3)}")
    print(f"State change: {np.round(s_after - s_before, 3)}")


def test_forward_dynamics(fwd, device):
    """Test if forward dynamics work"""
    print("\n=== FORWARD DYNAMICS TEST ===")

    # Test state
    test_state = np.array([0., 0., 0., 0., -0.989, 0.15])
    test_action = np.array([1.0, 1.0])  # Big action

    with torch.no_grad():
        s_tensor = torch.FloatTensor(test_state).unsqueeze(0).to(device)
        a_tensor = torch.FloatTensor(test_action).unsqueeze(0).to(device)
        next_state = fwd(s_tensor, a_tensor).cpu().numpy()[0]

    print(f"Input state: {np.round(test_state, 3)}")
    print(f"Input action: {test_action}")
    print(f"Predicted next state: {np.round(next_state, 3)}")
    print(f"State change by model: {np.round(next_state - test_state, 3)}")


def run_episode_with_bold_actions(env, planner, sg, episode_num):
    """Run episode with more aggressive actions"""
    obs, _ = env.reset(seed=seed + episode_num)
    s = obs_to_state(obs)
    print(f'\n=== Episode {episode_num + 1} BOLD VERSION ===')
    print(f'Start state: {np.round(s, 3)}')
    print(f'Goal state:  {np.round(sg, 3)}')
    print(f'Distance to goal: {np.linalg.norm(s - sg):.3f}')

    trajectory = [s[:2].copy()]
    actions_taken = []
    planner.prev_solution = None
    success = False

    for t in range(100):  # More steps
        if success_state(s, sg):
            print(f'Episode {episode_num + 1}: SUCCESS at step {t}')
            success = True
            break

        # Plan action
        a = planner.plan(s, sg,
                         horizon=planning_horizon, iters=cem_iters,
                         pop=cem_pop, elite_frac=cem_elite_frac,
                         device=device)

        # 🔥 Make actions more aggressive if far from goal
        distance_to_goal = np.linalg.norm(s[:2] - sg[:2])
        if distance_to_goal > 0.1:  # If far from goal
            # Scale up actions
            a = a * 2.0  # Double the actions!
            a = np.clip(a, env.action_space.low, env.action_space.high)

        print(f"Step {t}: distance={distance_to_goal:.3f}, action={np.round(a, 3)}")

        obs, _, terminated, truncated, info = env.step(a)
        s = obs_to_state(obs)

        trajectory.append(s[:2].copy())
        actions_taken.append(a.copy())

        if terminated or truncated:
            reason = "success" if (terminated and 'success' in info and info['success']) else "failure"
            if reason == "success":
                success = True
            break
    else:
        print(f'Episode {episode_num + 1}: Completed {t + 1} steps')

    # Plot results
    trajectory = np.array(trajectory)
    actions_taken = np.array(actions_taken)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Trajectory plot
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'b-o', linewidth=2, markersize=3)
    ax1.scatter(trajectory[0, 0], trajectory[0, 1], color='green', s=150, marker='s', label='Start')
    ax1.scatter(sg[0], sg[1], color='red', s=150, marker='*', label='Goal')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_title(f'Episode {episode_num + 1} - Bold Actions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # Actions plot
    if len(actions_taken) > 0:
        steps = range(len(actions_taken))
        ax2.plot(steps, actions_taken[:, 0], 'r-o', label='Steering', markersize=3)
        ax2.plot(steps, actions_taken[:, 1], 'b-o', label='Throttle', markersize=3)
        ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax2.axhline(y=-1.0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Action Value')
        ax2.set_title('Bold Actions Taken')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'bold_episode_{episode_num + 1}.png', dpi=150, bbox_inches='tight')
    plt.show()

    return success


if __name__ == '__main__':
    print('Setting up environments...')
    env = make_env(render=True)
    env_goal = make_env()

    env.reset(seed=seed)
    env_goal.reset(seed=seed + 1000)

    # Debug environment first
    debug_environment(env)

    print('Loading dynamics model...')
    fwd = ForwardDynamics().to(device)
    model_path = os.path.join(models_dir, 'fwd_state.pth')
    fwd.load_state_dict(torch.load(model_path, map_location=device))
    fwd.eval()

    # Test forward dynamics
    test_forward_dynamics(fwd, device)

    act_low, act_high = env.action_space.low, env.action_space.high
    planner = CEMPlannerState(fwd, act_low, act_high, action_dim=env.action_space.shape[0])

    sg = create_goal_state(env_goal)
    print('Goal state:', np.round(sg, 3))

    # Run just one episode with full debugging
    print("\n=== RUNNING ONE EPISODE WITH FULL DEBUG ===")
    success = run_episode_with_bold_actions(env, planner, sg, 0)

    print(f'\nResult: {"SUCCESS" if success else "FAILED"}')

    env.close()
    env_goal.close()