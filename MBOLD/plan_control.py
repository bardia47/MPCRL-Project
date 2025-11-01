import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from config import device, planning_horizon, cem_iters, cem_pop, cem_elite_frac, seed, models_dir
from env_utils import make_env, obs_to_state
from models import HybridForwardDynamics
from planning import CEMPlannerState, success_state


def debug_parking_environment(env):
    # Get environment info
    env.reset()
    # Test different actions to see movement scale
    print("\n--- Testing Action Effects ---")
    test_actions = [
        [0.0, 0.0],  # No action
        [0.1, 0.1],  # Small action
        [0.5, 0.5],  # Medium action
        [1.0, 1.0],  # Max action
    ]

    for i, action in enumerate(test_actions):
        env.reset()
        obs_before, _ = env.reset()
        s_before = obs_to_state(obs_before['observation'])

        obs_after, _, _, _, _ = env.step(np.array(action))
        s_after = obs_to_state(obs_after['observation'])

        pos_change = np.linalg.norm(s_after[:2] - s_before[:2])
        vel_change = np.linalg.norm(s_after[2:4] - s_before[2:4])

        print(f"Action {action}: pos_change={pos_change:.4f}, vel_change={vel_change:.4f}")


def create_better_goal_state(env):
    """Create goal state with environment understanding"""
    obs, _ = env.reset()

    sequence = [
        [-0.2, 0.3],
        [-0.2, 0.3],
        [0.3, 0.2],
        [0.3, 0.2],
        [0.0, -0.2],
        [0.0, -0.2],
        [0.0, 0.0],
        [0.0, 0.0],
    ]
    for _ in range(20):
        a = np.array([0.2, 0.3])
        obs, _, _, _, _ = env.step(a)
        s = obs_to_state(obs['observation'])
        print(s[:2])

    for i, a in enumerate(sequence):
        obs, _, terminated, truncated, info = env.step(a)
        state = obs_to_state(obs['observation'])
        print(f"  Step {i + 1}: action={a}, pos=[{state[0]:.3f}, {state[1]:.3f}]")
        if terminated:
            break
        if truncated:
            break
    goal_state = obs_to_state(obs['observation'])
    return goal_state


def run_episode_with_parking_debug(env, planner, sg, episode_num):
    """Run episode with parking-specific debugging"""
    obs, _ = env.reset(seed=seed + episode_num)
    s = obs_to_state(obs['observation'])
    trajectory = []
    actions_taken = []
    distances = []
    planner.prev_solution = None
    success = False

    for t in range(150):  # More steps for parking
        # Record current state
        trajectory.append(s[:2].copy())
        distance = np.linalg.norm(s[:2] - sg[:2])
        distances.append(distance)

        # Check success
        if success_state(s, sg):
            print(f'🎉 Episode {episode_num + 1}: SUCCESS at step {t}')
            success = True
            break

        # Plan action
        planned_action  = planner.plan(s, sg, horizon=planning_horizon, iters=cem_iters,
                         pop=cem_pop, elite_frac=cem_elite_frac, device=device)
        planned_steering = planned_action[0]
        accel = np.random.uniform(env.action_space.low[1], env.action_space.high[1])
        a = np.array([planned_steering, accel])
        obs, _, terminated, truncated, _ = env.step(a)

        a = np.clip(a, env.action_space.low, env.action_space.high)
        actions_taken.append(a.copy())

        # Execute action
        obs, _, terminated, truncated, info = env.step(a)
        s = obs_to_state(obs['observation'])

        # Debug output
        if t % 20 == 0 or distance < 0.1:
            print(f"Step {t:3d}: pos=[{s[0]:.3f}, {s[1]:.3f}], "
                  f"dist={distance:.4f}, action=[{a[0]:.3f}, {a[1]:.3f}]")

        # Check termination
        if terminated or truncated:
            if terminated and 'success' in info and info['success']:
                print(f'🎉 ENV SUCCESS at step {t}')
                success = True
            else:
                print(f'Episode terminated: {info}')
            break

    # Enhanced plotting
    trajectory = np.array(trajectory)
    actions_taken = np.array(actions_taken)
    distances = np.array(distances)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Trajectory plot
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, alpha=0.7)
    ax1.scatter(trajectory[0, 0], trajectory[0, 1], color='green', s=150,
                marker='s', label='Start', zorder=5)
    ax1.scatter(sg[0], sg[1], color='red', s=200, marker='*', label='Goal', zorder=5)

    # Add distance circles
    circles = [0.1, 0.5, 1.0]
    for r in circles:
        circle = plt.Circle((sg[0], sg[1]), r, fill=False, alpha=0.3,
                            linestyle='--', label=f'{r}m radius')
        ax1.add_patch(circle)

    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title('Parking Trajectory')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # 2. Distance over time
    ax2.plot(distances, 'g-', linewidth=2)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Distance to Goal (m)')
    ax2.set_title('Distance Progress')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.1, color='r', linestyle='--', alpha=0.5, label='Close threshold')
    ax2.legend()

    # 3. Actions over time
    if len(actions_taken) > 0:
        steps = range(len(actions_taken))
        ax3.plot(steps, actions_taken[:, 0], 'r-', label='Steering', linewidth=2)
        ax3.plot(steps, actions_taken[:, 1], 'b-', label='Acceleration', linewidth=2)
        ax3.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax3.axhline(y=-1.0, color='gray', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Action Value')
        ax3.set_title('Actions Over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # 4. Velocity profile
    velocities = [np.linalg.norm(pos[2:4]) for pos in [obs_to_state(env.reset()[0]['observation'])]]
    ax4.set_title('Velocity Profile (placeholder)')
    ax4.text(0.5, 0.5, 'Velocity data needs\nproper implementation',
             ha='center', va='center', transform=ax4.transAxes)

    plt.tight_layout()
    plt.savefig(f'parking_debug_{episode_num + 1}.png', dpi=150, bbox_inches='tight')
    plt.show()

    return success


if __name__ == '__main__':
    print('Setting up parking environment...')
    env = make_env(render=True)
    env_goal = make_env()

    env.reset(seed=seed)
    env_goal.reset(seed=seed + 1000)

    # Debug parking environment
    debug_parking_environment(env)

    print('Loading dynamics model...')
    fwd = HybridForwardDynamics().to(device)
    model_path = os.path.join(models_dir, 'fwd_state.pth')
    fwd.load_state_dict(torch.load(model_path, map_location=device))
    fwd.eval()

    act_low, act_high = env.action_space.low, env.action_space.high
    planner = CEMPlannerState(fwd, act_low, act_high, action_dim=env.action_space.shape[0])

    # Create better goal
    sg = create_better_goal_state(env_goal)

    # Run with parking-specific debug
    print("\n=== RUNNING PARKING EPISODE ===")
    success = run_episode_with_parking_debug(env, planner, sg, 0)

    print(f'\nResult: {"SUCCESS" if success else "FAILED"}')
    env.close()
    env_goal.close()