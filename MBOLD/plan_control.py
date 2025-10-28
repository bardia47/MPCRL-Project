import numpy as np
import torch
import gymnasium as gym
import highway_env
import matplotlib.pyplot as plt
import cv2
from config import device, planning_horizon, cem_iters, cem_pop, cem_elite_frac
from utils import preprocess_image, load_buffer
from train_distance import EncoderCNN, ForwardDynamics


class CEMPlanner:
    def __init__(self, forward_model, encoder, action_low, action_high, action_dim):
        self.fwd = forward_model
        self.enc = encoder
        self.action_dim = action_dim
        self.low = action_low
        self.high = action_high

    # python
    def plan(self, z0, zg, horizon=12, iters=5, pop=512, elite_frac=0.05):
        if hasattr(self, 'prev_solution') and self.prev_solution.shape[0] == horizon:
            mean = np.roll(self.prev_solution, -1, axis=0)
        else:
            mean = np.zeros((horizon, self.action_dim), dtype=np.float32)

        std = np.ones_like(mean) * 0.5
        n_elite = max(1, int(pop * elite_frac))

        for iteration in range(iters):
            noise_scale = max(0.1, 1.0 - iteration / iters)
            samples = np.random.randn(pop, horizon, self.action_dim).astype(np.float32) * std * noise_scale + mean
            samples = np.clip(samples, self.low, self.high)

            costs = self.evaluate_sequences(samples, z0, zg, horizon)
            elite_idx = costs.argsort()[:n_elite]
            elites = samples[elite_idx]
            mean = elites.mean(axis=0)
            std = elites.std(axis=0) + 1e-6

        self.prev_solution = mean
        return mean[0]
    # python
    def evaluate_sequences(self, sequences, z0, zg, horizon):
        pop = sequences.shape[0]
        zs = torch.tensor(z0, dtype=torch.float32, device=device).unsqueeze(0).repeat(pop, 1)
        zg_t = torch.tensor(zg, dtype=torch.float32, device=device).unsqueeze(0).repeat(pop, 1)

        seq_torch = torch.tensor(sequences, dtype=torch.float32, device=device)  # [pop, H, act_dim]
        costs = torch.zeros(pop, device=device)
        a_prev = torch.zeros(pop, self.action_dim, device=device)

        with torch.no_grad():
            for t in range(horizon):
                a_t = seq_torch[:, t, :]
                zs = self.fwd(zs, a_t)

                step_cost = 0.05 * torch.norm(zs - zg_t, dim=-1)
                costs += step_cost

                costs += 0.01 * torch.norm(a_t, dim=-1)
                costs += 0.05 * torch.norm(a_t - a_prev, dim=-1)
                a_prev = a_t

            terminal_cost = torch.norm(zs - zg_t, dim=-1)
            costs += terminal_cost

        return costs.cpu().numpy()

def create_proper_goal(env):
    """Create a proper parking goal that's different from initial state"""

    print("Creating proper goal state...")

    # Method 1: Create goal by moving to a specific parking spot
    obs, info = env.reset()

    # Save initial state
    initial_img = env.render()
    if initial_img is None:
        initial_img = env.render(mode="rgb_array")

    print("Initial state captured")

    # Execute specific parking sequence
    parking_sequence = [
        [-0.5,  1.0],  # steering right, accel forward
        [-0.5,  1.0],
        [ 0.8,  0.5],  # steering left, accel slow forward
        [ 0.8,  0.5],
        [ 0.0, -0.5],  # straight, reverse
        [ 0.0, -0.5],
        [ 0.0,  0.0],  # stop
    ]
    for i, action in enumerate(parking_sequence):
        env.step(action)
        print(f"Parking step {i + 1}/{len(parking_sequence)}")

    # Get goal state
    goal_img = env.render()
    if goal_img is None:
        goal_img = env.render(mode="rgb_array")

    print("Goal state created")

    # Verify goal is different from initial
    initial_processed = preprocess_image(initial_img)
    goal_processed = preprocess_image(goal_img)

    difference = torch.norm(initial_processed - goal_processed).item()
    print(f"Image difference between initial and goal: {difference:.3f}")

    if difference < 0.1:
        print("WARNING: Goal too similar to initial state!")

        # Create more different goal
        for _ in range(10):
            env.step([1.0, 1.0])  # More aggressive movement

        goal_img = env.render()
        if goal_img is None:
            goal_img = env.render(mode="rgb_array")

        goal_processed = preprocess_image(goal_img)
        difference = torch.norm(initial_processed - goal_processed).item()
        print(f"New goal difference: {difference:.3f}")

    return goal_img, initial_img


def run_episode_with_visualization(env, planner, enc, z_goal, episode_num, show_render=True):
    """Run episode with visual rendering"""

    obs, info = env.reset()

    trajectory_distances = []
    frames = []
    success = False

    print(f"\nEpisode {episode_num} started...")

    for t in range(200):
        # Get current frame for visualization
        raw = env.render()
        if raw is None:
            raw = env.render(mode="rgb_array")

        # Store frame for video
        frames.append(raw.copy())

        # Process for model
        cur_img = preprocess_image(raw).unsqueeze(0).to(device)

        with torch.no_grad():
            z_cur = enc(cur_img).squeeze(0).cpu().numpy()
            distance = np.linalg.norm(z_cur - z_goal)

        trajectory_distances.append(distance)


        # Print progress
        if t % 10 == 0:
            print(f"  Step {t:3d}: Distance = {distance:.3f}")

        # Check success
        if distance < 0.001:  # Reasonable threshold
            print(f"  SUCCESS at step {t}! Distance: {distance:.3f}")
            success = True
            break

        # Plan action
        a = planner.plan(z_cur, z_goal,
                         horizon=planning_horizon,
                         iters=cem_iters,
                         pop=cem_pop,
                         elite_frac=cem_elite_frac)

        # Execute action
        obs_step = env.step(a)
        try:
            obs_next, reward, terminated, truncated, info = obs_step
            done = terminated or truncated
        except:
            done = False

        if done:
            print(f"  Episode terminated at step {t}")
            break

    min_distance = min(trajectory_distances) if trajectory_distances else float('inf')

    if not success:
        print(f"  FAILED - Min distance: {min_distance:.3f}")

    return {
        'success': success,
        'steps': t + 1,
        'min_distance': min_distance,
        'trajectory': trajectory_distances,
        'frames': frames
    }


def save_episode_video(frames, filename, fps=10):
    """Save episode as video"""
    if not frames:
        return

    height, width, channels = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()
    print(f"Video saved: {filename}")


if __name__ == '__main__':
    print("Loading models...")
    enc = EncoderCNN().to(device)
    fwd = ForwardDynamics(latent_dim=64, action_dim=2).to(device)
    enc.load_state_dict(torch.load('models/encoder.pth', map_location=device))
    fwd.load_state_dict(torch.load('models/forward.pth', map_location=device))
    enc.eval();
    fwd.eval()

    print("Setting up environment...")
    env = gym.make('parking-v0', render_mode="rgb_array")
    env.unwrapped.configure({"simulation_frequency": 15})

    act_low, act_high = env.action_space.low, env.action_space.high
    planner = CEMPlanner(fwd, enc, act_low, act_high, env.action_space.shape[0])

    # Create PROPER goal (not from buffer!)
    goal_img, initial_img = create_proper_goal(env)

    with torch.no_grad():
        z_goal = enc(preprocess_image(goal_img).unsqueeze(0).to(device)).squeeze(0).cpu().numpy()

    results = []
    num_episodes = 3  # Fewer episodes for visualization

    for ep in range(num_episodes):
        episode_result = run_episode_with_visualization(
            env, planner, enc, z_goal, ep + 1, show_render=True
        )
        results.append(episode_result)

        # Save video of episode
        save_episode_video(
            episode_result['frames'],
            f'episode_{ep + 1}.mp4'
        )

    env.close()

    # Analyze results
    success_count = sum(1 for r in results if r['success'])
    success_rate = success_count / num_episodes

    print("\n" + "=" * 50)
    print("FINAL RESULTS:")
    print("=" * 50)
    print(f"Success Rate: {success_count}/{num_episodes} = {success_rate * 100:.1f}%")

    if success_rate < 1.0:
        avg_min_distance = np.mean([r['min_distance'] for r in results])
        print(f"Average Min Distance: {avg_min_distance:.3f}")
        print("This looks more realistic!")
    else:
        print("If still 100% success, check goal creation!")