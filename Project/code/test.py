import gymnasium as gym
try:
    import ale_py
    gym.register_envs(ale_py)
except ImportError:
    pass  # ale_py not installed, Atari envs won't be available
from gymnasium.wrappers import AtariPreprocessing
try:
    from gymnasium.wrappers import FrameStack
except ImportError:
    from gymnasium.wrappers import FrameStackObservation as FrameStack
import torch
import numpy as np
import argparse
from dqn_agent import DQNAgent
import os
import time

def parse_args():
    parser = argparse.ArgumentParser(description='DQN Evaluation')

    parser.add_argument('--env', type=str, default='CartPole-v1',
                        help='Gymnasium environment name')
    parser.add_argument('--load-path', type=str, required=True,
                        help='Path to load the pretrained model')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to evaluate')
    parser.add_argument('--render-mode', type=str, default='human',
                        help='Render mode (human, rgb_array, etc.)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frames per second for rendering')

    # DQN variants (must match training configuration)
    parser.add_argument('--double', action='store_true',
                        help='Enable Double DQN')
    parser.add_argument('--dueling', action='store_true',
                        help='Enable Dueling DQN')
    parser.add_argument('--priority', action='store_true',
                        help='Enable Prioritized Experience Replay')

    return parser.parse_args()

def create_env(env_name, render_mode='human', full_action_space=True):
    is_atari = env_name.startswith("ALE/") or "NoFrameskip" in env_name

    # Build gym.make kwargs
    make_kwargs = {"render_mode": render_mode}
    if is_atari:
        make_kwargs["full_action_space"] = full_action_space
        # Disable built-in frameskip for ALE/v5 envs (AtariPreprocessing handles it)
        if env_name.startswith("ALE/"):
            make_kwargs["frameskip"] = 1

    env = gym.make(env_name, **make_kwargs)

    if is_atari:
        print(f"Detected Atari environment: {env_name}. Applying AtariPreprocessing wrappers.")
        env = AtariPreprocessing(
            env, noop_max=30, frame_skip=4,
            screen_size=84, grayscale_obs=True,
            terminal_on_life_loss=False,  # Fair evaluation — don't end on life loss
            scale_obs=False
        )
        env = FrameStack(env, 4)
        state_dim = (4, 84, 84)
    else:
        state_dim = env.observation_space.shape[0]

    action_dim = env.action_space.n
    return env, state_dim, action_dim

def main():
    args = parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Auto-detect variant flags from checkpoint if not specified on CLI
    use_double = args.double
    use_dueling = args.dueling
    use_priority = args.priority

    if os.path.exists(args.load_path):
        checkpoint = torch.load(args.load_path, map_location=device, weights_only=False)
        if 'use_double' in checkpoint:
            use_double = checkpoint['use_double']
            use_dueling = checkpoint['use_dueling']
            use_priority = checkpoint['use_priority']
            print(f"Auto-detected variant flags from checkpoint: "
                  f"double={use_double}, dueling={use_dueling}, priority={use_priority}")
    else:
        print(f"Error: Model path {args.load_path} does not exist.")
        return

    # Create environment
    env, state_dim, action_dim = create_env(args.env, render_mode=args.render_mode)
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")

    # Initialize agent with correct variant flags
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        buffer_size=1,
        batch_size=1,
        use_double=use_double,
        use_dueling=use_dueling,
        use_priority=use_priority
    )

    # Load model weights
    agent.load(args.load_path)
    print(f"Loaded model from {args.load_path}")
    agent.epsilon = 0.01  # Greedy evaluation

    total_rewards = []

    for episode in range(args.episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        steps = 0

        while not (done or truncated):
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            state = next_state
            episode_reward += reward
            steps += 1

            if args.render_mode == 'human':
                time.sleep(1.0 / args.fps)

        total_rewards.append(episode_reward)
        print(f"Episode {episode+1}: Reward = {episode_reward}, Steps = {steps}")

    avg_reward = np.mean(total_rewards)
    print(f"\nAverage Reward over {args.episodes} episodes: {avg_reward:.2f}")

    env.close()

if __name__ == "__main__":
    main()
