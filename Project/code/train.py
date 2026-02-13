import gymnasium as gym
from gymnasium.wrappers import NormalizeReward, FrameStack, AtariPreprocessing
import torch
import numpy as np
import argparse
from dqn_agent import DQNAgent
from collections import deque
import json
import os
from datetime import datetime
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='DQN Training')
    
    parser.add_argument('--env', type=str, default='CartPole-v1',
                        help='Gymnasium environment name')
    parser.add_argument('--episodes', type=int, default=500,
                        help='Number of episodes to train')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--buffer-size', type=int, default=100000,
                        help='Size of replay buffer')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--target-update', type=int, default=1,
                        help='Number of steps between target network updates')
    parser.add_argument('--render', action='store_true',
                        help='Render the environment')
    
    # Training parameters
    parser.add_argument('--epsilon-start', type=float, default=1.0,
                        help='Starting epsilon for exploration')
    parser.add_argument('--epsilon-end', type=float, default=0.01,
                        help='Final epsilon for exploration')
    parser.add_argument('--epsilon-decay', type=float, default=0.99,
                        help='Decay rate for epsilon (applied per episode)')
    
    # DQN variants
    parser.add_argument('--double', action='store_true',
                        help='Enable Double DQN')
    parser.add_argument('--dueling', action='store_true',
                        help='Enable Dueling DQN')
    parser.add_argument('--priority', action='store_true',
                        help='Enable Prioritized Experience Replay')

    # Save and load
    parser.add_argument('--save-dir', type=str, default='train_data/checkpoints',
                        help='Directory to save models')
    parser.add_argument('--load-path', type=str,
                        help='Path to load a pretrained model')
    
    return parser.parse_args()

def create_env(env_name, seed=0, render_mode=None, terminal_on_life_loss=True):
    env = gym.make(env_name, render_mode=render_mode)

    is_atari = "Pong" in env_name or "Breakout" in env_name or "NoFrameskip" in env_name

    if is_atari:
        print(f"Detected Atari environment: {env_name}. Applying AtariPreprocessing wrappers.")
        env = AtariPreprocessing(
            env, noop_max=30, frame_skip=4,
            screen_size=84, grayscale_obs=True,
            terminal_on_life_loss=terminal_on_life_loss,
            scale_obs=False  # CNNDQN handles /255.0 internally
        )
        env = FrameStack(env, 4)
        state_dim = (4, 84, 84)
    else:
        state_dim = env.observation_space.shape[0]

    env.reset(seed=seed)
    action_dim = env.action_space.n
    return env, state_dim, action_dim

def save_training_config(args, save_dir):
    config = vars(args)
    config_path = os.path.join(save_dir, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

def aggregate_episode_q_values(q_values_list):
    """Aggregate Q-values statistics for an episode"""
    if not q_values_list:
        return None
    
    q_values_array = np.array(q_values_list)
    return {
        'mean': float(np.mean(q_values_array)),
        'max': float(np.max(q_values_array)),
        'min': float(np.min(q_values_array)),
        'std': float(np.std(q_values_array))
    }

def update_live_plot(fig, ax, rewards, avg_rewards):
    ax.clear()
    ax.plot(rewards, label='Episode Reward', alpha=0.3, color='#F28B82')
    ax.plot(avg_rewards, label='Average Reward (100 eps)', color='#D93025', linewidth=2)
    ax.set_title(f'Live Training Progress (Episode {len(rewards)})')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.01)

def main():
    args = parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
        
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Live plotting setup
    import matplotlib
    matplotlib.use('TkAgg')
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.show(block=False)
    
    render_mode = "human" if args.render else None
    
    # Create environment and agent
    env, state_dim, action_dim = create_env(args.env, render_mode=render_mode)
    # env = NormalizeReward(env, gamma=0.99, epsilon=1e-8) # Optional: Disable for now to see raw rewards
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")

    # Build variant label for logging
    variant_parts = []
    if args.double:
        variant_parts.append("Double")
    if args.dueling:
        variant_parts.append("Dueling")
    if args.priority:
        variant_parts.append("PER")
    variant_label = " + ".join(variant_parts) if variant_parts else "Plain"
    print(f"Running {variant_label} DQN")

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        learning_rate=args.lr,
        gamma=args.gamma,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        target_update=args.target_update,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        use_double=args.double,
        use_dueling=args.dueling,
        use_priority=args.priority
    )
    
    if args.load_path and os.path.exists(args.load_path):
        agent.load(args.load_path)
        print(f"Loaded model from {args.load_path}")
    
    # Create save directory
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    
    # Save training configuration
    save_training_config(args, save_dir)
    
    # Training metrics
    training_logs = {
        'rewards': [],
        'avg_rewards': [],
        'losses': [],
        'epsilons': [],
        'episode_lengths': [],
        'total_frames': 0,
        'frames_per_episode': [],
        'fps': [],
        'config': vars(args)
    }
    
    q_values_log = {
        'episodes': []  # Will store per-episode Q-value statistics
    }

    window_size = 100
    reward_window = deque(maxlen=window_size)
    best_avg_reward = float('-inf')
    total_frames = 0
    
    # Use standard range, tqdm will be used manually inside loop for step details if needed, 
    # but strictly requested dynamic plots/terminal output. 
    # Let's use tqdm for the outer loop but keep it clean.
    
    pbar = tqdm(range(args.episodes), desc="Training")
    
    for episode in pbar:
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = []
        done = False
        truncated = False
        steps = 0
        episode_start_time = time.time()
        
        # Lists to store Q-values for this episode
        episode_target_q = []
        episode_current_q = []
        episode_next_q = []
        episode_td_errors = []
        
        while not (done or truncated):
            steps += 1
            total_frames += 1
            
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Atari wrappers return LazyFrames, need conversion to tensor/array in agent or here.
            # Agent expects numpy array or similar. FrameStack returns LazyFrames.
            # np.array(state) handles it.
            
            agent.memory.push(state, action, reward, next_state, done or truncated)
            
            # Only update after collecting enough samples
            if len(agent.memory) > args.batch_size:
                loss, target_q_values, current_q_values, next_q_values, dones, td_errors = agent.update()
                if loss is not None:
                    episode_loss.append(loss)
                    # Store batch statistics instead of raw values
                    episode_target_q.append(target_q_values.mean().item())
                    episode_current_q.append(current_q_values.mean().item())
                    episode_next_q.append(next_q_values.mean().item())
                    episode_td_errors.append(td_errors.mean().item())

            state = next_state
            episode_reward += reward

        # Decay epsilon once per episode
        agent.decay_epsilon()

        # Calculate FPS
        episode_time = time.time() - episode_start_time
        fps = steps / episode_time if episode_time > 0 else 0
        
        # Aggregate episode Q-values
        episode_q_values = {
            'target_q': aggregate_episode_q_values(episode_target_q),
            'current_q': aggregate_episode_q_values(episode_current_q),
            'next_q': aggregate_episode_q_values(episode_next_q),
            'td_errors': aggregate_episode_q_values(episode_td_errors)
        }
        
        # Store episode Q-values
        q_values_log['episodes'].append({
            'episode': episode,
            'q_values': episode_q_values
        })
        
        # Update metrics
        avg_episode_loss = np.mean(episode_loss) if episode_loss else 0
        training_logs['rewards'].append(episode_reward)
        training_logs['losses'].append(avg_episode_loss)
        training_logs['epsilons'].append(agent.epsilon)
        training_logs['episode_lengths'].append(steps)
        training_logs['total_frames'] = total_frames
        training_logs['frames_per_episode'].append(steps)
        training_logs['fps'].append(fps)
        
        reward_window.append(episode_reward)
        avg_reward = np.mean(reward_window)
        training_logs['avg_rewards'].append(avg_reward)
        
        # Save best model
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            best_model_path = os.path.join(save_dir, 'best_model.pth')
            agent.save(best_model_path)
            
        # Update Live Plot every N episodes to save performance
        if episode % 5 == 0:
            update_live_plot(fig, ax, training_logs['rewards'], training_logs['avg_rewards'])
        
        # Update Progress Bar with neat metrics
        pbar.set_postfix({
            'Reward': f'{episode_reward:.2f}',
            'Avg': f'{avg_reward:.2f}',
            'Eps': f'{agent.epsilon:.2f}',
            'FPS': f'{fps:.0f}'
        })
        
        # Save intermediate logs
        if episode % 10 == 0:
            log_path = os.path.join(save_dir, 'training_logs.json')
            with open(log_path, 'w') as f:
                json.dump(training_logs, f, indent=4)
    
    # Save final model and logs
    final_model_path = os.path.join(save_dir, 'final_model.pth')
    agent.save(final_model_path)
    
    final_log_path = os.path.join(save_dir, 'training_logs.json')
    with open(final_log_path, 'w') as f:
        json.dump(training_logs, f, indent=4)

    final_q_values_path = os.path.join(save_dir, 'q_values_log.json')
    with open(final_q_values_path, 'w') as f:
        json.dump(q_values_log, f, indent=4)
    
    print("\nTraining completed!")
    print(f"Best average reward: {best_avg_reward:.2f}")
    print(f"Total frames seen: {total_frames}")
    print(f"Models and logs saved in {save_dir}")
    print("Close the plot window to exit.")
    plt.ioff()
    plt.show() # Keep window open at end
    
    env.close()

if __name__ == "__main__":
    main()