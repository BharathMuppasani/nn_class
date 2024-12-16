import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.ticker import MaxNLocator
from typing import Dict, List
import argparse
import glob
import os
import pandas as pd

def setup_style():
    """Set up clean and website-like plotting style with extended colors."""
    # Extended harmonious color palette
    colors = [
        '#F28B82',  # Soft Red
        '#9FA8DA',  # Muted Blue
        '#CE93D8',  # Soft Purple
        '#F6BF72',  # Muted Orange
        '#80CBC4',  # Soft Teal
        '#B0BEC5'   # Warm Gray
    ]
    
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.serif': ['Source Sans Pro'],
        'axes.titlesize': 16,
        'axes.labelsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.labelcolor': '#2D2D2D',
        'text.color': '#2D2D2D',
        'axes.edgecolor': '#DADADA',
        'grid.color': '#f2dad8',
        'grid.alpha': 0.4,
        'figure.facecolor': '#ffffff',
        'axes.facecolor': '#fffdfc',
        'lines.linewidth': 2.5
    })
    return colors

def smooth_data(data: list, window: int = 10) -> np.ndarray:
    """Apply moving average smoothing to data."""
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')

def calculate_frames_seen(episode_lengths: List[int]) -> List[int]:
    """Calculate cumulative frames seen by the agent."""
    return np.cumsum(episode_lengths).tolist()

def plot_q_values(q_values_log: Dict, save_dir: str, colors, start_episode: int = 1):
    """Create plots for Q-value metrics."""
    episodes = []
    target_q_data = {'mean': [], 'std': [], 'max': [], 'min': []}
    current_q_data = {'mean': [], 'std': [], 'max': [], 'min': []}
    next_q_data = {'mean': [], 'std': [], 'max': [], 'min': []}
    td_error_data = {'mean': [], 'std': [], 'max': [], 'min': []}
    
    # Extract data from the log, skipping null values and early episodes
    for episode_data in q_values_log['episodes']:
        episode_num = episode_data['episode']
        if episode_num < start_episode:
            continue
            
        q_values = episode_data['q_values']
        
        if all(q_values[key] is not None for key in ['target_q', 'current_q', 'next_q', 'td_errors']):
            episodes.append(episode_num)
            
            for metric, data in [('target_q', target_q_data), 
                               ('current_q', current_q_data),
                               ('next_q', next_q_data),
                               ('td_errors', td_error_data)]:
                stats = q_values[metric]
                data['mean'].append(stats['mean'])
                data['std'].append(stats['std'])
                data['max'].append(stats['max'])
                data['min'].append(stats['min'])

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Q-Values Analysis', fontsize=16, y=0.95, color='#2D2D2D')

    def plot_metric(ax, data, title, color):
        means = np.array(data['mean'])
        stds = np.array(data['std'])
        maxs = np.array(data['max'])
        mins = np.array(data['min'])
        
        ax.plot(episodes, means, label='Mean', color=color, linewidth=2)
        ax.fill_between(episodes, means - stds, means + stds, 
                       alpha=0.2, color=color, label='±1 std')
        ax.fill_between(episodes, mins, maxs, 
                       alpha=0.1, color=color, label='Min-Max')
        
        ax.set_title(title)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Value')
        ax.legend(frameon=True, facecolor='white', framealpha=0.9)
        ax.grid(True, alpha=0.2)

    plot_metric(ax1, target_q_data, 'Target Q-Values', colors[0])
    plot_metric(ax2, current_q_data, 'Current Q-Values', colors[1])
    plot_metric(ax3, next_q_data, 'Next Q-Values', colors[2])
    plot_metric(ax4, td_error_data, 'TD Errors', colors[3])

    plt.tight_layout()
    plt.savefig(f'{save_dir}/q_values_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_single_run(metrics: Dict, q_values_log: Dict, save_dir: str, colors, start_episode: int = 1):
    """Create plots for a single training run."""
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax3 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2 = fig.add_subplot(gs[0, 1])
    ax4 = fig.add_subplot(gs[1, 1])
    
    fig.suptitle('Training Metrics Overview', fontsize=16, y=0.95, color='#2D2D2D')
    
    # Slice data starting from start_episode
    episodes = range(start_episode, len(metrics['rewards']) + 1)
    rewards = metrics['rewards'][start_episode-1:]
    losses = metrics['losses'][start_episode-1:]
    episode_lengths = metrics['episode_lengths'][start_episode-1:]
    frames_seen = calculate_frames_seen(episode_lengths)
    episodes_array = np.array(episodes)
    
    # Plot Average Reward
    reward_color = colors[0]
    rewards_array = np.array(rewards)
    smoothed_rewards = smooth_data(rewards, window=20)
    rolling_std = np.array([np.std(rewards_array[max(0, i-20):i+1]) 
                           for i in range(len(rewards_array))])
    
    ax1.plot(episodes, rewards, alpha=0.2, label='Raw', linewidth=1, color=reward_color)
    ax1.fill_between(episodes_array[19:],
                    smoothed_rewards - rolling_std[19:],
                    smoothed_rewards + rolling_std[19:],
                    alpha=0.1, color=reward_color, label='±1 std')
    ax1.plot(episodes_array[19:], smoothed_rewards,
             label='Moving Average', linewidth=2, color=reward_color, zorder=5)
    
    # Plot Sample Efficiency
    eff_color = colors[2]
    smoothed_eff_rewards = smooth_data(rewards, window=50)
    rolling_std_eff = np.array([np.std(rewards_array[max(0, i-50):i+1]) 
                               for i in range(len(rewards_array))])
    
    ax2.plot(frames_seen, rewards, alpha=0.2, linewidth=1, color=eff_color, label='Raw')
    ax2.fill_between(frames_seen[49:],
                    smoothed_eff_rewards - rolling_std_eff[49:],
                    smoothed_eff_rewards + rolling_std_eff[49:],
                    alpha=0.1, color=eff_color, label='±1 std')
    ax2.plot(frames_seen[49:], smoothed_eff_rewards,
             linewidth=2, color=eff_color, zorder=5, label='Moving Average')
    
    # Plot Training Loss
    loss_color = colors[1]
    losses_array = np.array(losses)
    smoothed_losses = smooth_data(losses, window=10)
    rolling_std_loss = np.array([np.std(losses_array[max(0, i-10):i+1]) 
                                for i in range(len(losses_array))])
    
    ax3.plot(episodes, losses, alpha=0.2, label='Raw', linewidth=1, color=loss_color)
    ax3.fill_between(episodes_array[9:],
                    smoothed_losses - rolling_std_loss[9:],
                    smoothed_losses + rolling_std_loss[9:],
                    alpha=0.1, color=loss_color, label='±1 std')
    ax3.plot(episodes_array[9:], smoothed_losses,
             label='Moving Average', linewidth=2, color=loss_color, zorder=5)
    
    # Plot Episode Lengths
    length_color = colors[3]
    lengths_array = np.array(episode_lengths)
    smoothed_lengths = smooth_data(episode_lengths, window=20)
    rolling_std_length = np.array([np.std(lengths_array[max(0, i-20):i+1]) 
                                  for i in range(len(lengths_array))])
    
    ax4.plot(episodes, episode_lengths, alpha=0.2, label='Raw', linewidth=1, color=length_color)
    ax4.fill_between(episodes_array[19:],
                    smoothed_lengths - rolling_std_length[19:],
                    smoothed_lengths + rolling_std_length[19:],
                    alpha=0.1, color=length_color, label='±1 std')
    ax4.plot(episodes_array[19:], smoothed_lengths,
             label='Moving Average', linewidth=2, color=length_color, zorder=5)
    
    # Set titles and labels
    ax1.set_title('Average Episode Reward')
    ax1.set_ylabel('Reward')
    ax1.legend(frameon=True, facecolor='white', framealpha=0.9)
    
    ax2.set_title('Sample Efficiency')
    ax2.set_xlabel('Total Frames')
    ax2.set_ylabel('Average Reward')
    ax2.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Loss')
    ax3.legend(frameon=True, facecolor='white', framealpha=0.9)
    
    ax4.set_title('Episode Lengths')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Steps')
    ax4.legend(frameon=True, facecolor='white', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_metrics_overview.png', dpi=300, bbox_inches='tight')
    plt.close()

    if q_values_log:
        plot_q_values(q_values_log, save_dir, colors, start_episode)


def plot_comparison_q_values(run_dirs: List[str], save_dir: str, colors, start_episode: int = 1):
    """Create comprehensive Q-value comparison plots for multiple runs."""
    metrics_list = []
    q_values_list = []
    labels = []
    
    for run_dir in run_dirs:
        with open(os.path.join(run_dir, 'training_logs.json'), 'r') as f:
            metrics = json.load(f)
            metrics_list.append(metrics)
        
        q_values_path = os.path.join(run_dir, 'q_values_log.json')
        if os.path.exists(q_values_path):
            with open(q_values_path, 'r') as f:
                q_values = json.load(f)
                q_values_list.append(q_values)
        
        variant = []
        config = metrics['config']
        if config.get('double', False): variant.append('Double')
        if config.get('dueling', False): variant.append('Dueling')
        if config.get('priority', False): variant.append('PER')
        labels.append(' + '.join(variant) if variant else 'DQN')
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Q-Values Comparison Across Variants', fontsize=16, y=0.95, color='#2D2D2D')

    def plot_metric_comparison(ax, metric_name, title, base_color_idx):
        base_alpha = 0.6
        alpha_increment = 0.1
        std_alpha_base = 0.1
        minmax_alpha_base = 0.05
        alpha_reduction_factor = 0.8
        base_zorder = 2  # Starting z-order for line plots
        
        num_variants = len(q_values_list)
        if num_variants > 2:
            std_alpha_base *= alpha_reduction_factor
            minmax_alpha_base *= alpha_reduction_factor
        
        for idx, (q_values, label, color) in enumerate(zip(q_values_list, labels, colors)):
            episodes = []
            means = []
            stds = []
            mins = []
            maxs = []
            
            for episode_data in q_values['episodes']:
                episode_num = episode_data['episode']
                if episode_num < start_episode:
                    continue
                    
                q_vals = episode_data['q_values']
                if q_vals[metric_name] is not None:
                    episodes.append(episode_num)
                    means.append(q_vals[metric_name]['mean'])
                    stds.append(q_vals[metric_name]['std'])
                    mins.append(q_vals[metric_name]['min'])
                    maxs.append(q_vals[metric_name]['max'])
            
            if episodes:
                episodes = np.array(episodes)
                means = np.array(means)
                stds = np.array(stds)
                mins = np.array(mins)
                maxs = np.array(maxs)
                
                current_alpha = base_alpha + (idx * alpha_increment)
                current_std_alpha = std_alpha_base * (1 + idx * 0.2)
                current_minmax_alpha = minmax_alpha_base * (1 + idx * 0.2)
                current_zorder = base_zorder + idx  # Increment z-order

                # Add shaded regions and lines with updated z-order
                ax.fill_between(episodes, mins, maxs, alpha=current_minmax_alpha, color=color, zorder=current_zorder - 1)
                ax.fill_between(episodes, means - stds, means + stds, alpha=current_std_alpha, color=color, zorder=current_zorder - 1)
                ax.plot(episodes, means, label=label, color=color, linewidth=2, alpha=current_alpha, zorder=current_zorder)
        
        ax.legend(frameon=True, facecolor='white', framealpha=0.9)
        ax.set_title(title)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.2)
    
    plot_metric_comparison(ax1, 'target_q', 'Target Q-Values', 0)
    plot_metric_comparison(ax2, 'current_q', 'Current Q-Values', 1)
    plot_metric_comparison(ax3, 'next_q', 'Next Q-Values', 2)
    plot_metric_comparison(ax4, 'td_errors', 'TD Errors', 3)
    
    fig.text(0.02, 0.02, 'Shaded areas represent ±1 std dev (darker) and min-max range (lighter)', 
             fontsize=8, color='#666666')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/q_values_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_comparison(run_dirs: List[str], save_dir: str, colors, start_episode: int = 1):
    """Create comprehensive comparison plots for multiple training runs."""
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    
    metrics_list = []
    labels = []
    
    for run_dir in run_dirs:
        with open(os.path.join(run_dir, 'training_logs.json'), 'r') as f:
            metrics = json.load(f)
            metrics_list.append(metrics)
            variant = []
            config = metrics['config']
            if config.get('double', False): variant.append('Double')
            if config.get('dueling', False): variant.append('Dueling')
            if config.get('priority', False): variant.append('PER')
            labels.append(' + '.join(variant) if variant else 'DQN')
    
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax3 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2 = fig.add_subplot(gs[0, 1])
    ax4 = fig.add_subplot(gs[1, 1])
    
    fig.suptitle('DQN Variants Comparison', fontsize=16, y=0.95, color='#2D2D2D')
    
    base_alpha = 0.70
    increment = 0.05
    base_zorder = 2  # Starting z-order for line plots
    
    for idx, ((metrics, label), color) in enumerate(zip(zip(metrics_list, labels), colors)):
        # Slice data starting from start_episode
        episodes = range(start_episode, len(metrics['rewards']) + 1)
        rewards = metrics['rewards'][start_episode-1:]
        losses = metrics['losses'][start_episode-1:]
        episode_lengths = metrics['episode_lengths'][start_episode-1:]
        frames_seen = calculate_frames_seen(episode_lengths)
        
        z_order = base_zorder + idx  # Increment z-order for each variant
        
        # Plot rewards
        smoothed_rewards = smooth_data(rewards, window=20)
        ax1.plot(episodes[len(episodes)-len(smoothed_rewards):], 
                 smoothed_rewards, label=label, linewidth=2, color=color, alpha=base_alpha, zorder=z_order)
        ax1.plot(episodes, rewards, alpha=0.2, linewidth=1, color=color, zorder=z_order)
        
        # Plot sample efficiency
        smoothed_eff_rewards = smooth_data(rewards, window=50)
        ax2.plot(frames_seen[len(frames_seen)-len(smoothed_eff_rewards):], 
                 smoothed_eff_rewards, label=label, linewidth=2, color=color, alpha=base_alpha, zorder=z_order)
        
        # Plot losses
        smoothed_losses = smooth_data(losses, window=10)
        ax3.plot(episodes[len(episodes)-len(smoothed_losses):], 
                 smoothed_losses, label=label, linewidth=2, color=color, alpha=base_alpha, zorder=z_order)
        ax3.plot(episodes, losses, alpha=0.2, linewidth=1, color=color, zorder=z_order)
        
        # Plot episode lengths
        smoothed_lengths = smooth_data(episode_lengths, window=20)
        ax4.plot(episodes[len(episodes)-len(smoothed_lengths):], 
                 smoothed_lengths, label=label, linewidth=2, color=color, alpha=base_alpha, zorder=z_order)
        ax4.plot(episodes, episode_lengths, alpha=0.2, linewidth=1, color=color, zorder=z_order)
        
        base_alpha += increment
    
    ax1.set_title('Average Episode Reward')
    ax1.set_ylabel('Reward')
    ax1.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize='x-small')
    
    ax2.set_title('Sample Efficiency')
    ax2.set_xlabel('Total Frames')
    ax2.set_ylabel('Average Reward')
    ax2.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax2.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize='x-small')
    
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Loss')
    ax3.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize='x-small')
    
    ax4.set_title('Episode Lengths')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Steps')
    ax4.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize='x-small')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/variants_comparison_overview.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Additionally plot Q-values comparison if available
    has_q_values = all(os.path.exists(os.path.join(d, 'q_values_log.json')) for d in run_dirs)
    if has_q_values:
        plot_comparison_q_values(run_dirs, save_dir, colors, start_episode)

def main():
    parser = argparse.ArgumentParser(description='Plot DQN training metrics')
    parser.add_argument('--dirs', nargs='+', required=True,
                        help='Directory or directories containing training logs')
    parser.add_argument('--save-dir', type=str, default='plots',
                        help='Directory to save plots')
    parser.add_argument('--start-episode', type=int, default=5,
                        help='Episode number to start plotting from (default: 5)')
    
    args = parser.parse_args()
    colors = setup_style()
    
    if len(args.dirs) == 1:
        # Single run
        with open(os.path.join(args.dirs[0], 'training_logs.json'), 'r') as f:
            metrics = json.load(f)
        
        q_values_log = None
        q_values_path = os.path.join(args.dirs[0], 'q_values_log.json')
        if os.path.exists(q_values_path):
            with open(q_values_path, 'r') as f:
                q_values_log = json.load(f)
        
        plot_single_run(metrics, q_values_log, args.save_dir, colors, args.start_episode)
    else:
        # Multiple runs comparison
        plot_comparison(args.dirs, args.save_dir, colors, args.start_episode)

if __name__ == "__main__":
    main()