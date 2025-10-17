"""Visualization utilities for training metrics and results"""

from typing import List, Dict, Any, Optional
import os

# Import plotting libraries - these will be available when the environment is set up
try:
    import matplotlib.pyplot as plt  # type: ignore
    import numpy as np  # type: ignore
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    plt = None
    np = None
    print("Warning: Matplotlib/NumPy not available. Install with: pip install matplotlib numpy")


def plot_training_curves(
    rewards: List[float],
    losses: Dict[str, List[float]],
    save_path: Optional[str] = None,
    show: bool = False
) -> None:
    """
    Plot training curves for rewards and losses
    
    Args:
        rewards: List of episode rewards
        losses: Dictionary of loss values (e.g., {'policy_loss': [...], 'value_loss': [...]})
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    if not PLOTTING_AVAILABLE or plt is None or np is None:
        print("Plotting not available - skipping plot generation")
        return
        
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Training Progress', fontsize=16)
    
    # Plot rewards
    axes[0, 0].plot(rewards)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True)
    
    # Plot moving average of rewards
    if len(rewards) > 10:
        window_size = min(100, len(rewards) // 10)
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        axes[0, 1].plot(moving_avg)
        axes[0, 1].set_title(f'Moving Average Rewards (window={window_size})')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Average Reward')
        axes[0, 1].grid(True)
    
    # Plot policy loss
    if 'policy_loss' in losses:
        axes[1, 0].plot(losses['policy_loss'])
        axes[1, 0].set_title('Policy Loss')
        axes[1, 0].set_xlabel('Update Step')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True)
    
    # Plot value loss
    if 'value_loss' in losses:
        axes[1, 1].plot(losses['value_loss'])
        axes[1, 1].set_title('Value Loss')
        axes[1, 1].set_xlabel('Update Step')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_episode_statistics(
    episode_lengths: List[int],
    rewards: List[float],
    save_path: Optional[str] = None,
    show: bool = False
) -> None:
    """
    Plot episode statistics
    
    Args:
        episode_lengths: List of episode lengths
        rewards: List of episode rewards
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    if not PLOTTING_AVAILABLE or plt is None:
        print("Plotting not available - skipping plot generation")
        return
        
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Episode Statistics', fontsize=16)
    
    # Plot episode lengths
    axes[0].plot(episode_lengths)
    axes[0].set_title('Episode Lengths')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Steps')
    axes[0].grid(True)
    
    # Plot reward distribution
    axes[1].hist(rewards, bins=30, alpha=0.7, edgecolor='black')
    axes[1].set_title('Reward Distribution')
    axes[1].set_xlabel('Reward')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_hyperparameter_comparison(
    results: Dict[str, Dict[str, List[float]]],
    metric: str = 'rewards',
    save_path: Optional[str] = None,
    show: bool = False
) -> None:
    """
    Plot comparison of different hyperparameter configurations
    
    Args:
        results: Dictionary of results for different configurations
        metric: Metric to plot ('rewards', 'losses', etc.)
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    if not PLOTTING_AVAILABLE or plt is None or np is None:
        print("Plotting not available - skipping plot generation")
        return
        
    plt.figure(figsize=(10, 6))
    
    for config_name, config_results in results.items():
        if metric in config_results:
            values = config_results[metric]
            if metric == 'rewards' and len(values) > 10:
                # Plot moving average for rewards
                window_size = min(100, len(values) // 10)
                values = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
            plt.plot(values, label=config_name, alpha=0.8)
    
    plt.title(f'Hyperparameter Comparison - {metric}')
    plt.xlabel('Episode' if metric == 'rewards' else 'Update Step')
    plt.ylabel('Reward' if metric == 'rewards' else 'Loss')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def create_training_summary_plot(
    final_rewards: List[float],
    final_losses: Dict[str, float],
    hyperparams: Dict[str, Any],
    save_path: Optional[str] = None
) -> None:
    """
    Create a summary plot of training results
    
    Args:
        final_rewards: Final episode rewards
        final_losses: Final loss values
        hyperparams: Training hyperparameters
        save_path: Path to save the plot
    """
    if not PLOTTING_AVAILABLE or plt is None:
        print("Plotting not available - skipping plot generation")
        return
        
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Training Summary', fontsize=16)
    
    # Plot final rewards distribution
    axes[0, 0].hist(final_rewards, bins=20, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Final Rewards Distribution')
    axes[0, 0].set_xlabel('Reward')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True)
    
    # Plot final losses
    loss_names = list(final_losses.keys())
    loss_values = list(final_losses.values())
    axes[0, 1].bar(loss_names, loss_values)
    axes[0, 1].set_title('Final Losses')
    axes[0, 1].set_ylabel('Loss Value')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Plot training progress (last 100 episodes)
    if len(final_rewards) > 100:
        recent_rewards = final_rewards[-100:]
        axes[1, 0].plot(recent_rewards)
        axes[1, 0].set_title('Last 100 Episodes')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Reward')
        axes[1, 0].grid(True)
    
    # Display hyperparameters
    hyperparam_text = '\n'.join([f'{k}: {v}' for k, v in hyperparams.items()])
    axes[1, 1].text(0.1, 0.5, hyperparam_text, transform=axes[1, 1].transAxes, 
                    fontsize=10, verticalalignment='center')
    axes[1, 1].set_title('Hyperparameters')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()