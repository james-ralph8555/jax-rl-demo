"""Visualization utilities for training metrics and results"""

from typing import List, Dict, Any, Optional, Tuple
import os
import json

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import seaborn as sns  # type: ignore


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


def plot_policy_distribution(
    action_probs: List[np.ndarray],
    observations: List[np.ndarray],
    save_path: Optional[str] = None,
    show: bool = False,
    title: str = "Policy Distribution"
) -> None:
    """
    Plot policy distribution over time
    
    Args:
        action_probs: List of action probability arrays
        observations: List of observations corresponding to action probs
        save_path: Path to save the plot
        show: Whether to display the plot
        title: Plot title
    """
    if not action_probs:
        return
        
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=16)
    
    # Convert to numpy array for easier manipulation
    action_probs_array = np.array(action_probs)
    
    # Plot action probability over time
    if len(action_probs_array) > 0 and action_probs_array.ndim >= 2:
        axes[0, 0].plot(action_probs_array[:, 0], label='Action 0 (Left)', alpha=0.7)
        axes[0, 0].plot(action_probs_array[:, 1], label='Action 1 (Right)', alpha=0.7)
        axes[0, 0].set_title('Action Probabilities Over Time')
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('Probability')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        axes[0, 0].set_ylim([0, 1])
    
    # Plot probability distribution histogram
    if len(action_probs_array) > 0 and action_probs_array.ndim >= 2:
        axes[0, 1].hist(action_probs_array[:, 0], bins=30, alpha=0.7, label='Action 0', edgecolor='black')
        axes[0, 1].hist(action_probs_array[:, 1], bins=30, alpha=0.7, label='Action 1', edgecolor='black')
        axes[0, 1].set_title('Action Probability Distribution')
        axes[0, 1].set_xlabel('Probability')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # Plot policy entropy over time
    if len(action_probs_array) > 0 and action_probs_array.ndim >= 2:
        entropy = -np.sum(action_probs_array * np.log(action_probs_array + 1e-8), axis=1)
        axes[1, 0].plot(entropy, alpha=0.7)
        axes[1, 0].set_title('Policy Entropy Over Time')
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('Entropy')
        axes[1, 0].grid(True)
    
    # Plot confidence (max probability) over time
    if len(action_probs_array) > 0 and action_probs_array.ndim >= 2:
        confidence = np.max(action_probs_array, axis=1)
        axes[1, 1].plot(confidence, alpha=0.7, color='red')
        axes[1, 1].set_title('Policy Confidence Over Time')
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].set_ylabel('Max Probability')
        axes[1, 1].grid(True)
        axes[1, 1].set_ylim([0.5, 1.0])
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_advanced_learning_curves(
    rewards: List[float],
    losses: Dict[str, List[float]],
    eval_rewards: Optional[List[float]] = None,
    window_sizes: List[int] = [10, 50, 100],
    save_path: Optional[str] = None,
    show: bool = False
) -> None:
    """
    Plot advanced learning curves with multiple window sizes and evaluation metrics
    
    Args:
        rewards: List of episode rewards
        losses: Dictionary of loss values
        eval_rewards: Optional list of evaluation rewards
        window_sizes: List of moving average window sizes
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    fig = plt.figure(figsize=(16, 12))
    
    # Create grid specification for better layout
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Plot raw rewards
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(rewards, alpha=0.3, color='blue', label='Raw Rewards')
    ax1.set_title('Episode Rewards Over Time', fontsize=14)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.grid(True, alpha=0.3)
    
    # Plot moving averages with different window sizes
    colors = ['red', 'green', 'orange']
    for i, window in enumerate(window_sizes):
        if len(rewards) >= window:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax1.plot(range(window-1, len(rewards)), moving_avg, 
                    color=colors[i % len(colors)], linewidth=2, 
                    label=f'MA {window}')
    
    # Add target line
    ax1.axhline(y=195, color='black', linestyle='--', alpha=0.7, label='Target (195)')
    ax1.legend()
    
    # Plot loss curves
    loss_keys = [k for k in losses.keys() if 'loss' in k]
    if loss_keys:
        for i, key in enumerate(loss_keys[:3]):  # Plot up to 3 loss curves
            ax = fig.add_subplot(gs[1, i])
            ax.plot(losses[key], color=colors[i % len(colors)], alpha=0.8)
            ax.set_title(f'{key.replace("_", " ").title()}')
            ax.set_xlabel('Update Step')
            ax.set_ylabel('Loss')
            ax.grid(True, alpha=0.3)
    
    # Plot evaluation rewards if available
    if eval_rewards:
        ax_eval = fig.add_subplot(gs[2, :2])
        eval_episodes = np.arange(0, len(rewards), len(rewards)//len(eval_rewards))[:len(eval_rewards)]
        ax_eval.plot(eval_episodes, eval_rewards, 'o-', color='purple', linewidth=2, markersize=6)
        ax_eval.set_title('Evaluation Rewards', fontsize=14)
        ax_eval.set_xlabel('Episode')
        ax_eval.set_ylabel('Evaluation Reward')
        ax_eval.grid(True, alpha=0.3)
        ax_eval.axhline(y=195, color='black', linestyle='--', alpha=0.7, label='Target (195)')
        ax_eval.legend()
    
    # Plot learning statistics
    ax_stats = fig.add_subplot(gs[2, 2])
    if len(rewards) > 0:
        stats_text = f"Total Episodes: {len(rewards)}\n"
        stats_text += f"Final Reward: {rewards[-1]:.1f}\n"
        stats_text += f"Best Reward: {max(rewards):.1f}\n"
        stats_text += f"Mean Reward: {np.mean(rewards):.1f}\n"
        stats_text += f"Std Reward: {np.std(rewards):.1f}\n"
        
        if len(rewards) >= 100:
            recent_avg = np.mean(rewards[-100:])
            stats_text += f"Last 100 Avg: {recent_avg:.1f}\n"
            stats_text += f"Converged: {recent_avg >= 195}"
        
        ax_stats.text(0.1, 0.5, stats_text, transform=ax_stats.transAxes, 
                     fontsize=11, verticalalignment='center', 
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        ax_stats.set_title('Learning Statistics')
        ax_stats.axis('off')
    
    plt.suptitle('Advanced Learning Progress Analysis', fontsize=16, y=0.98)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_hyperparameter_heatmap(
    results: Dict[str, Dict[str, Any]],
    metric: str = 'final_reward',
    save_path: Optional[str] = None,
    show: bool = False
) -> None:
    """
    Create a heatmap visualization for hyperparameter analysis
    
    Args:
        results: Dictionary of results for different hyperparameter combinations
        metric: Metric to visualize in the heatmap
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    if not results:
        return
        
    # Extract hyperparameter names and values
    all_params = set()
    for run_name, run_results in results.items():
        if isinstance(run_results, dict) and 'hyperparams' in run_results:
            hyperparams = run_results['hyperparams']
            if isinstance(hyperparams, dict):
                all_params.update(hyperparams.keys())
    
    # Filter to only 2 main hyperparameters for visualization
    main_params = list(all_params)[:2]
    if len(main_params) < 2:
        print("Need at least 2 hyperparameters for heatmap visualization")
        return
    
    # Create data for heatmap
    param1_values = sorted(set(run['hyperparams'].get(main_params[0], 0) 
                              for run in results.values() 
                              if isinstance(run, dict) and 'hyperparams' in run and isinstance(run['hyperparams'], dict)))
    param2_values = sorted(set(run['hyperparams'].get(main_params[1], 0) 
                              for run in results.values() 
                              if isinstance(run, dict) and 'hyperparams' in run and isinstance(run['hyperparams'], dict)))
    
    heatmap_data = np.zeros((len(param1_values), len(param2_values)))
    
    for run_name, run_results in results.items():
        if (isinstance(run_results, dict) and 
            'hyperparams' in run_results and 
            isinstance(run_results['hyperparams'], dict) and 
            metric in run_results):
            
            p1_val = run_results['hyperparams'].get(main_params[0])
            p2_val = run_results['hyperparams'].get(main_params[1])
            
            if p1_val in param1_values and p2_val in param2_values:
                i = param1_values.index(p1_val)
                j = param2_values.index(p2_val)
                heatmap_data[i, j] = run_results[metric]
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, 
                xticklabels=[f"{v:.3f}" for v in param2_values],
                yticklabels=[f"{v:.3f}" for v in param1_values],
                annot=True, fmt=".1f", cmap="YlOrRd", cbar_kws={'label': metric})
    
    plt.title(f'Hyperparameter Heatmap: {metric}')
    plt.xlabel(main_params[1])
    plt.ylabel(main_params[0])
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_training_stability(
    rewards: List[float],
    window_size: int = 50,
    save_path: Optional[str] = None,
    show: bool = False
) -> None:
    """
    Plot training stability analysis
    
    Args:
        rewards: List of episode rewards
        window_size: Window size for rolling statistics
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    if len(rewards) < window_size:
        return
        
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Training Stability Analysis', fontsize=16)
    
    rewards_array = np.array(rewards)
    
    # Rolling mean
    rolling_mean = np.convolve(rewards_array, np.ones(window_size)/window_size, mode='valid')
    axes[0, 0].plot(range(window_size-1, len(rewards)), rolling_mean)
    axes[0, 0].set_title(f'Rolling Mean (window={window_size})')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Mean Reward')
    axes[0, 0].grid(True)
    axes[0, 0].axhline(y=195, color='red', linestyle='--', alpha=0.7)
    
    # Rolling standard deviation
    rolling_std = []
    for i in range(window_size-1, len(rewards)):
        window_data = rewards_array[i-window_size+1:i+1]
        rolling_std.append(np.std(window_data))
    
    axes[0, 1].plot(range(window_size-1, len(rewards)), rolling_std, color='orange')
    axes[0, 1].set_title(f'Rolling Std Dev (window={window_size})')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Std Dev')
    axes[0, 1].grid(True)
    
    # Coefficient of variation
    rolling_cv = []
    for i in range(window_size-1, len(rewards)):
        window_data = rewards_array[i-window_size+1:i+1]
        mean = np.mean(window_data)
        std = np.std(window_data)
        rolling_cv.append(std / mean if mean > 0 else 0)
    
    axes[1, 0].plot(range(window_size-1, len(rewards)), rolling_cv, color='green')
    axes[1, 0].set_title(f'Coefficient of Variation (window={window_size})')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('CV (Std/Mean)')
    axes[1, 0].grid(True)
    
    # Performance consistency (percentage of episodes above threshold)
    consistency = []
    for i in range(window_size-1, len(rewards)):
        window_data = rewards_array[i-window_size+1:i+1]
        above_threshold = np.sum(window_data >= 195) / len(window_data) * 100
        consistency.append(above_threshold)
    
    axes[1, 1].plot(range(window_size-1, len(rewards)), consistency, color='purple')
    axes[1, 1].set_title(f'Consistency (% episodes ≥ 195, window={window_size})')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Consistency (%)')
    axes[1, 1].grid(True)
    axes[1, 1].set_ylim([0, 100])
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def create_comprehensive_analysis(
    training_data: Dict[str, Any],
    save_dir: str,
    show_plots: bool = False
) -> None:
    """
    Create comprehensive analysis plots for training data
    
    Args:
        training_data: Dictionary containing all training data
        save_dir: Directory to save plots
        show_plots: Whether to display plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    rewards = training_data.get('episode_rewards', [])
    losses = training_data.get('losses', {})
    eval_rewards = training_data.get('eval_rewards', [])
    
    # Advanced learning curves
    plot_advanced_learning_curves(
        rewards, losses, eval_rewards,
        save_path=os.path.join(save_dir, 'advanced_learning_curves.png'),
        show=show_plots
    )
    
    # Training stability
    plot_training_stability(
        rewards,
        save_path=os.path.join(save_dir, 'training_stability.png'),
        show=show_plots
    )
    
    # Episode statistics
    episode_lengths = training_data.get('episode_lengths', [])
    if episode_lengths:
        plot_episode_statistics(
            episode_lengths, rewards,
            save_path=os.path.join(save_dir, 'episode_statistics.png'),
            show=show_plots
        )
    
    # Training summary
    final_losses: Dict[str, float] = {}
    for k, v in losses.items():
        if v and len(v) > 0:
            final_losses[k] = float(v[-1]) if hasattr(v[-1], 'item') else float(v[-1])
        else:
            final_losses[k] = 0.0
    hyperparams = training_data.get('hyperparams', {})
    create_training_summary_plot(
        rewards, final_losses, hyperparams,
        save_path=os.path.join(save_dir, 'training_summary.png')
    )
    
    print(f"Comprehensive analysis plots saved to {save_dir}")


def plot_gradient_flow(
    grad_norms: Dict[str, float],
    step: int,
    save_path: Optional[str] = None,
    show: bool = False
) -> None:
    """
    Plot gradient flow using a log-scale violin plot grouped by layer with
    actor/critic separation for clarity.

    Args:
        grad_norms: Dictionary of per-parameter gradient norms. Keys are path-like
            names (e.g., 'params/actor/Dense_0/kernel').
        step: Current training step (for title annotation).
        save_path: Optional path to save the figure.
        show: Whether to display the plot.
    """
    if not grad_norms:
        return

    # Build records for seaborn: x=layer, y=norm, hue=network(actor/critic)
    xs: list[str] = []
    ys: list[float] = []
    hues: list[str] = []

    def _parse_entry(name: str) -> tuple[str, str]:
        # name like 'params/actor/Dense_0/kernel' or 'actor/Dense_0/bias'
        parts = name.split('/')
        # Drop optional leading 'params'
        if parts and parts[0] == 'params':
            parts = parts[1:]
        # Determine network and layer
        network = 'shared'
        layer = '/'.join(parts)
        if 'actor' in parts:
            network = 'actor'
            try:
                idx = parts.index('actor')
                # Layer typically the next token (e.g., Dense_0); fall back gracefully
                if len(parts) > idx + 1:
                    layer = parts[idx + 1]
            except ValueError:
                pass
        elif 'critic' in parts:
            network = 'critic'
            try:
                idx = parts.index('critic')
                if len(parts) > idx + 1:
                    layer = parts[idx + 1]
            except ValueError:
                pass
        else:
            # If neither actor nor critic explicitly present, try to take penultimate as layer
            if len(parts) >= 2:
                layer = parts[-2]
        return layer, network

    for name, value in grad_norms.items():
        try:
            norm = float(np.asarray(value).reshape(()))
        except Exception:
            continue
        layer, network = _parse_entry(name)
        xs.append(layer)
        ys.append(max(norm, 1e-12))  # avoid non-positive for log-scale
        hues.append(network)

    if not ys:
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    unique_hues = sorted(set(hues))
    split = len(unique_hues) == 2  # split violins only if exactly two groups

    try:
        sns.violinplot(
            x=xs,
            y=ys,
            hue=hues,
            split=split,
            inner='quartile',
            scale='width',
            cut=0,
            ax=ax,
            palette='Set2',
        )
    except Exception:
        # Fallback without split if seaborn version/hue levels disagree
        sns.violinplot(
            x=xs,
            y=ys,
            hue=hues if len(unique_hues) > 0 else None,
            inner='quartile',
            scale='width',
            cut=0,
            ax=ax,
            palette='Set2',
        )

    ax.set_yscale('log')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Gradient Norm (log scale)')
    ax.set_title(f'Gradient Flow (Violin) at Step {step}')
    ax.grid(True, which='both', axis='y', alpha=0.2)

    # Reference baseline: global median
    median_val = float(np.median(ys))
    ax.axhline(median_val, color='gray', linestyle='--', linewidth=1, label='median')
    # Ensure legend shows median and hue categories cleanly
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels and labels[0] == 'hue':
        # Some seaborn versions add a label header; clean it up
        labels = labels[1:]
        handles = handles[1:]
    if labels:
        ax.legend(handles=handles, labels=labels, title='Network', loc='best', frameon=True)
    else:
        ax.legend(loc='best', frameon=True)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


def plot_kl_divergence(
    kl_values: List[float],
    step: int,
    save_path: Optional[str] = None,
    show: bool = False
) -> None:
    """
    Plot KL divergence analysis over time
    
    Args:
        kl_values: List of KL divergence values
        step: Current training step
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    if not kl_values:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f'KL Divergence Analysis - Step {step}', fontsize=16)
    
    # Plot KL divergence over time
    axes[0].plot(kl_values, alpha=0.7, color='blue')
    axes[0].set_title('KL Divergence Over Updates')
    axes[0].set_xlabel('Update Step')
    axes[0].set_ylabel('KL Divergence')
    axes[0].grid(True, alpha=0.3)
    
    # Add threshold lines for common KL targets
    axes[0].axhline(y=0.01, color='orange', linestyle='--', alpha=0.7, label='Conservative (0.01)')
    axes[0].axhline(y=0.02, color='red', linestyle='--', alpha=0.7, label='Aggressive (0.02)')
    axes[0].legend()
    
    # Plot histogram of recent KL values
    recent_kl = kl_values[-100:] if len(kl_values) >= 100 else kl_values
    axes[1].hist(recent_kl, bins=20, alpha=0.7, edgecolor='black', color='skyblue')
    axes[1].set_title(f'KL Distribution (Last {len(recent_kl)} values)')
    axes[1].set_xlabel('KL Divergence')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(True, alpha=0.3)
    
    # Add statistics text
    mean_kl = float(np.mean(recent_kl))
    std_kl = float(np.std(recent_kl))
    max_kl = float(np.max(recent_kl))
    
    stats_text = f'Mean: {mean_kl:.4f}\nStd: {std_kl:.4f}\nMax: {max_kl:.4f}'
    axes[1].text(0.95, 0.95, stats_text, transform=axes[1].transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
