#!/usr/bin/env python3
"""
Evaluation script for trained CartPole PPO models.

This script loads a trained model from MLflow or a local checkpoint and evaluates
its performance on the CartPole environment. It generates performance metrics
and creates video recordings of the policy in action.
"""

import argparse
import os
import sys
import pickle
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import gymnasium as gym
import imageio

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.agent.ppo import PPOAgent
from src.environment.cartpole import CartPoleWrapper
from src.visualization.mlflow_logger import MLflowLogger


def load_model_from_mlflow(run_id: str, experiment_name: str = "cartpole-ppo", artifact_path: str = "final_model") -> Tuple[dict, PPOAgent]:
    """
    Load a trained model from MLflow using run ID.
    
    Args:
        run_id: MLflow run ID
        experiment_name: Name of the MLflow experiment
        artifact_path: Path to the model artifact within the run (default: "final_model")
        
    Returns:
        Tuple of (network_params, agent)
    """
    import mlflow
    from mlflow.tracking import MlflowClient
    
    # Set up MLflow logger
    logger = MLflowLogger(experiment_name)
    
    # Download model artifacts
    with tempfile.TemporaryDirectory() as temp_dir:
        client = MlflowClient()
        
        # Get the run and download artifacts
        run = client.get_run(run_id)
        artifacts = client.list_artifacts(run_id, artifact_path)
        
        # Find the model state file
        model_state_path = None
        metadata_path = None
        
        for artifact in artifacts:
            if artifact.path.endswith('model_state.pkl'):
                local_path = client.download_artifacts(run_id, artifact.path, temp_dir)
                model_state_path = os.path.join(temp_dir, local_path)
            elif artifact.path.endswith('metadata.json'):
                local_path = client.download_artifacts(run_id, artifact.path, temp_dir)
                metadata_path = os.path.join(temp_dir, local_path)
        
        if model_state_path is None:
            raise FileNotFoundError(f"Model state file not found in run {run_id}, artifact path {artifact_path}")
        
        # Load model state
        with open(model_state_path, 'rb') as f:
            network_params = pickle.load(f)
        
        # Load metadata if available
        metadata = {}
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
    
    # Create agent with default hyperparameters (can be overridden from metadata)
    hyperparams = metadata.get('hyperparameters', {})
    agent = PPOAgent(
        observation_dim=4,  # CartPole observation space
        action_dim=2,       # CartPole action space
        learning_rate=hyperparams.get('learning_rate', 3e-4),
        clip_epsilon=hyperparams.get('clip_epsilon', 0.2),
        gamma=hyperparams.get('gamma', 0.99),
        gae_lambda=hyperparams.get('gae_lambda', 0.95),
        entropy_coef=hyperparams.get('entropy_coef', 0.01),
        value_coef=hyperparams.get('value_coef', 0.5),
        max_grad_norm=hyperparams.get('max_grad_norm', 0.5),
        epochs_per_update=hyperparams.get('epochs_per_update', 10),
        batch_size=hyperparams.get('batch_size', 64),
        minibatch_size=hyperparams.get('minibatch_size', 64),
    )
    
    # Set the loaded parameters
    agent.network_params = network_params
    
    return network_params, agent


def load_model_from_mlflow_uri(model_uri: str, experiment_name: str = "cartpole-ppo") -> Tuple[dict, PPOAgent]:
    """
    Load a trained model from MLflow URI.
    
    Args:
        model_uri: MLflow model URI (e.g., "runs:/<run_id>/final_model")
        experiment_name: Name of the MLflow experiment
        
    Returns:
        Tuple of (network_params, agent)
    """
    # Extract run ID from model URI
    if model_uri.startswith("runs:/"):
        run_id = model_uri.split("/")[1]
        artifact_path = "/".join(model_uri.split("/")[2:]) if len(model_uri.split("/")) > 2 else "final_model"
    else:
        raise ValueError(f"Unsupported model URI format: {model_uri}")
    
    return load_model_from_mlflow(run_id, experiment_name, artifact_path)


def load_model_from_checkpoint(checkpoint_path: str) -> Tuple[dict, PPOAgent]:
    """
    Load a trained model from a local checkpoint file.
    
    Args:
        checkpoint_path: Path to the checkpoint file (.pkl)
        
    Returns:
        Tuple of (network_params, agent)
    """
    # Load model state
    with open(checkpoint_path, 'rb') as f:
        model_state = pickle.load(f)
    
    # Create agent with default hyperparameters
    agent = PPOAgent(
        observation_dim=4,  # CartPole observation space
        action_dim=2,       # CartPole action space
    )
    
    # Set the loaded parameters
    agent.network_params = model_state
    
    return model_state, agent


def evaluate_agent(
    agent: PPOAgent,
    env: CartPoleWrapper,
    num_episodes: int = 100,
    max_steps_per_episode: int = 500,
    render: bool = False,
    video_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate a trained agent on the CartPole environment.
    
    Args:
        agent: Trained PPO agent
        env: CartPole environment
        num_episodes: Number of evaluation episodes
        max_steps_per_episode: Maximum steps per episode
        render: Whether to render the environment
        video_dir: Directory to save videos (required)
        
    Returns:
        Dictionary containing evaluation metrics
    """
    episode_rewards = []
    episode_lengths = []
    episode_data = []
    video_paths = []
    
    # Create video directory if not specified
    if video_dir is None:
        video_dir = "evaluation_videos"
    os.makedirs(video_dir, exist_ok=True)
    
    # Create environment with rgb_array render mode for frame capture
    frame_env = CartPoleWrapper(
        render_mode='rgb_array',
        max_episode_steps=max_steps_per_episode,
        normalize_observations=True
    )
    
    for episode in range(num_episodes):
        obs, info = frame_env.reset()
        episode_reward = 0
        episode_length = 0
        step_data = []
        episode_frames = []
        
        for step in range(max_steps_per_episode):
            # Greedy action selection for evaluation (deterministic)
            policy_logits, value = agent.network.apply(agent.network_params, obs[None, :])
            action = jnp.argmax(policy_logits, axis=-1)
            log_probs = jax.nn.log_softmax(policy_logits)
            # Extract log-prob of chosen action for logging
            lp = log_probs[0, int(action.item())]
            
            # Take environment step
            next_obs, reward, terminated, truncated, info = frame_env.step(int(action.item()))
            
            done = terminated or truncated
            
            # Capture frame
            frame = frame_env.env.render()
            episode_frames.append(frame)
            
            # Store step data
            step_data.append({
                'observation': obs.tolist(),
                'action': int(action.item()),
                'reward': reward,
                'next_observation': next_obs.tolist(),
                'done': done,
                'log_prob': float(lp.item()),
                'value': float(value.item())
            })
            
            episode_reward += float(reward)
            episode_length += 1
            obs = next_obs
            
            if render:
                env.env.render()
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_data.append(step_data)
        
        # Save video for this episode at 60fps
        video_path = os.path.join(video_dir, f"episode_{episode + 1:03d}.mp4")
        imageio.mimsave(video_path, episode_frames, fps=60)
        video_paths.append(video_path)
        
        print(f"Episode {episode + 1:3d}/{num_episodes} | "
              f"Reward: {episode_reward:6.1f} | "
              f"Length: {episode_length:3d} | "
              f"Video: {video_path}")
    
    # Close video environment
    frame_env.close()
    
    # Compute statistics
    episode_rewards = np.array(episode_rewards)
    episode_lengths = np.array(episode_lengths)
    
    metrics = {
        'num_episodes': num_episodes,
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'min_reward': float(np.min(episode_rewards)),
        'max_reward': float(np.max(episode_rewards)),
        'median_reward': float(np.median(episode_rewards)),
        'mean_length': float(np.mean(episode_lengths)),
        'std_length': float(np.std(episode_lengths)),
        'min_length': float(np.min(episode_lengths)),
        'max_length': float(np.max(episode_lengths)),
        'success_rate': float(np.mean(episode_rewards >= 195)),  # CartPole success criterion
        'episode_rewards': episode_rewards.tolist(),
        'episode_lengths': episode_lengths.tolist(),
        'episode_data': episode_data,
        'video_paths': video_paths,
        'video_dir': video_dir
    }
    
    return metrics


def create_evaluation_report(metrics: Dict[str, Any], output_path: str) -> None:
    """
    Create a detailed evaluation report.
    
    Args:
        metrics: Evaluation metrics
        output_path: Path to save the report
    """
    report = {
        'evaluation_summary': {
            'total_episodes': metrics['num_episodes'],
            'mean_reward': metrics['mean_reward'],
            'std_reward': metrics['std_reward'],
            'success_rate': metrics['success_rate'],
            'converged': metrics['mean_reward'] >= 195
        },
        'detailed_metrics': {
            'reward_statistics': {
                'mean': metrics['mean_reward'],
                'std': metrics['std_reward'],
                'min': metrics['min_reward'],
                'max': metrics['max_reward'],
                'median': metrics['median_reward']
            },
            'length_statistics': {
                'mean': metrics['mean_length'],
                'std': metrics['std_length'],
                'min': metrics['min_length'],
                'max': metrics['max_length']
            }
        },
        'episode_data': {
            'rewards': metrics['episode_rewards'],
            'lengths': metrics['episode_lengths']
        }
    }
    
    # Save report
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nEvaluation report saved to: {output_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate trained PPO model on CartPole')
    
    # Model source (exactly one required)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--run-id', type=str,
                           help='MLflow run ID (e.g., "ed91b0290af04f9b9fef07e6d72b44f6")')
    model_group.add_argument('--model-uri', type=str, 
                           help='MLflow model URI (e.g., "runs:/<run_id>/final_model")')
    model_group.add_argument('--checkpoint', type=str,
                           help='Path to local checkpoint file (.pkl)')
    
    # Evaluation parameters
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of evaluation episodes (default: 100)')
    parser.add_argument('--max-steps', type=int, default=500,
                       help='Maximum steps per episode (default: 500)')
    parser.add_argument('--render', action='store_true',
                       help='Render environment during evaluation')
    parser.add_argument('--video-dir', type=str, default='evaluation_videos',
                       help='Directory to save videos (default: evaluation_videos)')
    
    # Output parameters
    parser.add_argument('--output', type=str, default='evaluation_report.json',
                       help='Output file for evaluation report (default: evaluation_report.json)')
    parser.add_argument('--experiment', type=str, default='cartpole-ppo',
                       help='MLflow experiment name (default: cartpole-ppo)')
    parser.add_argument('--artifact-path', type=str, default='final_model',
                       help='MLflow artifact path within the run (default: final_model)')
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    print("=" * 60)
    print("CARTPOLE PPO MODEL EVALUATION")
    print("=" * 60)
    
    # Load model
    print("Loading model...")
    try:
        if args.run_id:
            network_params, agent = load_model_from_mlflow(args.run_id, args.experiment, args.artifact_path)
            print(f"Loaded model from MLflow run: {args.run_id}")
        elif args.model_uri:
            network_params, agent = load_model_from_mlflow_uri(args.model_uri, args.experiment)
            print(f"Loaded model from MLflow URI: {args.model_uri}")
        else:
            network_params, agent = load_model_from_checkpoint(args.checkpoint)
            print(f"Loaded model from checkpoint: {args.checkpoint}")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Create environment
    print("Creating environment...")
    render_mode = 'human' if args.render else None
    env = CartPoleWrapper(
        render_mode=render_mode,
        max_episode_steps=args.max_steps,
        normalize_observations=True
    )
    
    # Evaluate agent
    print(f"\nEvaluating agent for {args.episodes} episodes...")
    print("-" * 60)
    
    metrics = evaluate_agent(
        agent=agent,
        env=env,
        num_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        render=args.render,
        video_dir=args.video_dir
    )
    
    # Print results
    print("-" * 60)
    print("\nEvaluation Results:")
    print(f"Episodes evaluated: {metrics['num_episodes']}")
    print(f"Mean reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    print(f"Min/Max reward: {metrics['min_reward']:.2f} / {metrics['max_reward']:.2f}")
    print(f"Mean episode length: {metrics['mean_length']:.1f} ± {metrics['std_length']:.1f}")
    print(f"Success rate (reward ≥ 195): {metrics['success_rate']:.2%}")
    print(f"Converged: {metrics['mean_reward'] >= 195}")
    print(f"\nVideos saved to: {metrics['video_dir']}")
    for video_path in metrics['video_paths']:
        print(f"  - {video_path}")
    
    # Create evaluation report
    create_evaluation_report(metrics, args.output)
    
    # Close environment
    env.close()
    
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()
