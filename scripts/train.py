#!/usr/bin/env python3
"""
Training script for CartPole PPO with MLflow integration.

This script trains a PPO agent on the CartPole environment with comprehensive
MLflow tracking for hyperparameters, metrics, and model artifacts.
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax
from src.agent.ppo import PPOAgent
from src.environment.cartpole import CartPoleWrapper
from src.training.trainer import PPOTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train PPO agent on CartPole with MLflow tracking"
    )
    
    # Training parameters
    parser.add_argument("--episodes", type=int, default=1000,
                        help="Maximum training episodes (default: 1000)")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Maximum steps per episode (default: 500)")
    parser.add_argument("--target-reward", type=float, default=195.0,
                        help="Target reward for convergence (default: 195.0)")
    parser.add_argument("--convergence-window", type=int, default=20,
                        help="Episodes to check for convergence (default: 20)")
    parser.add_argument("--early-stopping-patience", type=int, default=10,
                        help="Episodes to wait after convergence before stopping (default: 10)")
    
    # Evaluation parameters
    parser.add_argument("--eval-frequency", type=int, default=50,
                        help="Frequency of evaluation episodes (default: 50)")
    parser.add_argument("--eval-episodes", type=int, default=10,
                        help="Number of evaluation episodes (default: 10)")
    
    # Logging parameters
    parser.add_argument("--log-frequency", type=int, default=10,
                        help="Frequency of logging metrics (default: 10)")
    
    # PPO hyperparameters
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="Learning rate (default: 3e-4)")
    parser.add_argument("--clip-epsilon", type=float, default=0.2,
                        help="PPO clipping parameter (default: 0.2)")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor (default: 0.99)")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="GAE lambda parameter (default: 0.95)")
    parser.add_argument("--entropy-coef", type=float, default=0.01,
                        help="Entropy coefficient (default: 0.01)")
    parser.add_argument("--value-coef", type=float, default=0.5,
                        help="Value function coefficient (default: 0.5)")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="Maximum gradient norm (default: 0.5)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for PPO updates (default: 64)")
    parser.add_argument("--epochs-per-update", type=int, default=10,
                        help="Number of epochs per PPO update (default: 10)")
    
    # MLflow parameters
    parser.add_argument("--experiment-name", type=str, default="cartpole-ppo",
                        help="MLflow experiment name (default: cartpole-ppo)")
    parser.add_argument("--disable-mlflow", action="store_true",
                        help="Disable MLflow tracking")
    
    # GIF recording parameters
    parser.add_argument("--video-record-freq", type=int, default=200,
                        help="Frequency of GIF recording during evaluation (default: 200)")
    

    
    # Random seed
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    print("=" * 60)
    print("CARTPOLE PPO TRAINING WITH MLFLOW INTEGRATION")
    print("=" * 60)
    
    # Create random key
    key = jax.random.PRNGKey(args.seed)
    
    # Create environment
    print("Creating CartPole environment...")
    
    # Determine GIF directory
    video_dir = None
    if not args.disable_mlflow and args.video_record_freq:
        # Create a local directory for GIFs that will be logged to MLflow
        video_dir = "data/evaluation_gifs"
        os.makedirs(video_dir, exist_ok=True)
    
    env = CartPoleWrapper(
        max_episode_steps=args.max_steps,
        video_record_freq=args.video_record_freq if not args.disable_mlflow else None,
        video_dir=video_dir
    )
    
    # Create agent
    print("Creating PPO agent...")
    agent = PPOAgent(
        observation_dim=env.observation_dim,
        action_dim=env.action_dim,
        learning_rate=args.learning_rate,
        clip_epsilon=args.clip_epsilon,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        batch_size=args.batch_size,
        epochs_per_update=args.epochs_per_update,
        key=key
    )
    
    # Create trainer
    print("Creating trainer...")
    trainer = PPOTrainer(
        agent=agent,
        env=env,
        max_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        target_reward=args.target_reward,
        convergence_window=args.convergence_window,
        early_stopping_patience=args.early_stopping_patience,
        eval_frequency=args.eval_frequency,
        eval_episodes=args.eval_episodes,
        log_frequency=args.log_frequency,
        enable_mlflow=not args.disable_mlflow,
        mlflow_experiment_name=args.experiment_name,
        video_record_frequency=args.video_record_freq,
        log_gradient_flow=True,
        key=key
    )
    
    # Training callback
    def training_callback(metrics):
        episode = metrics['episode']
        step_metrics = metrics['step_metrics']
        
        if episode % args.log_frequency == 0:
            print(f"Episode {episode:4d} | "
                  f"Reward: {step_metrics['avg_reward']:6.1f} | "
                  f"Length: {step_metrics['avg_length']:5.1f} | "
                  f"Loss: {step_metrics.get('total_loss', 0):.3f}")
        
        if metrics['eval_metrics']:
            eval_metrics = metrics['eval_metrics']
            print(f"  Evaluation: {eval_metrics['avg_reward']:.1f} ± {eval_metrics['std_reward']:.1f}")
    
    print(f"\nStarting training...")
    if not args.disable_mlflow:
        print(f"MLflow tracking URI: {os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')}")
        print(f"Experiment name: {args.experiment_name}")
    print("-" * 60)
    
    # Train the agent
    results = trainer.train(callback=training_callback)
    
    # Display results
    print("\n" + "=" * 60)
    print("TRAINING RESULTS")
    print("=" * 60)
    print(f"Total episodes: {results['total_episodes']}")
    print(f"Total steps: {results['total_steps']}")
    print(f"Training time: {results['training_time']:.2f} seconds")
    print(f"Final evaluation reward: {results['final_avg_reward']:.2f} ± {results['final_std_reward']:.2f}")
    print(f"Best average reward: {results['best_avg_reward']:.2f}")
    print(f"Converged: {results['converged']}")
    
    if results['converged']:
        print("\nAgent successfully converged!")
        if not args.disable_mlflow:
            print("Model has been registered to MLflow Model Registry")
    else:
        print("\nAgent did not converge within the training period")
    
    if not args.disable_mlflow:
        print("\n" + "=" * 60)
        print("MLFLOW INTEGRATION SUMMARY")
        print("=" * 60)
        print("Hyperparameters logged to MLflow")
        print("Training metrics tracked in real-time")
        print("Episode rewards and losses logged")
        print("Evaluation metrics recorded")
        print("Training curves saved as artifacts")
        print("Model checkpoint saved to MLflow")
        if results['converged']:
            print("Model registered to Model Registry")
        print("Dashboard data created for visualization")
        
        print(f"\nTo view the results in MLflow:")
        print(f"1. Start MLflow UI: mlflow ui")
        print(f"2. Open browser: {os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')}")
        print(f"3. Navigate to experiment: {args.experiment_name}")
    
    # Close environment
    env.close()
    
    return results


if __name__ == "__main__":
    results = main()