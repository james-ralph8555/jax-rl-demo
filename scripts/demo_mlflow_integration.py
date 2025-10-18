#!/usr/bin/env python3
"""Demo script showing MLflow integration with PPO training"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    """Run a demo training session with MLflow integration"""
    print("=" * 60)
    print("CARTPOLE PPO WITH MLFLOW INTEGRATION DEMO")
    print("=" * 60)
    
    import jax
    import jax.numpy as jnp
    from src.agent.ppo import PPOAgent
    from src.environment.cartpole import CartPoleWrapper
    from src.training.trainer import PPOTrainer
    
    # Create environment and agent
    print("Creating CartPole environment...")
    env = CartPoleWrapper()
    
    print("Creating PPO agent...")
    key = jax.random.PRNGKey(42)
    agent = PPOAgent(
        observation_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        learning_rate=3e-4,
        clip_epsilon=0.2,
        gamma=0.99,
        gae_lambda=0.95,
        entropy_coef=0.01,
        value_coef=0.5,
        batch_size=64,
        epochs_per_update=10,
        key=key
    )
    
    print("Creating trainer with MLflow integration...")
    trainer = PPOTrainer(
        agent=agent,
        env=env,
        max_episodes=100,
        max_steps_per_episode=500,
        target_reward=195.0,
        convergence_window=20,
        early_stopping_patience=10,
        eval_frequency=10,
        eval_episodes=5,
        log_frequency=5,
        enable_mlflow=True,
        mlflow_experiment_name="demo-cartpole-ppo",
        key=key
    )
    
    print("\nStarting training with MLflow tracking...")
    print("MLflow tracking URI:", os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    print("Experiment name: demo-cartpole-ppo")
    print("\nTraining progress:")
    print("-" * 60)
    
    # Train the agent
    results = trainer.train()
    
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
        print("\n🎉 Agent successfully converged!")
        print("Model has been registered to MLflow Model Registry")
    else:
        print("\n⚠️  Agent did not converge within the training period")
    
    print("\n" + "=" * 60)
    print("MLFLOW INTEGRATION SUMMARY")
    print("=" * 60)
    print("✓ Hyperparameters logged to MLflow")
    print("✓ Training metrics tracked in real-time")
    print("✓ Episode rewards and losses logged")
    print("✓ Evaluation metrics recorded")
    print("✓ Training curves saved as artifacts")
    print("✓ Model checkpoint saved to MLflow")
    if results['converged']:
        print("✓ Model registered to Model Registry")
    print("✓ Dashboard data created for visualization")
    
    print(f"\nTo view the results in MLflow:")
    print(f"1. Start MLflow UI: mlflow ui")
    print(f"2. Open browser: {os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')}")
    print(f"3. Navigate to experiment: demo-cartpole-ppo")
    
    return results

if __name__ == "__main__":
    results = main()