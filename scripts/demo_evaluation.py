#!/usr/bin/env python3
"""
Demo script showing how to use the evaluation script.

This script demonstrates how to evaluate a trained model using the evaluation script.
It can be used as a reference for running evaluations with different configurations.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.agent.ppo import PPOAgent
from src.environment.cartpole import CartPoleWrapper
from src.training.trainer import PPOTrainer
import jax


def train_demo_model():
    """Train a simple demo model for evaluation purposes."""
    print("Training demo model for evaluation...")
    
    # Create environment
    env = CartPoleWrapper(max_episode_steps=500)
    
    # Create agent
    key = jax.random.PRNGKey(42)
    agent = PPOAgent(
        observation_dim=4,
        action_dim=2,
        learning_rate=3e-4,
        key=key
    )
    
    # Create trainer
    trainer = PPOTrainer(
        agent=agent,
        env=env,
        max_episodes=50,
        eval_frequency=25,
        enable_mlflow=False  # Disable MLflow for demo
    )
    
    # Train model
    results = trainer.train()
    
    # Save model checkpoint
    checkpoint_path = "demo_model_checkpoint.pkl"
    import pickle
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(agent.network_params, f)
    
    print(f"Demo model saved to: {checkpoint_path}")
    print(f"Final reward: {results['final_avg_reward']:.2f}")
    
    env.close()
    return checkpoint_path


def run_evaluation_examples():
    """Run examples of evaluation script usage."""
    
    # Train a demo model first
    checkpoint_path = train_demo_model()
    
    print("\n" + "="*60)
    print("EVALUATION EXAMPLES")
    print("="*60)
    
    # Example 1: Basic evaluation
    print("\n1. Basic evaluation (10 episodes):")
    print("Command: python scripts/evaluate_model.py --checkpoint demo_model_checkpoint.pkl --episodes 10")
    
    result = subprocess.run([
        sys.executable, "scripts/evaluate_model.py",
        "--checkpoint", checkpoint_path,
        "--episodes", "10",
        "--output", "basic_evaluation.json"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ Basic evaluation completed successfully")
        print("Output saved to: basic_evaluation.json")
    else:
        print("✗ Basic evaluation failed")
        print("Error:", result.stderr)
    
    # Example 2: Evaluation with video recording
    print("\n2. Evaluation with video recording (5 episodes):")
    print("Command: python scripts/evaluate_model.py --checkpoint demo_model_checkpoint.pkl --episodes 5 --record-video --video-dir demo_videos")
    
    result = subprocess.run([
        sys.executable, "scripts/evaluate_model.py",
        "--checkpoint", checkpoint_path,
        "--episodes", "5",
        "--record-video",
        "--video-dir", "demo_videos",
        "--output", "video_evaluation.json"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ Video evaluation completed successfully")
        print("Videos saved to: demo_videos/")
        print("Output saved to: video_evaluation.json")
    else:
        print("✗ Video evaluation failed")
        print("Error:", result.stderr)
    
    # Example 3: Show help
    print("\n3. Available evaluation options:")
    print("Command: python scripts/evaluate_model.py --help")
    
    result = subprocess.run([
        sys.executable, "scripts/evaluate_model.py",
        "--help"
    ], capture_output=True, text=True)
    
    print(result.stdout)
    
    # Cleanup
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"\nCleaned up demo model: {checkpoint_path}")


def print_usage_instructions():
    """Print detailed usage instructions for the evaluation script."""
    
    print("\n" + "="*60)
    print("EVALUATION SCRIPT USAGE INSTRUCTIONS")
    print("="*60)
    
    print("""
The evaluation script (scripts/evaluate_model.py) can be used to evaluate
trained PPO models on the CartPole environment.

BASIC USAGE:
  python scripts/evaluate_model.py --checkpoint <path_to_checkpoint.pkl>
  python scripts/evaluate_model.py --model-uri <mlflow_model_uri>

REQUIRED ARGUMENTS:
  --checkpoint PATH    Path to local model checkpoint file (.pkl)
  --model-uri URI      MLflow model URI (e.g., "runs:/<run_id>/final_model")
  
  Note: Exactly one of --checkpoint or --model-uri must be specified.

OPTIONAL ARGUMENTS:
  --episodes N         Number of evaluation episodes (default: 100)
  --max-steps N        Maximum steps per episode (default: 500)
  --render             Render environment during evaluation
  --video-dir DIR      Directory to save videos (default: evaluation_videos)
  --output FILE        Output file for evaluation report (default: evaluation_report.json)
  --experiment NAME    MLflow experiment name (default: cartpole-ppo)

EXAMPLES:

1. Evaluate a local checkpoint:
   python scripts/evaluate_model.py --checkpoint models/best_model.pkl --episodes 50

2. Evaluate with custom video directory:
   python scripts/evaluate_model.py --checkpoint models/best_model.pkl --episodes 10 --video-dir videos/

3. Evaluate an MLflow model:
   python scripts/evaluate_model.py --model-uri "runs:/1234567890abcdef/final_model" --episodes 100

4. Render evaluation in real-time:
   python scripts/evaluate_model.py --checkpoint models/best_model.pkl --episodes 5 --render

OUTPUT:
The script generates:
- Console output with evaluation metrics
- JSON report with detailed statistics
- Video recording of policy behavior (always saved)
- Success rate and convergence information

METRICS REPORTED:
- Mean/Std/Min/Max episode rewards
- Episode length statistics
- Success rate (episodes with reward ≥ 195)
- Convergence status
- Per-episode data for detailed analysis
""")


if __name__ == "__main__":
    print("CartPole PPO Evaluation Demo")
    print("="*40)
    
    # Check if user wants to run demos or just see instructions
    if len(sys.argv) > 1 and sys.argv[1] == "--instructions":
        print_usage_instructions()
    else:
        print("This demo will:")
        print("1. Train a simple model for demonstration")
        print("2. Run evaluation examples")
        print("3. Show usage instructions")
        print()
        
        response = input("Continue? (y/n): ")
        if response.lower() == 'y':
            run_evaluation_examples()
            print_usage_instructions()
        else:
            print_usage_instructions()