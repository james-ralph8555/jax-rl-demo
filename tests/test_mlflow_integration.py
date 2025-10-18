#!/usr/bin/env python3
"""Test script for MLflow integration with PPO training"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_mlflow_logger_basic():
    """Test basic MLflow logger functionality"""
    print("Testing MLflow Logger Basic Functionality...")
    
    from src.visualization.mlflow_logger import MLflowLogger
    
    # Test logger initialization
    logger = MLflowLogger(experiment_name="test-cartpole-ppo")
    
    # Test starting a run
    run_id = logger.start_run(run_name="test-run")
    print(f"Started MLflow run: {run_id}")
    
    # Test logging parameters
    params = {
        'learning_rate': 0.001,
        'batch_size': 64,
        'epochs': 10
    }
    logger.log_hyperparameters(params)
    print("Logged hyperparameters")
    
    # Test logging metrics
    metrics = {
        'episode_reward': 150.5,
        'policy_loss': 0.123,
        'value_loss': 0.456,
        'total_loss': 0.579
    }
    logger.log_metrics(metrics, step=1)
    print("Logged metrics")
    
    # Test episode data logging
    episode_rewards = [100.0, 120.0, 140.0, 160.0, 180.0]
    episode_lengths = [50, 60, 70, 80, 90]
    logger.log_episode_data(episode_rewards, episode_lengths, step=5)
    print("Logged episode data")
    
    # Test evaluation metrics
    eval_metrics = {
        'avg_reward': 165.0,
        'std_reward': 15.0,
        'max_reward': 200.0,
        'min_reward': 130.0
    }
    logger.log_evaluation_metrics(eval_metrics, step=5)
    print("Logged evaluation metrics")
    
    # Test training curves
    losses = {
        'policy_loss': [0.5, 0.4, 0.3, 0.2, 0.1],
        'value_loss': [0.8, 0.7, 0.6, 0.5, 0.4]
    }
    logger.log_training_curves(episode_rewards, losses)
    print("Logged training curves")
    
    # Test dashboard data
    training_results = {
        'total_episodes': 100,
        'total_steps': 5000,
        'training_time': 120.5,
        'final_avg_reward': 175.0,
        'final_std_reward': 12.0,
        'best_avg_reward': 185.0,
        'converged': True
    }
    logger.create_dashboard_data(training_results)
    print("Created dashboard data")
    
    # End run
    logger.end_run()
    print("Ended MLflow run")
    
    print("✓ MLflow Logger Basic Functionality Test Passed")
    return True

def test_mlflow_integration_with_training():
    """Test MLflow integration with actual PPO training"""
    print("\nTesting MLflow Integration with PPO Training...")
    
    import jax
    import jax.numpy as jnp
    from src.agent.ppo import PPOAgent
    from src.environment.cartpole import CartPoleWrapper
    from src.training.trainer import PPOTrainer
    
    # Create environment and agent
    env = CartPoleWrapper()
    key = jax.random.PRNGKey(42)
    
    # Create agent with smaller parameters for faster testing
    agent = PPOAgent(
        observation_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        learning_rate=3e-4,
        clip_epsilon=0.2,
        gamma=0.99,
        gae_lambda=0.95,
        entropy_coef=0.01,
        value_coef=0.5,
        batch_size=16,  # Smaller batch for faster testing
        epochs_per_update=3,  # Fewer epochs for faster testing
        key=key
    )
    
    # Create trainer with MLflow enabled
    trainer = PPOTrainer(
        agent=agent,
        env=env,
        max_episodes=20,  # Small number for testing
        max_steps_per_episode=200,
        target_reward=150.0,  # Lower target for faster convergence
        convergence_window=10,
        early_stopping_patience=5,
        eval_frequency=5,
        eval_episodes=3,
        log_frequency=2,
        enable_mlflow=True,
        mlflow_experiment_name="test-ppo-integration",
        key=key
    )
    
    print("Starting training with MLflow integration...")
    
    # Train for a few episodes
    results = trainer.train()
    
    # Check results
    assert results['total_episodes'] > 0, "Training should complete at least one episode"
    assert 'episode_rewards' in results, "Results should contain episode rewards"
    assert 'losses' in results, "Results should contain losses"
    
    print(f"Training completed: {results['total_episodes']} episodes")
    print(f"Final reward: {results['final_avg_reward']:.2f}")
    print(f"Converged: {results['converged']}")
    
    print("✓ MLflow Integration with PPO Training Test Passed")
    return True

def test_mlflow_model_registry():
    """Test MLflow model registry functionality"""
    print("\nTesting MLflow Model Registry...")
    
    from src.visualization.mlflow_logger import MLflowLogger
    
    # Create logger
    logger = MLflowLogger(experiment_name="test-model-registry")
    
    # Start a run
    run_id = logger.start_run(run_name="test-model-registry")
    
    # Create a dummy model (using a simple dict as placeholder)
    dummy_model = {
        'network_params': {'dummy': 'params'},
        'model_info': 'test_model'
    }
    
    # Log model
    logger.log_model(dummy_model, "test_model")
    print("Logged model to MLflow")
    
    # Try to register model
    model_uri = f"runs:/{run_id}/test_model"
    model_version = logger.register_model(model_uri, "test-cartpole-model", "Staging")
    
    if model_version:
        print(f"Model registered with version: {model_version}")
    else:
        print("Model registration skipped (MLflow might not be available)")
    
    # End run
    logger.end_run()
    
    print("✓ MLflow Model Registry Test Passed")
    return True

def test_mlflow_error_handling():
    """Test MLflow error handling when MLflow is not available"""
    print("\nTesting MLflow Error Handling...")
    
    # Test that MLflow operations work correctly
    from src.visualization.mlflow_logger import MLflowLogger
    
    # Create logger
    logger = MLflowLogger(experiment_name="test-error-handling")
    
    # All operations should work without exceptions
    run_id = logger.start_run(run_name="test-error")
    
    logger.log_params({'test': 'param'})
    logger.log_metrics({'test': 1.0})
    logger.log_model({'dummy': 'model'})
    logger.log_episode_data([100.0, 150.0], [50, 75], 1)
    logger.log_evaluation_metrics({'avg_reward': 125.0}, 1)
    logger.log_training_curves([100.0, 150.0], {'loss': [0.5, 0.3]})
    logger.create_dashboard_data({'converged': True})
    logger.end_run()
    
    print("✓ MLflow Error Handling Test Passed")
    
    return True

def main():
    """Run all MLflow integration tests"""
    print("=" * 60)
    print("MLFLOW INTEGRATION TESTS")
    print("=" * 60)
    
    tests = [
        test_mlflow_logger_basic,
        test_mlflow_integration_with_training,
        test_mlflow_model_registry,
        test_mlflow_error_handling
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with error: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"MLFLOW INTEGRATION TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("🎉 All MLflow integration tests passed!")
        return 0
    else:
        print("❌ Some MLflow integration tests failed!")
        return 1

if __name__ == "__main__":
    exit(main())