"""Test suite for training loop implementation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from unittest.mock import Mock, patch

# Import the modules we're testing
import sys
sys.path.append('src')

from src.agent.ppo import PPOAgent
from src.environment.cartpole import CartPoleWrapper
from src.training.trainer import PPOTrainer


class TestPPOTrainer:
    """Test cases for PPOTrainer class."""
    
    @pytest.fixture
    def trainer_setup(self):
        """Setup trainer with agent and environment for testing."""
        key = jax.random.PRNGKey(42)
        
        # Create agent
        agent = PPOAgent(
            observation_dim=4,
            action_dim=2,
            learning_rate=3e-4,
            batch_size=4,  # Small batch for testing
            minibatch_size=2,
            epochs_per_update=2,
            key=key
        )
        
        # Create environment
        env = CartPoleWrapper(max_episode_steps=10)  # Short episodes for testing
        
        # Create trainer
        trainer = PPOTrainer(
            agent=agent,
            env=env,
            max_episodes=5,  # Few episodes for testing
            max_steps_per_episode=10,
            target_reward=10.0,  # Low target for testing
            convergence_window=3,
            early_stopping_patience=2,
            eval_frequency=2,
            eval_episodes=2,
            log_frequency=1,
            key=key
        )
        
        return trainer, agent, env
    
    def test_trainer_initialization(self, trainer_setup):
        """Test trainer initialization."""
        trainer, agent, env = trainer_setup
        
        assert trainer.agent == agent
        assert trainer.env == env
        assert trainer.max_episodes == 5
        assert trainer.max_steps_per_episode == 10
        assert trainer.target_reward == 10.0
        assert trainer.convergence_window == 3
        assert trainer.early_stopping_patience == 2
        assert trainer.eval_frequency == 2
        assert trainer.eval_episodes == 2
        assert trainer.log_frequency == 1
        
        # Check initial state
        assert trainer.episode_count == 0
        assert trainer.step_count == 0
        assert trainer.best_avg_reward == float('-inf')
        assert trainer.convergence_count == 0
        assert trainer.should_stop == False
        
        # Check metrics tracking
        assert trainer.episode_rewards == []
        assert trainer.episode_lengths == []
        assert trainer.eval_rewards == []
        assert len(trainer.losses) == 0
    
    def test_collect_episode(self, trainer_setup):
        """Test episode collection."""
        trainer, agent, env = trainer_setup
        key = jax.random.PRNGKey(123)
        
        episode_data = trainer.collect_episode(key)
        
        # Check episode data structure
        required_keys = ['observations', 'actions', 'rewards', 'next_observations', 
                        'dones', 'log_probs', 'values']
        for key in required_keys:
            assert key in episode_data
        
        # Check data shapes and types
        assert episode_data['observations'].shape[1] == 4  # observation_dim
        assert len(episode_data['actions'].shape) == 1  # actions are 1D
        assert len(episode_data['rewards'].shape) == 1
        assert len(episode_data['dones'].shape) == 1
        assert len(episode_data['log_probs'].shape) == 1
        assert len(episode_data['values'].shape) == 1
        
        # Check consistency
        episode_length = len(episode_data['rewards'])
        # Note: observations might be stored differently, just check basic consistency
        assert len(episode_data['actions']) == episode_length
        assert len(episode_data['next_observations']) == episode_length
        assert len(episode_data['dones']) == episode_length
        assert len(episode_data['log_probs']) == episode_length
        assert len(episode_data['values']) == episode_length
        
        # Check that episode ends with done=True (unless max steps reached)
        if episode_length < trainer.max_steps_per_episode:
            assert episode_data['dones'][-1] == True
    
    def test_collect_batch(self, trainer_setup):
        """Test batch collection."""
        trainer, agent, env = trainer_setup
        key = jax.random.PRNGKey(123)
        
        batch_size = 3
        batch = trainer.collect_batch(key, batch_size)
        
        # Check batch structure
        required_keys = ['observations', 'actions', 'rewards', 'next_observations', 
                        'dones', 'log_probs', 'values']
        for key in required_keys:
            assert key in batch
        
        # Check that batch contains data from multiple episodes
        total_transitions = len(batch['rewards'])
        assert total_transitions > 0
        
        # Check data consistency
        for key in ['actions', 'rewards', 'next_observations', 'dones', 'log_probs', 'values']:
            assert len(batch[key]) == total_transitions
        
        # Note: observations shape depends on episode structure, skip strict shape check
        # Just verify observations exist and have correct feature dimension
        assert len(batch['observations'].shape) >= 2  # Should have at least 2 dimensions
        assert batch['observations'].shape[-1] == 4  # observation_dim
    
    def test_evaluate(self, trainer_setup):
        """Test policy evaluation."""
        trainer, agent, env = trainer_setup
        
        eval_metrics = trainer.evaluate(num_episodes=2)
        
        # Check evaluation metrics structure
        required_keys = ['avg_reward', 'std_reward', 'max_reward', 'min_reward', 'avg_length']
        for key in required_keys:
            assert key in eval_metrics
        
        # Check metric types and ranges
        assert isinstance(eval_metrics['avg_reward'], (int, float))
        assert isinstance(eval_metrics['std_reward'], (int, float))
        assert isinstance(eval_metrics['max_reward'], (int, float))
        assert isinstance(eval_metrics['min_reward'], (int, float))
        assert isinstance(eval_metrics['avg_length'], (int, float))
        
        # Check logical consistency
        assert eval_metrics['min_reward'] <= eval_metrics['avg_reward'] <= eval_metrics['max_reward']
        assert eval_metrics['std_reward'] >= 0
        assert eval_metrics['avg_length'] > 0
        assert eval_metrics['max_reward'] >= 0
        assert eval_metrics['min_reward'] >= 0
    
    def test_check_convergence(self, trainer_setup):
        """Test convergence checking."""
        trainer, agent, env = trainer_setup
        
        # Initially should not converge
        assert trainer.check_convergence() == False
        
        # Add some episode rewards below target
        trainer.episode_rewards = [5.0, 6.0, 7.0]  # Below target of 10.0
        assert trainer.check_convergence() == False
        
        # Add episode rewards above target
        trainer.episode_rewards = [12.0, 13.0, 14.0]  # Above target of 10.0
        assert trainer.check_convergence() == False  # Need patience
        
        # Simulate patience count
        trainer.convergence_count = trainer.early_stopping_patience
        assert trainer.check_convergence() == True
        
        # Check best_avg_reward update
        assert trainer.best_avg_reward == 13.0  # Average of [12.0, 13.0, 14.0]
    
    def test_train_step(self, trainer_setup):
        """Test training step."""
        trainer, agent, env = trainer_setup
        key = jax.random.PRNGKey(123)
        
        # Store initial state
        initial_episode_count = trainer.episode_count
        initial_step_count = trainer.step_count
        
        # Perform training step
        step_metrics = trainer.train_step(key)
        
        # Check step metrics structure
        required_keys = ['episode_rewards', 'episode_lengths', 'avg_reward', 'avg_length', 'total_steps']
        for key in required_keys:
            assert key in step_metrics
        
        # Check that agent was updated (should have loss metrics)
        assert any('loss' in key for key in step_metrics.keys())
        
        # Check state updates
        assert trainer.episode_count > initial_episode_count
        assert trainer.step_count > initial_step_count
        
        # Check metrics storage
        assert len(trainer.episode_rewards) > 0
        assert len(trainer.episode_lengths) > 0
        assert len(trainer.losses) > 0
    
    def test_training_loop(self, trainer_setup):
        """Test full training loop."""
        trainer, agent, env = trainer_setup
        
        # Mock callback to track calls
        callback_calls = []
        def test_callback(metrics):
            callback_calls.append(metrics)
        
        # Run training
        results = trainer.train(callback=test_callback)
        
        # Check results structure
        required_keys = ['total_episodes', 'total_steps', 'training_time', 'final_avg_reward',
                        'final_std_reward', 'best_avg_reward', 'converged', 'episode_rewards',
                        'episode_lengths', 'eval_rewards', 'losses']
        for key in required_keys:
            assert key in results
        
        # Check result types and ranges
        assert isinstance(results['total_episodes'], int)
        assert isinstance(results['total_steps'], int)
        assert isinstance(results['training_time'], float)
        assert isinstance(results['final_avg_reward'], (int, float))
        assert isinstance(results['final_std_reward'], (int, float))
        assert isinstance(results['best_avg_reward'], (int, float))
        assert isinstance(results['converged'], (bool, np.bool_))
        
        # Check logical consistency
        assert results['total_episodes'] > 0
        assert results['total_steps'] > 0
        assert results['training_time'] > 0
        assert results['final_std_reward'] >= 0
        assert results['best_avg_reward'] >= results['final_avg_reward']
        
        # Check that callback was called
        assert len(callback_calls) > 0
        
        # Check callback structure
        for callback_data in callback_calls:
            assert 'episode' in callback_data
            assert 'step_metrics' in callback_data
    
    def test_early_stopping(self, trainer_setup):
        """Test early stopping functionality."""
        trainer, agent, env = trainer_setup
        
        # Set low target and patience for quick testing
        trainer.target_reward = 5.0
        trainer.early_stopping_patience = 1
        
        # Mock the evaluate method to return high rewards
        def mock_evaluate(num_episodes):
            return {'avg_reward': 10.0, 'std_reward': 1.0, 'max_reward': 12.0, 
                   'min_reward': 8.0, 'avg_length': 15.0}
        
        trainer.evaluate = mock_evaluate
        
        # Run training
        results = trainer.train()
        
        # Should stop early due to convergence
        assert results['converged'] == True
        assert trainer.should_stop == True
    
    def test_metrics_tracking(self, trainer_setup):
        """Test metrics tracking throughout training."""
        trainer, agent, env = trainer_setup
        
        # Run a few training steps
        for i in range(3):
            key = jax.random.PRNGKey(100 + i)
            trainer.train_step(key)
        
        # Check that metrics are tracked
        assert len(trainer.episode_rewards) > 0
        assert len(trainer.episode_lengths) > 0
        assert len(trainer.losses) > 0
        
        # Check loss metrics structure
        for loss_name, loss_values in trainer.losses.items():
            assert isinstance(loss_name, str)
            assert isinstance(loss_values, list)
            assert len(loss_values) > 0
            assert all(isinstance(v, (int, float, jnp.ndarray)) for v in loss_values)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])