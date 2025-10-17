"""Tests for CartPole environment wrapper and utilities."""

import pytest
import numpy as np
import jax.numpy as jnp
from unittest.mock import Mock, patch

# Import the modules we're testing
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.environment.cartpole import CartPoleWrapper, create_cartpole_env, batch_normalize_observations, compute_episode_returns
from src.environment.utils import (
    preprocess_observation, create_episode_buffer, reset_episode_buffer,
    compute_advantages_and_returns, normalize_advantages, batch_episodes,
    validate_episode_data, get_cartpole_observation_bounds, compute_episode_statistics
)


class TestCartPoleWrapper:
    """Test cases for CartPoleWrapper class."""
    
    @patch('src.environment.cartpole.make')
    def test_init(self, mock_make):
        """Test CartPoleWrapper initialization."""
        mock_env = Mock()
        mock_env.observation_space.shape = (4,)
        mock_env.action_space.n = 2
        mock_make.return_value = mock_env
        
        wrapper = CartPoleWrapper()
        
        assert wrapper.observation_dim == 4
        assert wrapper.action_dim == 2
        assert wrapper.normalize_observations is True
        mock_make.assert_called_once_with('CartPole-v1', render_mode=None, max_episode_steps=500)
    
    @patch('src.environment.cartpole.make')
    def test_reset(self, mock_make):
        """Test environment reset."""
        mock_env = Mock()
        mock_env.observation_space.shape = (4,)
        mock_env.action_space.n = 2
        mock_env.reset.return_value = (np.array([0.1, 0.2, 0.3, 0.4]), {})
        mock_make.return_value = mock_env
        
        wrapper = CartPoleWrapper(normalize_observations=False)
        obs, info = wrapper.reset()
        
        assert isinstance(obs, jnp.ndarray)
        assert obs.shape == (4,)
        assert np.allclose(obs, jnp.array([0.1, 0.2, 0.3, 0.4]))
        assert info == {}
    
    @patch('src.environment.cartpole.make')
    def test_step(self, mock_make):
        """Test environment step."""
        mock_env = Mock()
        mock_env.observation_space.shape = (4,)
        mock_env.action_space.n = 2
        mock_env.step.return_value = (np.array([0.1, 0.2, 0.3, 0.4]), 1.0, False, False, {})
        mock_make.return_value = mock_env
        
        wrapper = CartPoleWrapper(normalize_observations=False)
        obs, reward, terminated, truncated, info = wrapper.step(0)
        
        assert isinstance(obs, jnp.ndarray)
        assert obs.shape == (4,)
        assert reward == 1.0
        assert terminated is False
        assert truncated is False
        assert info == {}
    
    @patch('src.environment.cartpole.make')
    def test_normalization_stats(self, mock_make):
        """Test normalization statistics management."""
        mock_env = Mock()
        mock_env.observation_space.shape = (4,)
        mock_env.action_space.n = 2
        mock_make.return_value = mock_env
        
        wrapper = CartPoleWrapper()
        
        # Test setting stats
        mean = jnp.array([0.0, 0.0, 0.0, 0.0])
        std = jnp.array([1.0, 1.0, 1.0, 1.0])
        wrapper.set_normalization_stats(mean, std)
        
        retrieved_mean, retrieved_std = wrapper.get_observation_stats()
        assert jnp.allclose(retrieved_mean, mean)
        assert jnp.allclose(retrieved_std, std + 1e-8)


class TestEnvironmentUtilities:
    """Test cases for environment utility functions."""
    
    def test_preprocess_observation(self):
        """Test observation preprocessing."""
        obs = jnp.array([1.0, 2.0, 0.1, 3.0])
        
        # Test without normalization
        processed = preprocess_observation(obs, normalize=False)
        assert jnp.allclose(processed, obs.astype(jnp.float32))
        
        # Test with normalization
        processed = preprocess_observation(obs, normalize=True)
        expected = obs / jnp.array([4.8, 10.0, 0.42, 10.0])
        assert jnp.allclose(processed, expected)
        
        # Test with clipping
        processed = preprocess_observation(obs, normalize=False, clip_range=(-1.0, 1.0))
        assert jnp.all(processed <= 1.0)
        assert jnp.all(processed >= -1.0)
    
    def test_create_episode_buffer(self):
        """Test episode buffer creation."""
        buffer = create_episode_buffer(100)
        
        assert 'observations' in buffer
        assert 'actions' in buffer
        assert 'rewards' in buffer
        assert 'dones' in buffer
        assert 'log_probs' in buffer
        assert 'values' in buffer
        
        assert buffer['observations'].shape == (100, 4)
        assert buffer['actions'].shape == (100,)
        assert buffer['rewards'].shape == (100,)
        assert buffer['dones'].shape == (100,)
        assert buffer['log_probs'].shape == (100,)
        assert buffer['values'].shape == (100,)
    
    def test_reset_episode_buffer(self):
        """Test episode buffer reset."""
        buffer = create_episode_buffer(10)
        # Modify some values
        buffer['rewards'] = buffer['rewards'].at[0].set(1.0)
        buffer['actions'] = buffer['actions'].at[0].set(1)
        
        reset_buffer = reset_episode_buffer(buffer)
        
        assert jnp.allclose(reset_buffer['rewards'], jnp.zeros(10))
        assert jnp.allclose(reset_buffer['actions'], jnp.zeros(10, dtype=jnp.int32))
    
    def test_compute_advantages_and_returns(self):
        """Test GAE and returns computation."""
        rewards = jnp.array([1.0, 1.0, 1.0])
        values = jnp.array([0.5, 0.5, 0.5])
        dones = jnp.array([False, False, True])
        
        advantages, returns = compute_advantages_and_returns(rewards, values, dones)
        
        assert advantages.shape == (3,)
        assert returns.shape == (3,)
        assert not jnp.any(jnp.isnan(advantages))
        assert not jnp.any(jnp.isnan(returns))
    
    def test_normalize_advantages(self):
        """Test advantage normalization."""
        advantages = jnp.array([1.0, 2.0, 3.0])
        normalized = normalize_advantages(advantages)
        
        assert jnp.allclose(jnp.mean(normalized), 0.0, atol=1e-6)
        assert jnp.allclose(jnp.std(normalized), 1.0, atol=1e-6)
    
    def test_batch_episodes(self):
        """Test episode batching."""
        episodes = [
            {
                'observations': jnp.array([[1.0, 2.0, 3.0, 4.0]]),
                'actions': jnp.array([0]),
                'rewards': jnp.array([1.0]),
                'dones': jnp.array([True]),
                'log_probs': jnp.array([0.5]),
                'values': jnp.array([0.5]),
            },
            {
                'observations': jnp.array([[2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0]]),
                'actions': jnp.array([1, 0]),
                'rewards': jnp.array([1.0, 1.0]),
                'dones': jnp.array([False, True]),
                'log_probs': jnp.array([0.3, 0.7]),
                'values': jnp.array([0.3, 0.7]),
            }
        ]
        
        batched = batch_episodes(episodes)
        
        assert batched['observations'].shape == (2, 2, 4)
        assert batched['actions'].shape == (2, 2)
        assert batched['rewards'].shape == (2, 2)
        assert 'lengths' in batched
        assert 'mask' in batched
        assert jnp.array_equal(batched['lengths'], jnp.array([1, 2]))
    
    def test_validate_episode_data(self):
        """Test episode data validation."""
        valid_episode = {
            'observations': jnp.array([[1.0, 2.0, 3.0, 4.0]]),
            'actions': jnp.array([0]),
            'rewards': jnp.array([1.0]),
            'dones': jnp.array([True]),
            'log_probs': jnp.array([0.5]),
            'values': jnp.array([0.5]),
        }
        
        assert validate_episode_data(valid_episode) is True
        
        # Test invalid episode (missing key)
        invalid_episode = {
            'observations': jnp.array([[1.0, 2.0, 3.0, 4.0]]),
            'actions': jnp.array([0]),
            'rewards': jnp.array([1.0]),
        }
        
        assert validate_episode_data(invalid_episode) is False
        
        # Test invalid episode (wrong shape)
        invalid_episode = {
            'observations': jnp.array([[1.0, 2.0, 3.0]]),  # Wrong shape
            'actions': jnp.array([0]),
            'rewards': jnp.array([1.0]),
            'dones': jnp.array([True]),
            'log_probs': jnp.array([0.5]),
            'values': jnp.array([0.5]),
        }
        
        assert validate_episode_data(invalid_episode) is False
    
    def test_get_cartpole_observation_bounds(self):
        """Test CartPole observation bounds."""
        lower, upper = get_cartpole_observation_bounds()
        
        assert lower.shape == (4,)
        assert upper.shape == (4,)
        assert jnp.all(lower < upper)
        assert lower[0] == -4.8  # cart position
        assert upper[0] == 4.8
        assert lower[2] == -0.42  # pole angle
        assert upper[2] == 0.42
    
    def test_compute_episode_statistics(self):
        """Test episode statistics computation."""
        episode = {
            'observations': jnp.array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]),
            'actions': jnp.array([0, 1]),
            'rewards': jnp.array([1.0, 1.0]),
            'dones': jnp.array([False, True]),
            'log_probs': jnp.array([0.5, 0.5]),
            'values': jnp.array([0.5, 0.5]),
        }
        
        stats = compute_episode_statistics(episode)
        
        assert stats['episode_length'] == 2.0
        assert stats['total_reward'] == 2.0
        assert stats['mean_reward'] == 1.0
        assert stats['terminated_early'] is True


class TestJAXFunctions:
    """Test JAX-compiled functions."""
    
    def test_batch_normalize_observations(self):
        """Test batch observation normalization."""
        obs = jnp.array([[1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 2.0, 3.0]])
        mean = jnp.array([0.5, 1.5, 2.5, 3.5])
        std = jnp.array([0.5, 0.5, 0.5, 0.5])
        
        normalized = batch_normalize_observations(obs, mean, std)
        
        expected = (obs - mean[None, :]) / (std[None, :] + 1e-8)
        assert jnp.allclose(normalized, expected)
    
    def test_compute_episode_returns(self):
        """Test episode returns computation."""
        rewards = jnp.array([1.0, 1.0, 1.0])
        dones = jnp.array([False, False, True])
        
        returns = compute_episode_returns(rewards, dones, gamma=0.99)
        
        assert returns.shape == (3,)
        # Last return should be just the last reward
        assert jnp.allclose(returns[-1], 1.0)
        # Earlier returns should include discounted future rewards
        assert returns[0] > returns[1] > returns[2]


if __name__ == "__main__":
    pytest.main([__file__])