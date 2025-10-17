"""Integration tests for the CartPole environment."""

import pytest
import jax.numpy as jnp
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.environment.cartpole import CartPoleWrapper, create_cartpole_env
from src.environment.utils import (
    preprocess_observation, create_episode_buffer, compute_advantages_and_returns,
    batch_episodes, compute_episode_statistics
)


class TestCartPoleIntegration:
    """Integration tests using real CartPole environment."""
    
    def test_create_cartpole_env(self):
        """Test creating CartPole environment."""
        env = create_cartpole_env(normalize_observations=False)
        
        assert env.observation_dim == 4
        assert env.action_dim == 2
        assert env.normalize_observations is False
        
        env.close()
    
    def test_environment_reset_step(self):
        """Test environment reset and step cycle."""
        env = create_cartpole_env(normalize_observations=False)
        
        # Test reset
        obs, info = env.reset(seed=42)
        assert isinstance(obs, jnp.ndarray)
        assert obs.shape == (4,)
        assert isinstance(info, dict)
        
        # Test step
        action = 0  # Left action
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert isinstance(obs, jnp.ndarray)
        assert obs.shape == (4,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        
        env.close()
    
    def test_episode_collection(self):
        """Test collecting a full episode."""
        env = create_cartpole_env(normalize_observations=False)
        buffer = create_episode_buffer(500)
        
        obs, info = env.reset(seed=42)
        step_count = 0
        
        for i in range(500):
            action = 0  # Always go left for testing
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Store in buffer
            buffer['observations'] = buffer['observations'].at[i].set(obs)
            buffer['actions'] = buffer['actions'].at[i].set(action)
            buffer['rewards'] = buffer['rewards'].at[i].set(reward)
            buffer['dones'] = buffer['dones'].at[i].set(terminated or truncated)
            buffer['log_probs'] = buffer['log_probs'].at[i].set(0.5)  # Dummy value
            buffer['values'] = buffer['values'].at[i].set(0.5)  # Dummy value
            
            obs = next_obs
            step_count += 1
            
            if terminated or truncated:
                break
        
        # Verify episode was collected
        assert step_count > 0
        assert buffer['rewards'][:step_count].sum() > 0
        
        # Compute episode statistics
        stats = compute_episode_statistics({
            key: buffer[key][:step_count] for key in buffer.keys()
        })
        
        assert stats['episode_length'] == step_count
        assert stats['total_reward'] > 0
        
        env.close()
    
    def test_normalization_functionality(self):
        """Test observation normalization."""
        env = create_cartpole_env(normalize_observations=True)
        
        obs, info = env.reset(seed=42)
        
        # Observations should be normalized
        assert jnp.all(jnp.abs(obs) < 10.0)  # Should be reasonably small
        
        # Take a few steps to update running statistics
        for _ in range(10):
            action = 0
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        
        # Check that normalization stats were updated
        mean, std = env.get_observation_stats()
        assert mean.shape == (4,)
        assert std.shape == (4,)
        
        env.close()
    
    def test_batch_processing(self):
        """Test batch processing of multiple episodes."""
        # Create two short episodes
        episodes = []
        
        for episode_idx in range(2):
            env = create_cartpole_env(normalize_observations=False)
            buffer = create_episode_buffer(100)
            
            obs, info = env.reset(seed=42 + episode_idx)
            step_count = 0
            
            for i in range(50):  # Short episodes
                action = episode_idx % 2  # Alternate actions
                next_obs, reward, terminated, truncated, info = env.step(action)
                
                buffer['observations'] = buffer['observations'].at[i].set(obs)
                buffer['actions'] = buffer['actions'].at[i].set(action)
                buffer['rewards'] = buffer['rewards'].at[i].set(reward)
                buffer['dones'] = buffer['dones'].at[i].set(terminated or truncated)
                buffer['log_probs'] = buffer['log_probs'].at[i].set(0.5)
                buffer['values'] = buffer['values'].at[i].set(0.5)
                
                obs = next_obs
                step_count += 1
                
                if terminated or truncated:
                    break
            
            # Create episode dict
            episode = {key: buffer[key][:step_count] for key in buffer.keys()}
            episodes.append(episode)
            
            env.close()
        
        # Test batch processing
        batched = batch_episodes(episodes)
        
        assert batched['observations'].shape[0] == 2  # 2 episodes
        assert batched['actions'].shape[0] == 2
        assert 'lengths' in batched
        assert 'mask' in batched
        
        # Verify lengths are correct
        assert batched['lengths'][0] > 0
        assert batched['lengths'][1] > 0


if __name__ == "__main__":
    pytest.main([__file__])