"""CartPole environment wrapper using Gymnasium."""

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import make
from gymnasium.wrappers import RecordEpisodeStatistics
from typing import Tuple, Dict, Any, Optional


class CartPoleWrapper:
    """CartPole environment wrapper with preprocessing and normalization."""
    
    def __init__(self, 
                 render_mode: Optional[str] = None,
                 max_episode_steps: int = 500,
                 normalize_observations: bool = True):
        """
        Initialize CartPole environment wrapper.
        
        Args:
            render_mode: Rendering mode ('human', 'rgb_array', or None)
            max_episode_steps: Maximum steps per episode
            normalize_observations: Whether to normalize observations
        """
        self.env = make('CartPole-v1', render_mode=render_mode, max_episode_steps=max_episode_steps)
        self.env = RecordEpisodeStatistics(self.env)
        self.normalize_observations = normalize_observations
        
        # Observation and action space dimensions
        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        
        # Observation statistics for normalization
        self.obs_mean = jnp.zeros(self.observation_dim)
        self.obs_std = jnp.ones(self.observation_dim)
        
        # Running statistics for online normalization
        self.running_count = 0
        self.running_mean = jnp.zeros(self.observation_dim)
        self.running_var = jnp.ones(self.observation_dim)
        
    def reset(self, seed: Optional[int] = None) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Reset the environment.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (observation, info)
        """
        obs, info = self.env.reset(seed=seed)
        obs = jnp.array(obs, dtype=jnp.float32)
        
        if self.normalize_observations:
            obs = self._normalize_observation(obs)
            
        return obs, info
    
    def step(self, action: int) -> Tuple[jnp.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = jnp.array(obs, dtype=jnp.float32)
        
        if self.normalize_observations:
            obs = self._normalize_observation(obs)
            # Update running statistics
            self._update_running_stats(obs)
            
        return obs, float(reward), terminated, truncated, info
    
    def _normalize_observation(self, obs: jnp.ndarray) -> jnp.ndarray:
        """
        Normalize observation using running statistics.
        
        Args:
            obs: Raw observation
            
        Returns:
            Normalized observation
        """
        return (obs - self.obs_mean) / (self.obs_std + 1e-8)
    
    def _update_running_stats(self, obs: jnp.ndarray) -> None:
        """
        Update running statistics for online normalization.
        
        Args:
            obs: New observation to incorporate
        """
        self.running_count += 1
        
        # Update running mean and variance using Welford's algorithm
        delta = obs - self.running_mean
        self.running_mean += delta / self.running_count
        delta2 = obs - self.running_mean
        self.running_var += delta * delta2
        
        # Update normalization parameters
        if self.running_count > 1:
            self.obs_mean = self.running_mean
            self.obs_std = jnp.sqrt(self.running_var / (self.running_count - 1) + 1e-8)
    
    def set_normalization_stats(self, mean: jnp.ndarray, std: jnp.ndarray) -> None:
        """
        Set normalization statistics manually.
        
        Args:
            mean: Mean values for normalization
            std: Standard deviation values for normalization
        """
        self.obs_mean = mean
        self.obs_std = std + 1e-8
    
    def get_observation_stats(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Get current observation statistics.
        
        Returns:
            Tuple of (mean, std)
        """
        return self.obs_mean, self.obs_std
    
    def close(self) -> None:
        """Close the environment."""
        self.env.close()
    
    @property
    def action_space(self):
        """Get action space."""
        return self.env.action_space
    
    @property
    def observation_space(self):
        """Get observation space."""
        return self.env.observation_space


def create_cartpole_env(**kwargs) -> CartPoleWrapper:
    """
    Factory function to create CartPole environment.
    
    Args:
        **kwargs: Arguments to pass to CartPoleWrapper
        
    Returns:
        CartPoleWrapper instance
    """
    return CartPoleWrapper(**kwargs)


# JAX-compatible functions for batch processing
@jax.jit
def batch_normalize_observations(obs: jnp.ndarray, 
                                mean: jnp.ndarray, 
                                std: jnp.ndarray) -> jnp.ndarray:
    """
    Batch normalize observations.
    
    Args:
        obs: Batch of observations
        mean: Mean values for normalization
        std: Standard deviation values for normalization
        
    Returns:
        Normalized observations
    """
    return (obs - mean[None, :]) / (std[None, :] + 1e-8)


@jax.jit
def compute_episode_returns(rewards: jnp.ndarray, 
                          dones: jnp.ndarray, 
                          gamma: float = 0.99) -> jnp.ndarray:
    """
    Compute discounted returns for an episode.
    
    Args:
        rewards: Array of rewards
        dones: Array of done flags
        gamma: Discount factor
        
    Returns:
        Array of discounted returns
    """
    returns = jnp.zeros_like(rewards)
    running_return = 0.0
    
    def compute_step(carry, idx):
        running_return, returns = carry
        reward = rewards[idx]
        done = dones[idx]
        
        running_return = reward + gamma * running_return * (1.0 - done)
        returns = returns.at[idx].set(running_return)
        
        return (running_return, returns), None
    
    (running_return, returns), _ = jax.lax.scan(
        compute_step, (0.0, returns), jnp.arange(len(rewards) - 1, -1, -1)
    )
    
    return returns