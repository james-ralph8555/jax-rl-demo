"""CartPole environment wrapper using Gymnasium."""

import jax
import jax.numpy as jnp
import numpy as np
import imageio
from gymnasium import make
from gymnasium.wrappers import RecordEpisodeStatistics
from typing import Tuple, Dict, Any, Optional, List
import os


class CartPoleWrapper:
    """CartPole environment wrapper with preprocessing and normalization."""
    
    def __init__(self, 
                 render_mode: Optional[str] = None,
                 max_episode_steps: int = 500,
                 normalize_observations: bool = True,
                 video_record_freq: Optional[int] = None,
                 video_dir: Optional[str] = None):
        """
        Initialize CartPole environment wrapper.
        
        Args:
            render_mode: Rendering mode ('human', 'rgb_array', or None)
            max_episode_steps: Maximum steps per episode
            normalize_observations: Whether to normalize observations
            video_record_freq: Frequency of GIF recording (None to disable)
            video_dir: Directory to save GIFs (required if video_record_freq is set)
        """
        self.video_record_freq = video_record_freq
        if video_record_freq is not None and video_dir is None:
            raise ValueError("video_dir must be provided when video_record_freq is set")
        self.video_dir = video_dir
        
        # Set render mode to rgb_array if GIF recording is requested
        if video_record_freq is not None and render_mode is None:
            render_mode = 'rgb_array'
        
        # Create base environment
        self.env = make('CartPole-v1', render_mode=render_mode, max_episode_steps=max_episode_steps)
        self.env = RecordEpisodeStatistics(self.env)
        
        self.normalize_observations = normalize_observations
        
        # Observation and action space dimensions
        try:
            self.observation_dim = self.env.observation_space.shape[0]
        except AttributeError:
            self.observation_dim = 4
        
        try:
            self.action_dim = self.env.action_space.n
        except AttributeError:
            self.action_dim = 2
        
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
            # Update running stats with raw observation, then normalize
            self._update_running_stats(obs)
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
        # Ensure terminated and truncated remain Python booleans
        terminated = bool(terminated)
        truncated = bool(truncated)
        
        if self.normalize_observations:
            # Update running statistics with raw observation, then normalize
            self._update_running_stats(obs)
            obs = self._normalize_observation(obs)
            
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
    
    def get_recorded_gifs(self) -> List[str]:
        """
        Get list of recorded GIF files.
        
        Returns:
            List of GIF file paths
        """
        gifs = []
        if self.video_record_freq is not None and self.video_dir and os.path.exists(self.video_dir):
            for file in os.listdir(self.video_dir):
                if file.endswith('.gif'):
                    gifs.append(os.path.join(self.video_dir, file))
        return sorted(gifs)
    
    def record_episode_gif(self, episode_id: int, max_steps: int = 500) -> Optional[str]:
        """
        Record a single episode as GIF.
        
        Args:
            episode_id: Episode identifier
            max_steps: Maximum steps to record
            
        Returns:
            Path to recorded GIF file or None if failed
        """
        if not self.video_dir:
            return None
            
        try:
            # Create a temporary environment for GIF recording
            temp_env = make('CartPole-v1', render_mode='rgb_array', max_episode_steps=max_steps)
            
            frames = []
            obs, info = temp_env.reset()
            frames.append(temp_env.render())
            
            for step in range(max_steps):
                action = temp_env.action_space.sample()  # Random action for demo
                obs, reward, terminated, truncated, info = temp_env.step(action)
                frames.append(temp_env.render())
                if terminated or truncated:
                    break
            
            temp_env.close()
            
            # Save as GIF
            gif_path = os.path.join(self.video_dir, f"eval_episode_{episode_id}.gif")
            imageio.mimsave(gif_path, frames, fps=30)
            return gif_path
            
        except Exception as e:
            print(f"Warning: Could not record episode GIF: {e}")
        
        return None
    
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
