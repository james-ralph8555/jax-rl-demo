"""Environment utilities for CartPole PPO."""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Dict, Any, List, Optional


def preprocess_observation(obs: jnp.ndarray, 
                          normalize: bool = True,
                          clip_range: Optional[Tuple[float, float]] = None) -> jnp.ndarray:
    """
    Preprocess observation with optional normalization and clipping.
    
    Args:
        obs: Raw observation
        normalize: Whether to normalize observation
        clip_range: Optional range to clip values to
        
    Returns:
        Preprocessed observation
    """
    processed_obs = obs.astype(jnp.float32)
    
    if normalize:
        # Normalize to zero mean and unit variance
        # These are typical CartPole observation ranges
        obs_ranges = jnp.array([4.8, 10.0, 0.42, 10.0])  # cart_pos, cart_vel, pole_angle, pole_vel
        processed_obs = processed_obs / obs_ranges
    
    if clip_range is not None:
        processed_obs = jnp.clip(processed_obs, clip_range[0], clip_range[1])
    
    return processed_obs


def create_episode_buffer(max_episode_length: int = 500) -> Dict[str, jnp.ndarray]:
    """
    Create buffer for storing episode data.
    
    Args:
        max_episode_length: Maximum length of an episode
        
    Returns:
        Dictionary with initialized arrays for episode data
    """
    return {
        'observations': jnp.zeros((max_episode_length, 4)),  # CartPole has 4D observations
        'actions': jnp.zeros(max_episode_length, dtype=jnp.int32),
        'rewards': jnp.zeros(max_episode_length),
        'dones': jnp.zeros(max_episode_length, dtype=jnp.bool_),
        'log_probs': jnp.zeros(max_episode_length),
        'values': jnp.zeros(max_episode_length),
    }


def reset_episode_buffer(buffer: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
    """
    Reset episode buffer to zeros.
    
    Args:
        buffer: Episode buffer to reset
        
    Returns:
        Reset buffer
    """
    return {key: jnp.zeros_like(value) for key, value in buffer.items()}


def compute_advantages_and_returns(rewards: jnp.ndarray,
                                 values: jnp.ndarray,
                                 dones: jnp.ndarray,
                                 gamma: float = 0.99,
                                 gae_lambda: float = 0.95) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute Generalized Advantage Estimation (GAE) and returns.
    
    Args:
        rewards: Array of rewards
        values: Array of value predictions
        dones: Array of done flags
        gamma: Discount factor
        gae_lambda: GAE parameter
        
    Returns:
        Tuple of (advantages, returns)
    """
    # Add next value (0 for terminal states)
    next_values = jnp.concatenate([values[1:], jnp.array([0.0])])
    
    # Compute TD errors
    td_errors = rewards + gamma * next_values * (1.0 - dones) - values
    
    # Compute GAE advantages
    advantages = jnp.zeros_like(rewards)
    running_advantage = 0.0
    
    def compute_gae_step(carry, idx):
        running_advantage, advantages = carry
        td_error = td_errors[idx]
        done = dones[idx]
        
        running_advantage = td_error + gamma * gae_lambda * running_advantage * (1.0 - done)
        advantages = advantages.at[idx].set(running_advantage)
        
        return (running_advantage, advantages), None
    
    (running_advantage, advantages), _ = jax.lax.scan(
        compute_gae_step, (0.0, advantages), jnp.arange(len(rewards) - 1, -1, -1)
    )
    
    # Compute returns
    returns = advantages + values
    
    return advantages, returns


def normalize_advantages(advantages: jnp.ndarray) -> jnp.ndarray:
    """
    Normalize advantages to zero mean and unit variance.
    
    Args:
        advantages: Array of advantages
        
    Returns:
        Normalized advantages
    """
    mean = jnp.mean(advantages)
    std = jnp.std(advantages) + 1e-8
    return (advantages - mean) / std


def batch_episodes(episodes: List[Dict[str, jnp.ndarray]]) -> Dict[str, jnp.ndarray]:
    """
    Batch multiple episodes together.
    
    Args:
        episodes: List of episode dictionaries
        
    Returns:
        Batched episode data
    """
    # Filter out empty episodes
    valid_episodes = [ep for ep in episodes if len(ep['rewards']) > 0]
    
    if not valid_episodes:
        return {}
    
    # Find actual lengths of each episode
    lengths = [len(ep['rewards']) for ep in valid_episodes]
    max_length = max(lengths)
    batch_size = len(valid_episodes)
    
    # Create batched arrays
    batched = {}
    for key in valid_episodes[0].keys():
        if key == 'actions':
            batched[key] = jnp.full((batch_size, max_length), -1, dtype=jnp.int32)
        elif key == 'observations':
            batched[key] = jnp.zeros((batch_size, max_length, 4))
        else:
            batched[key] = jnp.zeros((batch_size, max_length))
    
    # Fill batched arrays
    for i, episode in enumerate(valid_episodes):
        length = lengths[i]
        for key, value in episode.items():
            if len(value) > 0:
                if key == 'observations':
                    # Handle 2D observations
                    batched[key] = batched[key].at[i, :length].set(value)
                else:
                    # Handle 1D arrays
                    batched[key] = batched[key].at[i, :length].set(value)
    
    # Add episode lengths and masks
    batched['lengths'] = jnp.array(lengths)
    batched['mask'] = jnp.arange(max_length) < jnp.array(lengths)[:, None]
    
    return batched


def validate_episode_data(episode: Dict[str, jnp.ndarray]) -> bool:
    """
    Validate episode data consistency.
    
    Args:
        episode: Episode dictionary to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_keys = ['observations', 'actions', 'rewards', 'dones', 'log_probs', 'values']
    
    # Check all required keys are present
    if not all(key in episode for key in required_keys):
        return False
    
    # Check all arrays have the same length (except observations which might be 2D)
    length = len(episode['rewards'])
    for key in ['actions', 'rewards', 'dones', 'log_probs', 'values']:
        if len(episode[key]) != length:
            return False
    
    # Check observations have correct shape
    if episode['observations'].shape[0] != length or episode['observations'].shape[1] != 4:
        return False
    
    return True


def get_cartpole_observation_bounds() -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Get typical observation bounds for CartPole environment.
    
    Returns:
        Tuple of (lower_bounds, upper_bounds)
    """
    lower_bounds = jnp.array([-4.8, -10.0, -0.42, -10.0])
    upper_bounds = jnp.array([4.8, 10.0, 0.42, 10.0])
    return lower_bounds, upper_bounds


def compute_episode_statistics(episode: Dict[str, jnp.ndarray]) -> Dict[str, float]:
    """
    Compute statistics for an episode.
    
    Args:
        episode: Episode dictionary
        
    Returns:
        Dictionary with episode statistics
    """
    length = len(episode['rewards'])
    total_reward = jnp.sum(episode['rewards'])
    mean_reward = jnp.mean(episode['rewards']) if length > 0 else 0.0
    
    # Check if episode terminated successfully (reached max length)
    terminated_early = jnp.any(episode['dones'])
    
    return {
        'episode_length': float(length),
        'total_reward': float(total_reward),
        'mean_reward': float(mean_reward),
        'terminated_early': bool(terminated_early),
    }