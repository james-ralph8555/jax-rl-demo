#!/usr/bin/env python3
"""
Demo script to showcase the CartPole environment wrapper functionality.
Run with: nix develop --command python demo_environment.py
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.environment.cartpole import create_cartpole_env
from src.environment.utils import create_episode_buffer, compute_episode_statistics
import jax.numpy as jnp


def demo_basic_functionality():
    """Demonstrate basic environment functionality."""
    print("=== CartPole Environment Demo ===\n")
    
    # Create environment
    env = create_cartpole_env(normalize_observations=True)
    print(f"Environment created:")
    print(f"  - Observation dimension: {env.observation_dim}")
    print(f"  - Action dimension: {env.action_dim}")
    print(f"  - Normalization enabled: {env.normalize_observations}")
    print()
    
    # Reset environment
    obs, info = env.reset(seed=42)
    print(f"Initial observation: {obs}")
    print(f"Observation shape: {obs.shape}")
    print()
    
    # Take a few steps
    print("Taking 10 steps with random actions...")
    total_reward = 0.0
    for step in range(10):
        action = 0 if step < 5 else 1  # Alternate actions
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"  Step {step + 1}: action={action}, reward={reward:.1f}, "
              f"obs=[{', '.join([f'{x:.3f}' for x in obs])}]")
        
        if terminated or truncated:
            print(f"  Episode ended after {step + 1} steps")
            break
    
    print(f"\nTotal reward: {total_reward}")
    print()
    
    # Show normalization statistics
    mean, std = env.get_observation_stats()
    print("Normalization statistics:")
    print(f"  Mean: [{', '.join([f'{x:.3f}' for x in mean])}]")
    print(f"  Std:  [{', '.join([f'{x:.3f}' for x in std])}]")
    print()
    
    env.close()


def demo_episode_collection():
    """Demonstrate episode collection with buffer."""
    print("=== Episode Collection Demo ===\n")
    
    env = create_cartpole_env(normalize_observations=False)
    buffer = create_episode_buffer(500)
    
    # Collect one episode
    print("Collecting one episode...")
    obs, info = env.reset(seed=123)
    step_count = 0
    
    for i in range(500):
        action = 0  # Always go left
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # Store in buffer
        buffer['observations'] = buffer['observations'].at[i].set(obs)
        buffer['actions'] = buffer['actions'].at[i].set(action)
        buffer['rewards'] = buffer['rewards'].at[i].set(reward)
        buffer['dones'] = buffer['dones'].at[i].set(terminated or truncated)
        buffer['log_probs'] = buffer['log_probs'].at[i].set(0.5)  # Dummy
        buffer['values'] = buffer['values'].at[i].set(0.5)  # Dummy
        
        obs = next_obs
        step_count += 1
        
        if terminated or truncated:
            break
    
    # Compute statistics
    episode_data = {key: buffer[key][:step_count] for key in buffer.keys()}
    stats = compute_episode_statistics(episode_data)
    
    print(f"Episode completed:")
    print(f"  - Length: {stats['episode_length']} steps")
    print(f"  - Total reward: {stats['total_reward']:.1f}")
    print(f"  - Mean reward: {stats['mean_reward']:.2f}")
    print(f"  - Terminated early: {stats['terminated_early']}")
    print()
    
    env.close()


def demo_normalization_comparison():
    """Demonstrate effect of normalization."""
    print("=== Normalization Comparison Demo ===\n")
    
    # Create environments with and without normalization
    env_norm = create_cartpole_env(normalize_observations=True)
    env_no_norm = create_cartpole_env(normalize_observations=False)
    
    # Reset both environments with same seed
    obs_norm, _ = env_norm.reset(seed=456)
    obs_no_norm, _ = env_no_norm.reset(seed=456)
    
    print("Observation comparison (same seed):")
    print(f"  Without normalization: [{', '.join([f'{x:.3f}' for x in obs_no_norm])}]")
    print(f"  With normalization:    [{', '.join([f'{x:.3f}' for x in obs_norm])}]")
    print()
    
    # Take a few steps to see how normalization evolves
    print("After 5 steps:")
    for _ in range(5):
        action = 1
        obs_norm, _, _, _, _ = env_norm.step(action)
        obs_no_norm, _, _, _, _ = env_no_norm.step(action)
    
    print(f"  Without normalization: [{', '.join([f'{x:.3f}' for x in obs_no_norm])}]")
    print(f"  With normalization:    [{', '.join([f'{x:.3f}' for x in obs_norm])}]")
    print()
    
    env_norm.close()
    env_no_norm.close()


if __name__ == "__main__":
    demo_basic_functionality()
    demo_episode_collection()
    demo_normalization_comparison()
    print("Demo completed successfully! 🎉")