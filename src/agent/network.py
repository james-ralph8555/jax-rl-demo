import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple, Optional


class ActorNetwork(nn.Module):
    """Policy network (actor) for PPO agent."""
    
    action_dim: int
    hidden_dims: Tuple[int, ...] = (64, 64)
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Forward pass through hidden layers
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(features=hidden_dim)(x)
            x = nn.tanh(x)
        
        # Output layer for action logits
        action_logits = nn.Dense(features=self.action_dim)(x)
        
        return action_logits


class CriticNetwork(nn.Module):
    """Value network (critic) for PPO agent."""
    
    hidden_dims: Tuple[int, ...] = (64, 64)
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Forward pass through hidden layers
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(features=hidden_dim)(x)
            x = nn.tanh(x)
        
        # Output layer for value
        value = nn.Dense(features=1)(x)
        
        return value


class PPONetwork(nn.Module):
    """Combined actor-critic network for PPO agent."""
    
    action_dim: int
    hidden_dims: Tuple[int, ...] = (64, 64)
    
    def setup(self):
        self.actor = ActorNetwork(action_dim=self.action_dim, hidden_dims=self.hidden_dims)
        self.critic = CriticNetwork(hidden_dims=self.hidden_dims)
    
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass returning both action logits and state value."""
        action_logits = self.actor(x)
        value = self.critic(x)
        return action_logits, value
    
    def actor_forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through actor only."""
        return self.actor(x)
    
    def critic_forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through critic only."""
        return self.critic(x)


def create_networks(
    observation_dim: int,
    action_dim: int,
    hidden_dims: Tuple[int, ...] = (64, 64),
    key: Optional[jax.random.PRNGKey] = None
) -> Tuple[PPONetwork, dict]:
    """
    Create and initialize PPO networks.
    
    Args:
        observation_dim: Dimension of the observation space
        action_dim: Dimension of the action space
        hidden_dims: Tuple of hidden layer dimensions
        key: PRNG key for initialization
        
    Returns:
        Tuple of (network, network_params)
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    
    # Create network
    network = PPONetwork(action_dim=action_dim, hidden_dims=hidden_dims)
    
    # Initialize with dummy input
    dummy_obs = jnp.zeros((1, observation_dim))
    network_params = network.init(key, dummy_obs)
    
    return network, network_params


def sample_action(
    action_logits: jnp.ndarray,
    key: jax.random.PRNGKey
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Sample action from action logits and return action and log probability.
    
    Args:
        action_logits: Action logits from the network
        key: PRNG key for sampling
        
    Returns:
        Tuple of (action, log_probability)
    """
    # Sample action from categorical distribution
    action = jax.random.categorical(key, action_logits)
    
    # Calculate log probability
    log_probs = jax.nn.log_softmax(action_logits)
    
    # Handle both batched and unbatched inputs
    if action_logits.ndim == 2:
        # Batched input
        batch_idx = jnp.arange(action_logits.shape[0])
        log_prob = log_probs[batch_idx, action]
    else:
        # Unbatched input
        log_prob = log_probs[action]
    
    return action, log_prob


def evaluate_action(
    action_logits: jnp.ndarray,
    action: jnp.ndarray
) -> jnp.ndarray:
    """
    Evaluate log probability of an action given action logits.
    
    Args:
        action_logits: Action logits from the network
        action: Action to evaluate
        
    Returns:
        Log probability of the action
    """
    log_prob = jax.nn.log_softmax(action_logits)
    log_prob = log_prob[action]
    
    return log_prob


def get_entropy(action_logits: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate entropy of the action distribution.
    
    Args:
        action_logits: Action logits from the network
        
    Returns:
        Entropy of the distribution
    """
    log_probs = jax.nn.log_softmax(action_logits)
    probs = jax.nn.softmax(action_logits)
    entropy = -jnp.sum(probs * log_probs, axis=-1)
    
    return entropy