import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Any, Optional


def init_optimizer(
    learning_rate: float = 3e-4
) -> Any:
    """
    Initialize Adam optimizer using Optax.
    
    Args:
        learning_rate: Learning rate for the optimizer
        
    Returns:
        Optax optimizer
    """
    import optax
    
    # Create optimizer with Adam
    optimizer = optax.adam(learning_rate=learning_rate)
    
    return optimizer


def discount_rewards(
    rewards: jnp.ndarray,
    dones: jnp.ndarray,
    gamma: float = 0.99
) -> jnp.ndarray:
    """
    Compute discounted rewards with bootstrapping.
    
    Args:
        rewards: Array of rewards
        dones: Array of episode termination flags
        gamma: Discount factor
        
    Returns:
        Array of discounted rewards
    """
    def _discount_scan(carry, inp):
        reward, done = inp
        new_carry = jnp.where(done, 0.0, carry) * gamma + reward
        return new_carry, new_carry
    
    # Reverse scan for proper discounting
    _, discounted = jax.lax.scan(
        _discount_scan, 
        jnp.zeros_like(rewards[-1]), 
        (rewards, dones),
        reverse=True
    )
    
    return discounted


def compute_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    dones: jnp.ndarray,
    next_values: jnp.ndarray,
    gamma: float = 0.99,
    lambda_gae: float = 0.95
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute Generalized Advantage Estimation (GAE).
    
    Args:
        rewards: Array of rewards
        values: Array of value predictions
        dones: Array of episode termination flags
        next_values: Array of next state value predictions
        gamma: Discount factor
        lambda_gae: GAE parameter
        
    Returns:
        Tuple of (advantages, returns)
    """
    # Compute TD errors
    td_errors = rewards + gamma * next_values * (1.0 - dones) - values
    
    def _gae_scan(carry, td_error):
        gae = carry * lambda_gae * gamma + td_error
        return gae, gae
    
    # Compute GAE using reverse scan
    _, advantages = jax.lax.scan(
        _gae_scan,
        jnp.zeros_like(td_errors[-1]),
        td_errors,
        reverse=True
    )
    
    # Compute returns
    returns = advantages + values
    
    return advantages, returns


def normalize_advantages(
    advantages: jnp.ndarray,
    epsilon: float = 1e-8
) -> jnp.ndarray:
    """
    Normalize advantages for training stability.
    
    Args:
        advantages: Array of advantages
        epsilon: Small constant to avoid division by zero
        
    Returns:
        Normalized advantages
    """
    mean = jnp.mean(advantages)
    std = jnp.std(advantages)
    
    return (advantages - mean) / (std + epsilon)


def create_minibatches(
    *arrays: jnp.ndarray,
    batch_size: int,
    key: jax.random.PRNGKey
) -> Tuple[Tuple[jnp.ndarray, ...], jax.random.PRNGKey]:
    """
    Create random minibatches from arrays.
    
    Args:
        *arrays: Arrays to batch
        batch_size: Size of each minibatch
        key: PRNG key for shuffling
        
    Returns:
        Tuple of (minibatches, new_key)
    """
    # Get total number of samples
    total_samples = arrays[0].shape[0]
    
    # Shuffle indices
    key, subkey = jax.random.split(key)
    indices = jax.random.permutation(subkey, total_samples)
    
    # Create minibatches
    minibatches = []
    for start_idx in range(0, total_samples, batch_size):
        end_idx = min(start_idx + batch_size, total_samples)
        batch_indices = indices[start_idx:end_idx]
        
        batch = tuple(arr[batch_indices] for arr in arrays)
        minibatches.append(batch)
    
    return tuple(minibatches), key


def compute_ppo_loss(
    action_logits: jnp.ndarray,
    values: jnp.ndarray,
    actions: jnp.ndarray,
    old_log_probs: jnp.ndarray,
    advantages: jnp.ndarray,
    returns: jnp.ndarray,
    clip_epsilon: float = 0.2,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Compute PPO loss with policy clipping.
    
    Args:
        action_logits: Action logits from the network
        values: Value predictions from the network
        actions: Actions taken
        old_log_probs: Log probabilities of actions under old policy
        advantages: Advantage estimates
        returns: Return estimates
        clip_epsilon: Clipping parameter for PPO
        entropy_coef: Entropy coefficient
        value_coef: Value function coefficient
        
    Returns:
        Tuple of (total_loss, loss_dict)
    """
    # Compute current log probabilities
    from .network import evaluate_action, get_entropy
    current_log_probs = jax.vmap(evaluate_action)(action_logits, actions)
    
    # Compute entropy
    entropy = jax.vmap(get_entropy)(action_logits)
    
    # Compute policy ratio
    ratio = jnp.exp(current_log_probs - old_log_probs)
    
    # Compute clipped policy loss
    policy_loss_unclipped = ratio * advantages
    policy_loss_clipped = jnp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    policy_loss = -jnp.minimum(policy_loss_unclipped, policy_loss_clipped).mean()
    
    # Compute value loss
    value_loss = jnp.square(values.squeeze() - returns).mean()
    
    # Compute entropy loss
    entropy_loss = -entropy.mean()
    
    # Compute KL divergence between old and new policies
    kl_divergence = jnp.mean(old_log_probs - current_log_probs)
    
    # Total loss
    total_loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
    
    loss_dict = {
        'policy_loss': policy_loss,
        'value_loss': value_loss,
        'entropy_loss': entropy_loss,
        'total_loss': total_loss,
        'kl_divergence': kl_divergence
    }
    
    return total_loss, loss_dict