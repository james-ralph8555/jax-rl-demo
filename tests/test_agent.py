"""Tests for agent network and utilities"""

import jax
import jax.numpy as jnp
import pytest
from src.agent.network import (
    ActorNetwork,
    CriticNetwork,
    PPONetwork,
    create_networks,
    sample_action,
    evaluate_action,
    get_entropy
)
from src.agent.utils import (
    discount_rewards,
    compute_gae,
    normalize_advantages,
    create_minibatches,
    compute_ppo_loss
)


class TestActorNetwork:
    """Test cases for ActorNetwork."""
    
    def test_init(self):
        """Test network initialization."""
        action_dim = 2
        network = ActorNetwork(action_dim=action_dim)
        assert network.action_dim == action_dim
        assert network.hidden_dims == (64, 64)
    
    def test_forward_pass(self):
        """Test forward pass through actor network."""
        action_dim = 2
        obs_dim = 4
        network = ActorNetwork(action_dim=action_dim)
        
        # Initialize with dummy input
        key = jax.random.PRNGKey(0)
        dummy_obs = jnp.zeros((1, obs_dim))
        params = network.init(key, dummy_obs)
        
        # Test forward pass
        action_logits = network.apply(params, dummy_obs)
        assert action_logits.shape == (1, action_dim)
    
    def test_batch_forward_pass(self):
        """Test batch forward pass."""
        action_dim = 2
        obs_dim = 4
        batch_size = 10
        network = ActorNetwork(action_dim=action_dim)
        
        # Initialize with dummy input
        key = jax.random.PRNGKey(0)
        dummy_obs = jnp.zeros((batch_size, obs_dim))
        params = network.init(key, dummy_obs)
        
        # Test forward pass
        action_logits = network.apply(params, dummy_obs)
        assert action_logits.shape == (batch_size, action_dim)


class TestCriticNetwork:
    """Test cases for CriticNetwork."""
    
    def test_init(self):
        """Test network initialization."""
        network = CriticNetwork()
        assert network.hidden_dims == (64, 64)
    
    def test_forward_pass(self):
        """Test forward pass through critic network."""
        obs_dim = 4
        network = CriticNetwork()
        
        # Initialize with dummy input
        key = jax.random.PRNGKey(0)
        dummy_obs = jnp.zeros((1, obs_dim))
        params = network.init(key, dummy_obs)
        
        # Test forward pass
        value = network.apply(params, dummy_obs)
        assert value.shape == (1, 1)
    
    def test_batch_forward_pass(self):
        """Test batch forward pass."""
        obs_dim = 4
        batch_size = 10
        network = CriticNetwork()
        
        # Initialize with dummy input
        key = jax.random.PRNGKey(0)
        dummy_obs = jnp.zeros((batch_size, obs_dim))
        params = network.init(key, dummy_obs)
        
        # Test forward pass
        value = network.apply(params, dummy_obs)
        assert value.shape == (batch_size, 1)


class TestPPONetwork:
    """Test cases for PPONetwork."""
    
    def test_init(self):
        """Test network initialization."""
        action_dim = 2
        network = PPONetwork(action_dim=action_dim)
        assert network.action_dim == action_dim
        assert network.hidden_dims == (64, 64)
    
    def test_forward_pass(self):
        """Test forward pass through PPO network."""
        action_dim = 2
        obs_dim = 4
        network = PPONetwork(action_dim=action_dim)
        
        # Initialize with dummy input
        key = jax.random.PRNGKey(0)
        dummy_obs = jnp.zeros((1, obs_dim))
        params = network.init(key, dummy_obs)
        
        # Test forward pass
        action_logits, value = network.apply(params, dummy_obs)
        assert action_logits.shape == (1, action_dim)
        assert value.shape == (1, 1)
    
    def test_individual_forward_passes(self):
        """Test individual actor and critic forward passes."""
        action_dim = 2
        obs_dim = 4
        network = PPONetwork(action_dim=action_dim)
        
        # Initialize with dummy input
        key = jax.random.PRNGKey(0)
        dummy_obs = jnp.zeros((1, obs_dim))
        params = network.init(key, dummy_obs)
        
        # Test individual forward passes
        actor_output = network.apply(params, dummy_obs, method=network.actor_forward)
        critic_output = network.apply(params, dummy_obs, method=network.critic_forward)
        
        assert actor_output.shape == (1, action_dim)
        assert critic_output.shape == (1, 1)


class TestNetworkUtilities:
    """Test cases for network utility functions."""
    
    def test_create_networks(self):
        """Test network creation utility."""
        obs_dim = 4
        action_dim = 2
        
        # Create networks
        key = jax.random.PRNGKey(0)
        network, params = create_networks(obs_dim, action_dim, key=key)
        
        # Test forward pass
        dummy_obs = jnp.zeros((1, obs_dim))
        action_logits, value = network.apply(params, dummy_obs)
        
        assert action_logits.shape == (1, action_dim)
        assert value.shape == (1, 1)
    
    def test_sample_action(self):
        """Test action sampling."""
        action_dim = 2
        batch_size = 10
        
        # Create dummy action logits
        key = jax.random.PRNGKey(0)
        action_logits = jax.random.normal(key, (batch_size, action_dim))
        
        # Test action sampling
        key, subkey = jax.random.split(key)
        actions, log_probs = jax.vmap(sample_action)(action_logits, jax.random.split(subkey, batch_size))
        
        assert actions.shape == (batch_size,)
        assert log_probs.shape == (batch_size,)
        assert jnp.all(actions >= 0) and jnp.all(actions < action_dim)
    
    def test_evaluate_action(self):
        """Test action evaluation."""
        action_dim = 2
        batch_size = 10
        
        # Create dummy action logits and actions
        key = jax.random.PRNGKey(0)
        action_logits = jax.random.normal(key, (batch_size, action_dim))
        actions = jax.random.randint(key, (batch_size,), 0, action_dim)
        
        # Test action evaluation
        log_probs = jax.vmap(evaluate_action)(action_logits, actions)
        
        assert log_probs.shape == (batch_size,)
        assert jnp.all(log_probs < 0)  # Log probabilities should be negative
    
    def test_get_entropy(self):
        """Test entropy calculation."""
        action_dim = 2
        batch_size = 10
        
        # Create dummy action logits
        key = jax.random.PRNGKey(0)
        action_logits = jax.random.normal(key, (batch_size, action_dim))
        
        # Test entropy calculation
        entropies = jax.vmap(get_entropy)(action_logits)
        
        assert entropies.shape == (batch_size,)
        assert jnp.all(entropies > 0)  # Entropy should be positive


class TestAgentUtilities:
    """Test cases for agent utility functions."""
    
    def test_discount_rewards(self):
        """Test reward discounting."""
        rewards = jnp.array([1.0, 1.0, 1.0, 1.0])
        dones = jnp.array([0.0, 0.0, 0.0, 1.0])
        gamma = 0.99
        
        discounted = discount_rewards(rewards, dones, gamma)
        
        # Expected: [1 + 0.99 + 0.99^2 + 0.99^3, 1 + 0.99 + 0.99^2, 1 + 0.99, 1]
        expected = jnp.array([
            1.0 + 0.99 + 0.99**2 + 0.99**3,
            1.0 + 0.99 + 0.99**2,
            1.0 + 0.99,
            1.0
        ])
        
        assert jnp.allclose(discounted, expected, atol=1e-6)
    
    def test_compute_gae(self):
        """Test GAE computation."""
        rewards = jnp.array([1.0, 1.0, 1.0])
        values = jnp.array([2.0, 2.0, 2.0])
        next_values = jnp.array([2.0, 2.0, 0.0])
        dones = jnp.array([0.0, 0.0, 1.0])
        gamma = 0.99
        lambda_gae = 0.95
        
        advantages, returns = compute_gae(rewards, values, dones, next_values, gamma, lambda_gae)
        
        assert advantages.shape == (3,)
        assert returns.shape == (3,)
        assert jnp.allclose(returns, advantages + values)
    
    def test_normalize_advantages(self):
        """Test advantage normalization."""
        advantages = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        normalized = normalize_advantages(advantages)
        
        assert jnp.abs(jnp.mean(normalized)) < 1e-6
        assert jnp.abs(jnp.std(normalized) - 1.0) < 1e-6
    
    def test_create_minibatches(self):
        """Test minibatch creation."""
        batch_size = 3
        total_samples = 10
        
        key = jax.random.PRNGKey(0)
        data1 = jnp.arange(total_samples)
        data2 = jnp.arange(total_samples) * 2
        
        minibatches, new_key = create_minibatches(data1, data2, batch_size=batch_size, key=key)
        
        expected_num_batches = (total_samples + batch_size - 1) // batch_size
        assert len(minibatches) == expected_num_batches
        
        # Check that all samples are included
        all_indices = []
        for batch in minibatches:
            batch_data1, batch_data2 = batch
            assert batch_data1.shape[0] <= batch_size
            assert jnp.all(batch_data2 == batch_data1 * 2)
            all_indices.extend(batch_data1.tolist())
        
        assert set(all_indices) == set(range(total_samples))
    
    def test_compute_ppo_loss(self):
        """Test PPO loss computation."""
        batch_size = 4
        action_dim = 2
        obs_dim = 4
        
        key = jax.random.PRNGKey(0)
        
        # Create network and dummy data
        network, params = create_networks(obs_dim, action_dim, key=key)
        
        # Generate dummy observations and get network outputs
        dummy_obs = jax.random.normal(key, (batch_size, obs_dim))
        action_logits, values = network.apply(params, dummy_obs)
        
        # Generate dummy actions and other data
        actions = jax.random.randint(key, (batch_size,), 0, action_dim)
        old_log_probs = jax.random.normal(key, (batch_size,)) * 0.1
        advantages = jax.random.normal(key, (batch_size,))
        returns = jax.random.normal(key, (batch_size,))
        
        # Compute loss
        total_loss, loss_dict = compute_ppo_loss(
            action_logits, values, actions, old_log_probs, advantages, returns
        )
        
        assert total_loss.shape == ()
        expected_keys = {'policy_loss', 'value_loss', 'entropy_loss', 'total_loss'}
        assert set(loss_dict.keys()) == expected_keys
        
        # Check that total loss is combination of components
        expected_total = (
            loss_dict['policy_loss'] + 
            0.5 * loss_dict['value_loss'] + 
            0.01 * loss_dict['entropy_loss']
        )
        assert jnp.allclose(total_loss, expected_total)