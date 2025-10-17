"""Test PPO algorithm implementation"""

import jax
import jax.numpy as jnp
import numpy as np
from src.agent.ppo import PPOAgent
from src.environment.cartpole import CartPoleWrapper


def test_ppo_agent_initialization():
    """Test PPO agent initialization"""
    print("Testing PPO agent initialization...")
    
    # Create agent
    agent = PPOAgent(
        observation_dim=4,
        action_dim=2,
        learning_rate=3e-4,
        clip_epsilon=0.2,
        gamma=0.99,
        gae_lambda=0.95,
        key=jax.random.PRNGKey(42)
    )
    
    # Check attributes
    assert agent.observation_dim == 4
    assert agent.action_dim == 2
    assert agent.clip_epsilon == 0.2
    assert agent.gamma == 0.99
    assert agent.gae_lambda == 0.95
    assert agent.epochs_per_update == 10
    assert agent.batch_size == 64
    assert agent.minibatch_size == 64
    
    # Check network initialization
    assert agent.network is not None
    assert agent.network_params is not None
    assert agent.optimizer is not None
    assert agent.optimizer_state is not None
    
    print("✓ PPO agent initialization test passed")


def test_select_action():
    """Test action selection"""
    print("Testing action selection...")
    
    agent = PPOAgent(
        observation_dim=4,
        action_dim=2,
        key=jax.random.PRNGKey(42)
    )
    
    # Create dummy observation
    observation = jnp.array([0.1, -0.2, 0.3, -0.4], dtype=jnp.float32)
    key = jax.random.PRNGKey(123)
    
    # Select action
    action, log_prob, value_info = agent.select_action(
        agent.network_params, observation[None, :], key
    )
    
    # Check outputs
    assert action.shape == (1,)
    assert action.dtype == jnp.int32
    assert action.item() in [0, 1]  # Valid actions for CartPole
    assert log_prob.shape == (1,) or log_prob.shape == ()  # Allow scalar
    assert "value" in value_info
    assert value_info["value"].shape == (1, 1)
    
    print("✓ Action selection test passed")


def test_collect_episode():
    """Test episode collection"""
    print("Testing episode collection...")
    
    agent = PPOAgent(
        observation_dim=4,
        action_dim=2,
        key=jax.random.PRNGKey(42)
    )
    
    # Create environment
    env = CartPoleWrapper()
    
    # Collect episode
    key = jax.random.PRNGKey(123)
    episode_data = agent.collect_episode(
        agent.network_params, env.env, max_steps=10, key=key
    )
    
    # Check episode data structure
    required_keys = ["observations", "actions", "rewards", "log_probs", "values", "dones"]
    for key_name in required_keys:
        assert key_name in episode_data
    
    # Check data shapes
    assert episode_data["observations"].shape[1] == 4  # observation_dim
    assert episode_data["actions"].shape == (episode_data["observations"].shape[0],)
    assert episode_data["rewards"].shape == (episode_data["observations"].shape[0],)
    assert episode_data["log_probs"].shape == (episode_data["observations"].shape[0],)
    assert episode_data["values"].shape == (episode_data["observations"].shape[0],) or episode_data["values"].shape == (episode_data["observations"].shape[0], 1)
    assert episode_data["dones"].shape == (episode_data["observations"].shape[0],)
    
    # Check data types
    assert episode_data["observations"].dtype == jnp.float32
    assert episode_data["actions"].dtype == jnp.int32
    assert episode_data["rewards"].dtype == jnp.float32
    assert episode_data["log_probs"].dtype == jnp.float32
    assert episode_data["values"].dtype == jnp.float32
    assert episode_data["dones"].dtype == jnp.float32
    
    print("✓ Episode collection test passed")


def test_compute_advantages_and_returns():
    """Test advantage and return computation"""
    print("Testing advantage and return computation...")
    
    agent = PPOAgent(
        observation_dim=4,
        action_dim=2,
        gamma=0.99,
        gae_lambda=0.95,
        key=jax.random.PRNGKey(42)
    )
    
    # Create dummy data
    rewards = jnp.array([1.0, 1.0, 1.0, 0.0], dtype=jnp.float32)  # Episode ends with reward 0
    values = jnp.array([0.5, 0.6, 0.7, 0.4], dtype=jnp.float32)
    dones = jnp.array([0.0, 0.0, 0.0, 1.0], dtype=jnp.float32)  # Last step is done
    
    # Compute advantages and returns
    advantages, returns = agent.compute_advantages_and_returns(rewards, values, dones)
    
    # Check shapes
    assert advantages.shape == rewards.shape
    assert returns.shape == rewards.shape
    
    # Check that returns are reasonable (should be >= rewards due to bootstrapping)
    # Note: this might not always hold due to discounting, so we just check they're finite
    assert jnp.all(jnp.isfinite(returns))
    
    # Check that advantages are normalized (approximately zero mean, unit variance)
    assert jnp.abs(jnp.mean(advantages)) < 1e-6
    assert jnp.abs(jnp.std(advantages) - 1.0) < 1e-6
    
    print("✓ Advantage and return computation test passed")


def test_update_step():
    """Test single PPO update step"""
    print("Testing PPO update step...")
    
    agent = PPOAgent(
        observation_dim=4,
        action_dim=2,
        key=jax.random.PRNGKey(42)
    )
    
    # Create dummy batch data
    batch_size = 8
    key = jax.random.PRNGKey(123)
    key, obs_key, action_key, logprob_key, adv_key, ret_key = jax.random.split(key, 6)
    observations = jax.random.normal(obs_key, (batch_size, 4)).astype(jnp.float32)
    actions = jax.random.randint(action_key, (batch_size,), 0, 2).astype(jnp.int32)
    old_log_probs = jax.random.normal(logprob_key, (batch_size,)).astype(jnp.float32)
    advantages = jax.random.normal(adv_key, (batch_size,)).astype(jnp.float32)
    returns = jax.random.normal(ret_key, (batch_size,)).astype(jnp.float32)
    
    # Perform update step
    new_params, new_opt_state, stats = agent.update_step(
        agent.network_params,
        agent.optimizer_state,
        observations,
        actions,
        old_log_probs,
        advantages,
        returns,
        key
    )
    
    # Check that parameters changed
    def params_are_equal(p1, p2):
        if isinstance(p1, dict) and isinstance(p2, dict):
            return all(params_are_equal(p1[k], p2[k]) for k in p1.keys())
        return jnp.allclose(p1, p2, atol=1e-8)
    
    assert not params_are_equal(new_params, agent.network_params)
    
    # Check optimizer state changed
    assert new_opt_state != agent.optimizer_state
    
    # Check statistics
    required_stats = ["policy_loss", "value_loss", "entropy_loss", "total_loss", "grad_norm"]
    for stat_name in required_stats:
        assert stat_name in stats
        assert isinstance(stats[stat_name], jnp.ndarray)
        assert stats[stat_name].shape == ()
    
    # Check that losses are finite
    assert jnp.isfinite(stats["policy_loss"])
    assert jnp.isfinite(stats["value_loss"])
    assert jnp.isfinite(stats["entropy_loss"])
    assert jnp.isfinite(stats["total_loss"])
    assert jnp.isfinite(stats["grad_norm"])
    
    print("✓ PPO update step test passed")


def test_update():
    """Test full PPO update with episode data"""
    print("Testing full PPO update...")
    
    agent = PPOAgent(
        observation_dim=4,
        action_dim=2,
        epochs_per_update=2,  # Reduce for faster testing
        minibatch_size=4,     # Reduce for faster testing
        key=jax.random.PRNGKey(42)
    )
    
    # Create dummy episode data
    episode_length = 8
    key = jax.random.PRNGKey(123)
    key, obs_key, action_key, logprob_key, value_key = jax.random.split(key, 5)
    episode_data = {
        "observations": jax.random.normal(obs_key, (episode_length, 4)).astype(jnp.float32),
        "actions": jax.random.randint(action_key, (episode_length,), 0, 2).astype(jnp.int32),
        "rewards": jnp.ones((episode_length,)).astype(jnp.float32),
        "log_probs": jax.random.normal(logprob_key, (episode_length,)).astype(jnp.float32),
        "values": jax.random.normal(value_key, (episode_length,)).astype(jnp.float32),
        "dones": jnp.zeros((episode_length,)).astype(jnp.float32).at[-1].set(1.0)
    }
    
    key = jax.random.PRNGKey(123)
    
    # Perform update
    new_params, new_opt_state, stats = agent.update(
        agent.network_params,
        agent.optimizer_state,
        episode_data,
        key
    )
    
    # Check that parameters changed
    def params_are_equal(p1, p2):
        if isinstance(p1, dict) and isinstance(p2, dict):
            return all(params_are_equal(p1[k], p2[k]) for k in p1.keys())
        return jnp.allclose(p1, p2, atol=1e-8)
    
    assert not params_are_equal(new_params, agent.network_params)
    
    # Check statistics
    required_stats = ["policy_loss", "value_loss", "entropy_loss", "total_loss", "grad_norm", "num_updates"]
    for stat_name in required_stats:
        assert stat_name in stats
    
    # Check that num_updates is correct
    expected_updates = agent.epochs_per_update * (episode_length // agent.minibatch_size)
    assert stats["num_updates"] == expected_updates
    
    print("✓ Full PPO update test passed")


def test_train_step():
    """Test complete training step"""
    print("Testing complete training step...")
    
    agent = PPOAgent(
        observation_dim=4,
        action_dim=2,
        epochs_per_update=2,  # Reduce for faster testing
        minibatch_size=4,     # Reduce for faster testing
        key=jax.random.PRNGKey(42)
    )
    
    # Create environment
    env = CartPoleWrapper()
    
    key = jax.random.PRNGKey(123)
    
    # Perform training step
    new_params, new_opt_state, stats = agent.train_step(
        agent.network_params,
        agent.optimizer_state,
        env.env,
        key
    )
    
    # Check that parameters changed
    def params_are_equal(p1, p2):
        if isinstance(p1, dict) and isinstance(p2, dict):
            return all(params_are_equal(p1[k], p2[k]) for k in p1.keys())
        return jnp.allclose(p1, p2, atol=1e-8)
    
    assert not params_are_equal(new_params, agent.network_params)
    
    # Check training statistics
    required_stats = [
        "policy_loss", "value_loss", "entropy_loss", "total_loss", "grad_norm",
        "num_updates", "episode_length", "episode_reward"
    ]
    for stat_name in required_stats:
        assert stat_name in stats
    
    # Check episode statistics
    assert stats["episode_length"] > 0
    assert stats["episode_reward"] >= 0.0
    assert isinstance(stats["episode_length"], (int, jnp.ndarray))
    assert isinstance(stats["episode_reward"], (float, jnp.ndarray))
    
    print("✓ Complete training step test passed")


def test_ppo_hyperparameters():
    """Test PPO agent with different hyperparameters"""
    print("Testing PPO agent with different hyperparameters...")
    
    # Test with custom hyperparameters
    agent = PPOAgent(
        observation_dim=4,
        action_dim=2,
        learning_rate=1e-3,
        clip_epsilon=0.1,
        gamma=0.95,
        gae_lambda=0.9,
        entropy_coef=0.02,
        value_coef=0.8,
        max_grad_norm=1.0,
        epochs_per_update=5,
        batch_size=32,
        minibatch_size=16,
        key=jax.random.PRNGKey(42)
    )
    
    # Check that hyperparameters are set correctly
    assert agent.learning_rate == 1e-3
    assert agent.clip_epsilon == 0.1
    assert agent.gamma == 0.95
    assert agent.gae_lambda == 0.9
    assert agent.entropy_coef == 0.02
    assert agent.value_coef == 0.8
    assert agent.max_grad_norm == 1.0
    assert agent.epochs_per_update == 5
    assert agent.batch_size == 32
    assert agent.minibatch_size == 16
    
    # Test that agent still works with custom hyperparameters
    env = CartPoleWrapper()
    key = jax.random.PRNGKey(123)
    
    new_params, new_opt_state, stats = agent.train_step(
        agent.network_params,
        agent.optimizer_state,
        env.env,
        key
    )
    
    # Check that training still works
    assert new_params is not None
    assert new_opt_state is not None
    assert stats is not None
    assert "episode_reward" in stats
    
    print("✓ PPO hyperparameters test passed")


def run_all_ppo_tests():
    """Run all PPO tests"""
    print("Running PPO algorithm tests...")
    print("=" * 50)
    
    test_ppo_agent_initialization()
    test_select_action()
    test_collect_episode()
    test_compute_advantages_and_returns()
    test_update_step()
    test_update()
    test_train_step()
    test_ppo_hyperparameters()
    
    print("=" * 50)
    print("✅ All PPO algorithm tests passed!")


if __name__ == "__main__":
    run_all_ppo_tests()