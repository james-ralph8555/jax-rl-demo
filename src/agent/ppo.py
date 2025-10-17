"""PPO algorithm implementation"""

import jax
import jax.numpy as jnp
import optax
from typing import Tuple, Dict, Any, Optional
from functools import partial

from .network import PPONetwork, create_networks
from .utils import init_optimizer, compute_gae, normalize_advantages, create_minibatches, compute_ppo_loss


class PPOAgent:
    """Proximal Policy Optimization agent"""
    
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        learning_rate: float = 3e-4,
        clip_epsilon: float = 0.2,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        epochs_per_update: int = 10,
        batch_size: int = 64,
        minibatch_size: int = 64,
        key: Optional[jax.random.PRNGKey] = None
    ):
        """
        Initialize PPO agent
        
        Args:
            observation_dim: Dimension of observation space
            action_dim: Dimension of action space
            learning_rate: Learning rate for optimizer
            clip_epsilon: PPO clipping parameter
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            entropy_coef: Entropy regularization coefficient
            value_coef: Value function loss coefficient
            max_grad_norm: Maximum gradient norm for clipping
            epochs_per_update: Number of epochs per PPO update
            batch_size: Batch size for collecting experience
            minibatch_size: Mini-batch size for updates
            key: Random key for initialization
        """
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.epochs_per_update = epochs_per_update
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        
        if key is None:
            key = jax.random.PRNGKey(0)
        
        # Initialize networks and optimizer
        self.network, self.network_params = create_networks(
            observation_dim, action_dim, (64, 64), key
        )
        self.optimizer = init_optimizer(learning_rate)
        self.optimizer_state = self.optimizer.init(self.network_params)
        
        # Training state
        self.key = key
        
    def select_action(
        self, 
        network_params: dict, 
        observation: jnp.ndarray,
        key: jax.random.PRNGKey
    ) -> Tuple[jnp.ndarray, jnp.ndarray, dict]:
        """
        Select action given observation
        
        Args:
            network_params: Network parameters
            observation: Current observation
            key: Random key for sampling
            
        Returns:
            action: Selected action
            log_prob: Log probability of action
            value: Value estimate
        """
        # Forward pass through network
        policy_logits, value = self.network.apply(network_params, observation)
        
        # Sample action from policy
        from .network import sample_action
        action, log_prob = sample_action(policy_logits, key)
        
        return action, log_prob, {"value": value}
    
    def collect_episode(
        self,
        network_params: dict,
        env,
        max_steps: int = 1000,
        key: jax.random.PRNGKey = None
    ) -> Dict[str, jnp.ndarray]:
        """
        Collect one episode of experience
        
        Args:
            network_params: Network parameters
            env: Environment instance
            max_steps: Maximum steps per episode
            key: Random key for action sampling
            
        Returns:
            Dictionary containing episode data
        """
        if key is None:
            key = self.key
            
        observations, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
        
        obs, info = env.reset()
        done = False
        step = 0
        
        while not done and step < max_steps:
            key, subkey = jax.random.split(key)
            
            # Convert observation to JAX array
            obs_jax = jnp.array(obs, dtype=jnp.float32)
            
            # Select action
            action, log_prob, value_info = self.select_action(
                network_params, obs_jax[None, :], subkey
            )
            
            # Take action in environment
            next_obs, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated
            
            # Store experience
            observations.append(obs_jax)
            actions.append(jnp.array([action.item()], dtype=jnp.int32))
            rewards.append(jnp.array([reward], dtype=jnp.float32))
            log_probs.append(log_prob)
            values.append(value_info["value"].squeeze())  # Remove extra dimension
            dones.append(jnp.array([done], dtype=jnp.float32))
            
            obs = next_obs
            step += 1
        
        # Convert lists to arrays
        episode_data = {
            "observations": jnp.stack(observations),
            "actions": jnp.concatenate(actions),
            "rewards": jnp.concatenate(rewards),
            "log_probs": jnp.concatenate(log_probs),
            "values": jnp.stack(values),
            "dones": jnp.concatenate(dones)
        }
        
        return episode_data
    
    def compute_advantages_and_returns(
        self,
        rewards: jnp.ndarray,
        values: jnp.ndarray,
        dones: jnp.ndarray,
        next_value: jnp.ndarray = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute advantages and returns using GAE
        
        Args:
            rewards: Episode rewards
            values: Value estimates
            dones: Done flags
            next_value: Value of next state (0 if terminal)
            
        Returns:
            advantages: Computed advantages
            returns: Computed returns
        """
        if next_value is None:
            next_value = 0.0
            
        # Compute advantages using GAE
        advantages, _ = compute_gae(
            rewards, values, dones, next_value, self.gamma, self.gae_lambda
        )
        
        # Compute returns
        returns = advantages + values
        
        # Normalize advantages
        advantages = normalize_advantages(advantages)
        
        return advantages, returns
    
    @partial(jax.jit, static_argnums=(0,))
    def update_step(
        self,
        network_params: dict,
        optimizer_state: dict,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        old_log_probs: jnp.ndarray,
        advantages: jnp.ndarray,
        returns: jnp.ndarray,
        key: jax.random.PRNGKey
    ) -> Tuple[dict, dict, Dict[str, jnp.ndarray]]:
        """
        Single PPO update step
        
        Args:
            network_params: Current network parameters
            optimizer_state: Current optimizer state
            observations: Batch of observations
            actions: Batch of actions
            old_log_probs: Old log probabilities
            advantages: Advantage estimates
            returns: Return estimates
            key: Random key
            
        Returns:
            Updated network parameters
            Updated optimizer state
            Loss statistics
        """
        def loss_fn(params):
            # Forward pass
            policy_logits, values = self.network.apply(params, observations)
            
            # Compute PPO loss
            total_loss, loss_dict = compute_ppo_loss(
                policy_logits, values, actions, old_log_probs, advantages, returns,
                self.clip_epsilon, self.entropy_coef, self.value_coef
            )
            
            policy_loss = loss_dict['policy_loss']
            value_loss = loss_dict['value_loss']
            entropy_loss = loss_dict['entropy_loss']
            
            return total_loss, {
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy_loss": entropy_loss,
                "total_loss": total_loss
            }
        
        # Compute gradients and loss
        (loss, loss_info), grads = jax.value_and_grad(loss_fn, has_aux=True)(network_params)
        
        # Clip gradients
        grad_norm = optax.global_norm(grads)
        scale = jnp.minimum(1.0, self.max_grad_norm / (grad_norm + 1e-8))
        grads = jax.tree_util.tree_map(lambda g: g * scale, grads)
        
        # Apply optimizer updates
        updates, new_optimizer_state = self.optimizer.update(grads, optimizer_state)
        new_network_params = optax.apply_updates(network_params, updates)
        
        # Add gradient norm to info
        loss_info["grad_norm"] = grad_norm
        
        return new_network_params, new_optimizer_state, loss_info
    
    def update(
        self,
        network_params: dict,
        optimizer_state: dict,
        episode_data: Dict[str, jnp.ndarray],
        key: jax.random.PRNGKey
    ) -> Tuple[dict, dict, Dict[str, jnp.ndarray]]:
        """
        Perform PPO update on collected episode data
        
        Args:
            network_params: Current network parameters
            optimizer_state: Current optimizer state
            episode_data: Collected episode data
            key: Random key for minibatch creation
            
        Returns:
            Updated network parameters
            Updated optimizer state
            Training statistics
        """
        observations = episode_data["observations"]
        actions = episode_data["actions"]
        old_log_probs = episode_data["log_probs"]
        values = episode_data["values"]
        rewards = episode_data["rewards"]
        dones = episode_data["dones"]
        
        # Compute advantages and returns
        advantages, returns = self.compute_advantages_and_returns(
            rewards, values, dones
        )
        
        # Flatten data for minibatch processing
        observations = observations.reshape(-1, observations.shape[-1])
        actions = actions.reshape(-1)
        old_log_probs = old_log_probs.reshape(-1)
        advantages = advantages.reshape(-1)
        returns = returns.reshape(-1)
        
        # Training statistics
        total_stats = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy_loss": 0.0,
            "total_loss": 0.0,
            "grad_norm": 0.0,
            "num_updates": 0
        }
        
        # Perform multiple epochs of updates
        for epoch in range(self.epochs_per_update):
            # Create minibatches
            key, subkey = jax.random.split(key)
            minibatches, _ = create_minibatches(
                observations, actions, old_log_probs, advantages, returns,
                batch_size=self.minibatch_size, key=subkey
            )
            
            # Update on each minibatch
            for batch in minibatches:
                batch_obs, batch_actions, batch_old_log_probs, batch_advantages, batch_returns = batch
                
                network_params, optimizer_state, step_stats = self.update_step(
                    network_params, optimizer_state,
                    batch_obs, batch_actions, batch_old_log_probs,
                    batch_advantages, batch_returns, key
                )
                
                # Accumulate statistics
                for key_name in total_stats:
                    if key_name != "num_updates":
                        total_stats[key_name] += step_stats[key_name]
                total_stats["num_updates"] += 1
        
        # Average statistics
        for key_name in total_stats:
            if key_name != "num_updates":
                total_stats[key_name] /= total_stats["num_updates"]
        
        return network_params, optimizer_state, total_stats
    
    def train_step(
        self,
        network_params: dict,
        optimizer_state: dict,
        env,
        key: jax.random.PRNGKey
    ) -> Tuple[dict, dict, Dict[str, jnp.ndarray]]:
        """
        Complete training step: collect episode and update
        
        Args:
            network_params: Current network parameters
            optimizer_state: Current optimizer state
            env: Environment for training
            key: Random key
            
        Returns:
            Updated network parameters
            Updated optimizer state
            Training statistics
        """
        # Collect episode
        key, subkey = jax.random.split(key)
        episode_data = self.collect_episode(network_params, env, key=subkey)
        
        # Update networks
        key, subkey = jax.random.split(key)
        new_network_params, new_optimizer_state, stats = self.update(
            network_params, optimizer_state, episode_data, subkey
        )
        
        # Add episode statistics
        episode_length = len(episode_data["rewards"])
        episode_reward = jnp.sum(episode_data["rewards"])
        
        stats.update({
            "episode_length": episode_length,
            "episode_reward": episode_reward
        })
        
        return new_network_params, new_optimizer_state, stats