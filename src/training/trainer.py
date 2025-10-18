"""Training loop implementation for PPO agent."""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, Optional, Callable, Tuple
import time
from collections import defaultdict

from ..agent.ppo import PPOAgent
from ..environment.cartpole import CartPoleWrapper
from ..visualization.mlflow_logger import MLflowLogger


class PPOTrainer:
    """Trainer class for PPO agent with CartPole environment."""
    
    def __init__(
        self,
        agent: PPOAgent,
        env: CartPoleWrapper,
        max_episodes: int = 1000,
        max_steps_per_episode: int = 500,
        target_reward: float = 195.0,
        convergence_window: int = 100,
        early_stopping_patience: int = 200,
        eval_frequency: int = 50,
        eval_episodes: int = 10,
        save_frequency: int = 100,
        log_frequency: int = 10,
        enable_mlflow: bool = True,
        mlflow_experiment_name: str = "cartpole-ppo",
        key: Optional[jax.random.PRNGKey] = None
    ):
        """
        Initialize PPO trainer.
        
        Args:
            agent: PPO agent instance
            env: CartPole environment instance
            max_episodes: Maximum number of training episodes
            max_steps_per_episode: Maximum steps per episode
            target_reward: Target average reward for convergence
            convergence_window: Window size for convergence checking
            early_stopping_patience: Episodes to wait after convergence before stopping
            eval_frequency: Frequency of evaluation episodes
            eval_episodes: Number of episodes for evaluation
            save_frequency: Frequency of model saving
            log_frequency: Frequency of logging
            enable_mlflow: Whether to enable MLflow logging
            mlflow_experiment_name: Name of the MLflow experiment
            key: Random key for reproducibility
        """
        self.agent = agent
        self.env = env
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.target_reward = target_reward
        self.convergence_window = convergence_window
        self.early_stopping_patience = early_stopping_patience
        self.eval_frequency = eval_frequency
        self.eval_episodes = eval_episodes
        self.save_frequency = save_frequency
        self.log_frequency = log_frequency
        self.enable_mlflow = enable_mlflow
        self.mlflow_experiment_name = mlflow_experiment_name
        
        if key is None:
            key = jax.random.PRNGKey(42)
        self.key = key
        
        # Training state
        self.episode_count = 0
        self.step_count = 0
        self.best_avg_reward = float('-inf')
        self.convergence_count = 0
        self.should_stop = False
        
        # Metrics tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = defaultdict(list)
        self.eval_rewards = []
        
        # MLflow setup
        self.mlflow_logger = None
        if self.enable_mlflow:
            self.mlflow_logger = MLflowLogger(self.mlflow_experiment_name)
            self.mlflow_logger.setup_autologging()
        
    def collect_episode(self, key: jax.random.PRNGKey) -> Dict[str, jnp.ndarray]:
        """
        Collect a single episode of experience.
        
        Args:
            key: Random key for action selection
            
        Returns:
            Dictionary containing episode data
        """
        obs, info = self.env.reset()
        episode_data = {
            'observations': [obs],
            'actions': [],
            'rewards': [],
            'next_observations': [],
            'dones': [],
            'log_probs': [],
            'values': []
        }
        
        for step in range(self.max_steps_per_episode):
            # Select action
            key, action_key = jax.random.split(key)
            action, log_prob, value_info = self.agent.select_action(self.agent.network_params, obs[None, :], action_key)
            value = value_info["value"]
            
            # Take environment step
            next_obs, reward, terminated, truncated, info = self.env.step(int(action.item()))
            done = terminated or truncated
            
            # Store transition
            episode_data['actions'].append(action.item())
            episode_data['rewards'].append(reward)
            episode_data['next_observations'].append(next_obs)
            episode_data['dones'].append(done)
            episode_data['log_probs'].append(log_prob.item())
            episode_data['values'].append(value.item())
            
            obs = next_obs
            
            if done:
                break
        
        # Convert lists to arrays
        episode_data = {
            'observations': jnp.array(episode_data['observations']),
            'actions': jnp.array(episode_data['actions']),
            'rewards': jnp.array(episode_data['rewards']),
            'next_observations': jnp.array(episode_data['next_observations']),
            'dones': jnp.array(episode_data['dones']),
            'log_probs': jnp.array(episode_data['log_probs']),
            'values': jnp.array(episode_data['values'])
        }
        
        # Reshape actions to be 1D
        episode_data['actions'] = episode_data['actions'].reshape(-1)
        episode_data['log_probs'] = episode_data['log_probs'].reshape(-1)
        episode_data['values'] = episode_data['values'].reshape(-1)
        
        return episode_data
    
    def collect_batch(self, key: jax.random.PRNGKey, batch_size: int) -> Dict[str, jnp.ndarray]:
        """
        Collect a batch of episodes.
        
        Args:
            key: Random key for episode collection
            batch_size: Number of episodes to collect
            
        Returns:
            Batch of episode data
        """
        batch_data = []
        
        for i in range(batch_size):
            key, episode_key = jax.random.split(key)
            episode_data = self.collect_episode(episode_key)
            batch_data.append(episode_data)
        
        # Concatenate episodes
        batch = {
            'observations': jnp.concatenate([ep['observations'] for ep in batch_data]),
            'actions': jnp.concatenate([ep['actions'] for ep in batch_data]),
            'rewards': jnp.concatenate([ep['rewards'] for ep in batch_data]),
            'next_observations': jnp.concatenate([ep['next_observations'] for ep in batch_data]),
            'dones': jnp.concatenate([ep['dones'] for ep in batch_data]),
            'log_probs': jnp.concatenate([ep['log_probs'] for ep in batch_data]),
            'values': jnp.concatenate([ep['values'] for ep in batch_data])
        }
        
        return batch
    
    def evaluate(self, num_episodes: int) -> Dict[str, float]:
        """
        Evaluate the current policy.
        
        Args:
            num_episodes: Number of evaluation episodes
            
        Returns:
            Evaluation metrics
        """
        eval_rewards = []
        eval_lengths = []
        
        for _ in range(num_episodes):
            obs, info = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            for step in range(self.max_steps_per_episode):
                # Select action without exploration (deterministic)
                action, _, value_info = self.agent.select_action(self.agent.network_params, obs[None, :], jax.random.PRNGKey(0))
                
                obs, reward, terminated, truncated, info = self.env.step(int(action.item()))
                episode_reward += reward
                episode_length += 1
                
                if terminated or truncated:
                    break
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
        
        return {
            'avg_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'max_reward': np.max(eval_rewards),
            'min_reward': np.min(eval_rewards),
            'avg_length': np.mean(eval_lengths)
        }
    
    def check_convergence(self) -> bool:
        """
        Check if training has converged.
        
        Returns:
            True if converged, False otherwise
        """
        if len(self.episode_rewards) < self.convergence_window:
            return False
        
        recent_rewards = self.episode_rewards[-self.convergence_window:]
        avg_reward = np.mean(recent_rewards)
        
        if avg_reward >= self.target_reward:
            if avg_reward > self.best_avg_reward:
                self.best_avg_reward = avg_reward
                self.convergence_count = 0
            else:
                self.convergence_count += 1
            
            return self.convergence_count >= self.early_stopping_patience
        
        return False
    
    def train_step(self, key: jax.random.PRNGKey) -> Dict[str, Any]:
        """
        Perform one training step (collect batch and update).
        
        Args:
            key: Random key for this training step
            
        Returns:
            Training metrics for this step
        """
        # Use agent's train_step method which handles collection and update
        new_network_params, new_optimizer_state, update_metrics = self.agent.train_step(
            self.agent.network_params, self.agent.optimizer_state, self.env, key
        )
        
        # Extract episode statistics from update_metrics
        episode_reward = update_metrics.get('episode_reward', 0)
        episode_length = update_metrics.get('episode_length', 0)
        
        # Handle both JAX arrays and Python scalars
        if hasattr(episode_reward, 'item'):
            episode_reward = episode_reward.item()
        if hasattr(episode_length, 'item'):
            episode_length = episode_length.item()
            
        episode_rewards = [episode_reward]
        episode_lengths = [episode_length]
        
        # Update agent state
        self.agent.network_params = new_network_params
        self.agent.optimizer_state = new_optimizer_state
        
        # Update episode and step counts
        self.episode_count += 1
        self.step_count += episode_length
        
        # Extract episode statistics from update_metrics
        episode_reward = update_metrics.get('episode_reward', 0)
        episode_length = update_metrics.get('episode_length', 0)
        
        # Handle both JAX arrays and Python scalars
        if hasattr(episode_reward, 'item'):
            episode_reward = episode_reward.item()
        if hasattr(episode_length, 'item'):
            episode_length = episode_length.item()
            
        episode_rewards = [episode_reward]
        episode_lengths = [episode_length]
        
        # Update episode and step counts
        self.episode_count += 1
        self.step_count += episode_length
        
        # Store episode metrics
        self.episode_rewards.extend(episode_rewards)
        self.episode_lengths.extend(episode_lengths)
        
        # Store loss metrics
        for key, value in update_metrics.items():
            if 'loss' in key:
                self.losses[key].append(value)
        
        step_metrics = {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'avg_reward': np.mean(episode_rewards),
            'avg_length': np.mean(episode_lengths),
            'total_steps': episode_lengths[0],
            **update_metrics
        }
        
        return step_metrics
    
    def train(self, callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict[str, Any]:
        """
        Main training loop.
        
        Args:
            callback: Optional callback function called after each episode
            
        Returns:
            Training results and metrics
        """
        start_time = time.time()
        
        # Start MLflow run
        if self.mlflow_logger:
            run_name = f"ppo_run_{int(start_time)}"
            self.mlflow_logger.start_run(run_name)
            
            # Log hyperparameters
            hyperparams = {
                'max_episodes': self.max_episodes,
                'max_steps_per_episode': self.max_steps_per_episode,
                'target_reward': self.target_reward,
                'convergence_window': self.convergence_window,
                'early_stopping_patience': self.early_stopping_patience,
                'eval_frequency': self.eval_frequency,
                'eval_episodes': self.eval_episodes,
                'learning_rate': self.agent.learning_rate,
                'clip_epsilon': self.agent.clip_epsilon,
                'gamma': self.agent.gamma,
                'gae_lambda': self.agent.gae_lambda,
                'entropy_coef': self.agent.entropy_coef,
                'value_coef': self.agent.value_coef,
                'batch_size': self.agent.batch_size,
                'epochs_per_update': self.agent.epochs_per_update,
            }
            self.mlflow_logger.log_hyperparameters(hyperparams)
        
        print(f"Starting training for {self.max_episodes} episodes...")
        print(f"Target reward: {self.target_reward}")
        print(f"Convergence window: {self.convergence_window} episodes")
        if self.mlflow_logger:
            print(f"MLflow experiment: {self.mlflow_experiment_name}")
        
        for episode in range(self.max_episodes):
            if self.should_stop:
                print("Early stopping triggered!")
                break
            
            # Perform training step
            self.key, step_key = jax.random.split(self.key)
            step_metrics = self.train_step(step_key)
            
            # Update episode count and metrics
            self.episode_count += len(step_metrics['episode_rewards'])
            self.step_count += step_metrics['total_steps']
            
            # Store episode metrics
            self.episode_rewards.extend(step_metrics['episode_rewards'])
            self.episode_lengths.extend(step_metrics['episode_lengths'])
            
            # Store loss metrics
            for key, value in step_metrics.items():
                if 'loss' in key:
                    self.losses[key].append(value)
            
            # Logging
            if episode % self.log_frequency == 0:
                avg_reward = step_metrics['avg_reward']
                avg_length = step_metrics['avg_length']
                policy_loss = step_metrics.get('policy_loss', 0)
                value_loss = step_metrics.get('value_loss', 0)
                total_loss = step_metrics.get('total_loss', 0)
                
                print(f"Episode {episode:4d} | "
                      f"Avg Reward: {avg_reward:6.2f} | "
                      f"Avg Length: {avg_length:5.1f} | "
                      f"Policy Loss: {policy_loss:6.4f} | "
                      f"Value Loss: {value_loss:6.4f} | "
                      f"Total Loss: {total_loss:6.4f}")
                
                # Log to MLflow
                if self.mlflow_logger:
                    self.mlflow_logger.log_training_metrics(step_metrics, episode)
                    self.mlflow_logger.log_episode_data(self.episode_rewards, self.episode_lengths, episode)
            
            # Evaluation
            if episode % self.eval_frequency == 0 and episode > 0:
                eval_metrics = self.evaluate(self.eval_episodes)
                self.eval_rewards.append(eval_metrics['avg_reward'])
                
                print(f"Evaluation | "
                      f"Avg Reward: {eval_metrics['avg_reward']:6.2f} ± {eval_metrics['std_reward']:6.2f} | "
                      f"Max Reward: {eval_metrics['max_reward']:6.2f}")
                
                # Log evaluation metrics to MLflow
                if self.mlflow_logger:
                    self.mlflow_logger.log_evaluation_metrics(eval_metrics, episode)
            
            # Check convergence
            if self.check_convergence():
                self.should_stop = True
                print(f"Convergence achieved! Average reward: {self.best_avg_reward:.2f}")
            
            # Callback
            if callback is not None:
                callback({
                    'episode': episode,
                    'step_metrics': step_metrics,
                    'eval_metrics': self.evaluate(self.eval_episodes) if episode % self.eval_frequency == 0 and episode > 0 else None
                })
        
        # Final evaluation
        final_eval = self.evaluate(self.eval_episodes)
        
        training_time = time.time() - start_time
        
        results = {
            'total_episodes': self.episode_count,
            'total_steps': self.step_count,
            'training_time': training_time,
            'final_avg_reward': final_eval['avg_reward'],
            'final_std_reward': final_eval['std_reward'],
            'best_avg_reward': self.best_avg_reward,
            'converged': self.best_avg_reward >= self.target_reward,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'eval_rewards': self.eval_rewards,
            'losses': dict(self.losses)
        }
        
        # Log final results to MLflow
        if self.mlflow_logger:
            # Log final metrics
            final_metrics = {
                'final_avg_reward': final_eval['avg_reward'],
                'final_std_reward': final_eval['std_reward'],
                'best_avg_reward': self.best_avg_reward,
                'total_episodes': self.episode_count,
                'total_steps': self.step_count,
                'training_time': training_time,
                'converged': self.best_avg_reward >= self.target_reward,
            }
            self.mlflow_logger.log_metrics(final_metrics, step=self.episode_count)
            
            # Log training curves and model
            self.mlflow_logger.log_training_curves(self.episode_rewards, self.losses)
            self.mlflow_logger.log_model(self.agent.network_params, "final_model")
            self.mlflow_logger.create_dashboard_data(results)
            
            # Register model if it converged
            if results['converged']:
                model_uri = f"runs:/{self.mlflow_logger.run_id}/final_model"
                self.mlflow_logger.register_model(model_uri, "cartpole-ppo-model", "Staging")
                print("Model registered to MLflow Model Registry")
            
            # End MLflow run
            self.mlflow_logger.end_run()
        
        print(f"\nTraining completed!")
        print(f"Total episodes: {results['total_episodes']}")
        print(f"Total steps: {results['total_steps']}")
        print(f"Training time: {training_time:.2f} seconds")
        print(f"Final evaluation reward: {results['final_avg_reward']:.2f} ± {results['final_std_reward']:.2f}")
        print(f"Best average reward: {results['best_avg_reward']:.2f}")
        print(f"Converged: {results['converged']}")
        if self.mlflow_logger:
            print(f"MLflow run completed: {self.mlflow_logger.run_id}")
        
        return results