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
        early_stopping_patience: int = 10,
        eval_frequency: int = 50,
        eval_episodes: int = 10,
        save_frequency: int = 100,
        log_frequency: int = 10,
        enable_mlflow: bool = True,
        mlflow_experiment_name: str = "cartpole-ppo",
        video_record_frequency: int = 200,  # Record GIF every N episodes
        log_gradient_flow: bool = False,
        key: Optional[jax.Array] = None
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
            video_record_frequency: Record GIF every N episodes
            log_gradient_flow: Whether to log gradient flow visualization
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
        self.video_record_frequency = video_record_frequency
        self.log_gradient_flow = log_gradient_flow
        
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
        self.gradient_norms = []
        self.kl_divergences = []
        
        # MLflow setup
        self.mlflow_logger = None
        if self.enable_mlflow:
            self.mlflow_logger = MLflowLogger(self.mlflow_experiment_name)
            self.mlflow_logger.setup_autologging()
        
    def collect_episode(self, key: jax.Array) -> Dict[str, jnp.ndarray]:
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
    
    def collect_batch(self, key: jax.Array, batch_size: int) -> Dict[str, jnp.ndarray]:
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
    
    def evaluate(self, num_episodes: int, record_gif: bool = False, episode_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate the current policy.
        
        Args:
            num_episodes: Number of evaluation episodes
            record_gif: Whether to record a GIF of the first episode
            episode_id: Episode ID for GIF naming
            
        Returns:
            Evaluation metrics dictionary
        """
        eval_rewards = []
        eval_lengths = []
        gif_path = None
        
        for ep in range(num_episodes):
            obs, info = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            # Record GIF for first episode if requested
            if record_gif and ep == 0 and hasattr(self.env, 'record_episode_gif'):
                gif_path = self.env.record_episode_gif(episode_id or 0, self.max_steps_per_episode)
            
            for step in range(self.max_steps_per_episode):
                # Select action deterministically via argmax over policy logits
                policy_logits, value = self.agent.network.apply(self.agent.network_params, obs[None, :])
                action = jnp.argmax(policy_logits, axis=-1)
                
                obs, reward, terminated, truncated, info = self.env.step(int(action.item()))
                episode_reward += reward
                episode_length += 1
                
                if terminated or truncated:
                    break
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
        
        result = {
            'avg_reward': float(np.mean(eval_rewards)),
            'std_reward': float(np.std(eval_rewards)),
            'max_reward': float(np.max(eval_rewards)),
            'min_reward': float(np.min(eval_rewards)),
            'avg_length': float(np.mean(eval_lengths)),
            'gif_path': gif_path
        }
        
        return result
    
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
    
    def train_step(self, key: jax.Array) -> Dict[str, Any]:
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

        # Persist updated parameters/optimizer state on the agent
        self.agent.network_params = new_network_params
        self.agent.optimizer_state = new_optimizer_state
        
        # Extract episode statistics from update_metrics
        episode_reward = update_metrics.get('episode_reward', 0)
        episode_length = update_metrics.get('episode_length', 0)
        kl_divergence = update_metrics.get('kl_divergence', 0)
        grad_norms = update_metrics.get('grad_norms', {})
        
        # Convert to Python scalars
        episode_reward = float(episode_reward) if not isinstance(episode_reward, float) else episode_reward
        episode_length = int(episode_length) if not isinstance(episode_length, int) else episode_length
        kl_divergence = float(kl_divergence) if not isinstance(kl_divergence, float) else kl_divergence
            
        episode_rewards = [episode_reward]
        episode_lengths = [episode_length]
        
        # Note: do not mutate trainer-wide counters or losses here.
        # These are handled in train() to avoid double counting.
        
        step_metrics = {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'avg_reward': float(np.mean(np.array(episode_rewards))),
            'avg_length': float(np.mean(np.array(episode_lengths))),
            'total_steps': episode_lengths[0],
            'kl_divergence': kl_divergence,
            'grad_norms': grad_norms,
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
            
            # Store loss metrics as python floats
            for key_name, value in step_metrics.items():
                if isinstance(key_name, str) and ('loss' in key_name or 'entropy' in key_name):
                    try:
                        v = float(value.item()) if hasattr(value, 'item') else float(value)
                        self.losses[key_name].append(v)
                    except (TypeError, ValueError):
                        pass
            
            # Store KL divergence and gradient norms
            if 'kl_divergence' in step_metrics:
                try:
                    kl_val = float(step_metrics['kl_divergence'].item()) if hasattr(step_metrics['kl_divergence'], 'item') else float(step_metrics['kl_divergence'])
                    self.kl_divergences.append(kl_val)
                except (TypeError, ValueError):
                    pass
            
            if self.log_gradient_flow and 'grad_norms' in step_metrics:
                self.gradient_norms.append(step_metrics['grad_norms'])
            
            # Logging
            if episode % self.log_frequency == 0:
                avg_reward = step_metrics['avg_reward']
                avg_length = step_metrics['avg_length']
                policy_loss = step_metrics.get('policy_loss', 0)
                value_loss = step_metrics.get('value_loss', 0)
                total_loss = step_metrics.get('total_loss', 0)
                kl_divergence = step_metrics.get('kl_divergence', 0)
                
                print(f"Episode {episode:4d} | "
                      f"Avg Reward: {avg_reward:6.2f} | "
                      f"Avg Length: {avg_length:5.1f} | "
                      f"Policy Loss: {policy_loss:6.4f} | "
                      f"Value Loss: {value_loss:6.4f} | "
                      f"Total Loss: {total_loss:6.4f} | "
                      f"KL Div: {kl_divergence:6.4f}")
                
                # Log to MLflow
                if self.mlflow_logger:
                    self.mlflow_logger.log_training_metrics(step_metrics, episode)
                    self.mlflow_logger.log_episode_data(self.episode_rewards, self.episode_lengths, episode)
                    
                    # Log incremental analysis every log_frequency
                    if episode % (self.log_frequency * 2) == 0:
                        self.mlflow_logger.log_incremental_analysis(
                            self.episode_rewards,
                            dict(self.losses),
                            self.eval_rewards,
                            episode
                        )
            
            # Log gradient flow and KL divergence every 50 episodes (like other visualizations)
            if episode % 50 == 0 and episode > 0:
                if self.mlflow_logger:
                    # Log gradient flow if enabled
                    if self.log_gradient_flow and self.gradient_norms:
                        self.mlflow_logger.log_gradient_flow(self.gradient_norms[-1], episode)
                    
                    # Log KL divergence if available
                    if self.kl_divergences:
                        self.mlflow_logger.log_kl_divergence(self.kl_divergences, episode)
            
            # Evaluation
            if episode % self.eval_frequency == 0 and episode > 0:
                # Record GIF for evaluation episodes
                record_gif = (episode % self.video_record_frequency == 0)
                eval_metrics = self.evaluate(self.eval_episodes, record_gif=record_gif, episode_id=episode)
                self.eval_rewards.append(eval_metrics['avg_reward'])
                
                print(f"Evaluation | "
                      f"Avg Reward: {eval_metrics['avg_reward']:6.2f} ± {eval_metrics['std_reward']:6.2f} | "
                      f"Max Reward: {eval_metrics['max_reward']:6.2f}")
                
                # Log evaluation metrics to MLflow
                if self.mlflow_logger:
                    self.mlflow_logger.log_evaluation_metrics(
                        eval_metrics, 
                        episode, 
                        video_path=eval_metrics.get('gif_path')
                    )
                    
                    # Log incremental analysis
                    self.mlflow_logger.log_incremental_analysis(
                        self.episode_rewards,
                        dict(self.losses),
                        self.eval_rewards,
                        episode
                    )
            
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
            
            # Log enhanced visualizations
            self.mlflow_logger.log_advanced_learning_curves(
                self.episode_rewards, 
                dict(self.losses), 
                self.eval_rewards
            )
            self.mlflow_logger.log_training_stability(self.episode_rewards)
            
            # Create comprehensive analysis
            training_data = {
                'episode_rewards': self.episode_rewards,
                'episode_lengths': self.episode_lengths,
                'losses': dict(self.losses),
                'eval_rewards': self.eval_rewards,
                'hyperparams': {
                    'learning_rate': self.agent.learning_rate,
                    'clip_epsilon': self.agent.clip_epsilon,
                    'gamma': self.agent.gamma,
                    'gae_lambda': self.agent.gae_lambda,
                    'entropy_coef': self.agent.entropy_coef,
                    'value_coef': self.agent.value_coef,
                    'batch_size': self.agent.batch_size,
                    'epochs_per_update': self.agent.epochs_per_update,
                }
            }
            self.mlflow_logger.log_comprehensive_analysis(training_data)
            
            # Create performance report
            performance_report = self.mlflow_logger.create_performance_report(results)
            
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
