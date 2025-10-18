"""MLflow integration for experiment tracking and model registry."""

from collections.abc import Mapping, Sequence
from typing import Dict, Any, Optional, List
import os
import json
import tempfile
from pathlib import Path

import mlflow  # type: ignore
from mlflow.exceptions import MlflowException  # type: ignore
import numpy as np
from matplotlib import pyplot as plt  # type: ignore


def _to_float(value: Any) -> float:
    """Convert scalar-like values (incl. JAX/NumPy types) to Python float."""
    if value is None:
        raise ValueError("Cannot convert None to float")
    array = np.asarray(value)
    if array.ndim == 0 or array.size == 1:
        return float(array.reshape(()))
    raise ValueError(f"Expected scalar value, got shape {array.shape}")


def _to_float_sequence(values: Sequence[Any]) -> List[float]:
    """Convert a sequence of scalar-likes to floats, ignoring non-scalars."""
    return [_to_float(v) for v in values if np.asarray(v).size == 1]


def _flatten_metrics(metrics: Mapping[str, Any]) -> Dict[str, float]:
    """Return MLflow-friendly metrics dict with scalar floats only."""
    flattened: Dict[str, float] = {}
    for key, value in metrics.items():
        if value is None:
            continue
        if isinstance(value, Mapping):
            continue
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if not value:
                continue
            try:
                flattened[key] = _to_float(value[-1])
            except ValueError:
                continue
            continue
        try:
            flattened[key] = _to_float(value)
        except ValueError:
            continue
    return flattened


class MLflowLogger:
    """MLflow logger for tracking experiments and managing model registry"""
    
    def __init__(self, experiment_name: str = "cartpole-ppo", tracking_uri: Optional[str] = None):
        """
        Initialize MLflow logger
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI (defaults to localhost:5000)
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        self.run_id = None
        
        artifact_location = os.getenv("MLFLOW_ARTIFACT_LOCATION", "./data/ml_artifacts")
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                self.experiment_id = mlflow.create_experiment(
                    experiment_name,
                    artifact_location=artifact_location,
                )
            else:
                self.experiment_id = experiment.experiment_id
        except Exception as error:
            print(f"Warning: Could not configure MLflow experiment '{experiment_name}': {error}")
            self.experiment_id = None
    
    def start_run(self, run_name: Optional[str] = None) -> Optional[str]:
        """
        Start a new MLflow run
        
        Args:
            run_name: Name for the run
            
        Returns:
            Run ID if successful, None otherwise
        """
        if self.experiment_id is None:
            return None
            
        try:
            run = mlflow.start_run(
                experiment_id=self.experiment_id,
                run_name=run_name
            )
            self.run_id = run.info.run_id
            return self.run_id
        except Exception as e:
            print(f"Warning: Could not start MLflow run: {e}")
        return None
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters"""
        try:
            mlflow.log_params(params)
        except Exception as e:
            print(f"Warning: Could not log parameters: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None, 
                   model_id: Optional[str] = None, dataset: Optional[Any] = None) -> None:
        """
        Log metrics with optional model and dataset linking (MLflow 3.1.3)
        
        Args:
            metrics: Dictionary of metrics to log
            step: Training step
            model_id: Optional model ID to link metrics to
            dataset: Optional dataset to link metrics to
        """
        prepared = _flatten_metrics(metrics)
        if not prepared:
            return

        try:
            mlflow.log_metrics(prepared, step=step)
            if model_id is not None:
                mlflow.set_tag(f"step_{step}_model_id", model_id)
            if dataset is not None:
                mlflow.set_tag(f"step_{step}_dataset", str(dataset))
        except MlflowException as error:
            print(f"Warning: Could not log metrics: {error}")
    
    def log_model(self, model, artifact_path: str = "model", step: Optional[int] = None,
                 input_example: Optional[Any] = None) -> Optional[str]:
        """
        Log a Flax model to MLflow with MLflow 3.1.3 model URI format
        
        Args:
            model: The Flax model to log
            artifact_path: Path for the model artifact
            step: Training step for model checkpointing
            input_example: Optional input example for the model
            
        Returns:
            Model ID if successful, None otherwise
        """
        try:
            import pickle

            with tempfile.TemporaryDirectory() as temp_dir:
                # Save model state dict using pickle for better serialization
                model_state_path = f"{temp_dir}/model_state.pkl"
                with open(model_state_path, 'wb') as f:
                    pickle.dump(model, f)
                
                # Save model metadata
                metadata = {
                    "model_type": "flax",
                    "step": step,
                    "artifact_path": artifact_path
                }
                metadata_path = f"{temp_dir}/metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                mlflow.log_artifacts(temp_dir, artifact_path)
                
                # Generate model ID following MLflow conventions
                model_id = f"{self.run_id}_{artifact_path}_{step if step is not None else 'final'}"
                
                # Store model info as a tag for reference
                mlflow.set_tag(f"model_{artifact_path}_id", model_id)
                
                return model_id
        except Exception as e:
            print(f"Warning: Could not log model: {e}")
        return None
    
    def log_training_metrics(self, metrics: Dict[str, Any], step: int, 
                           model_id: Optional[str] = None, dataset: Optional[Any] = None) -> None:
        """
        Log training metrics with proper handling of JAX arrays and model linking
        
        Args:
            metrics: Dictionary of metrics to log
            step: Training step/episode number
            model_id: Optional model ID to link metrics to
            dataset: Optional dataset to link metrics to
        """
        processed_metrics = _flatten_metrics(metrics)
        self.log_metrics(processed_metrics, step=step, model_id=model_id, dataset=dataset)
    
    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        """
        Log hyperparameters with proper type handling
        
        Args:
            params: Dictionary of hyperparameters
        """
        # Convert all parameters to strings for MLflow compatibility
        processed_params = {}
        for key, value in params.items():
            if isinstance(value, (list, tuple)):
                processed_params[key] = str(value)
            else:
                processed_params[key] = str(value)
        
        self.log_params(processed_params)
    
    def log_episode_data(self, episode_rewards: List[float], episode_lengths: List[int], step: int) -> None:
        """
        Log episode-specific metrics with visualizations
        
        Args:
            episode_rewards: List of episode rewards
            episode_lengths: List of episode lengths
            step: Current episode number
        """
        rewards = _to_float_sequence(episode_rewards)
        if not rewards:
            return
        lengths = _to_float_sequence(episode_lengths)
        int_lengths = [int(l) for l in lengths]
        if not lengths:
            return

        avg_reward_10 = float(np.mean(rewards[-10:]))
        avg_reward_50 = float(np.mean(rewards[-50:]))
        avg_length_10 = float(np.mean(lengths[-10:]))
        
        metrics = {
            'episode_reward': rewards[-1],
            'episode_length': lengths[-1],
            'avg_reward_10': avg_reward_10,
            'avg_reward_50': avg_reward_50,
            'avg_length_10': avg_length_10,
        }
        
        self.log_metrics(metrics, step=step)
        
        # Log episode progress visualization every 50 episodes
        if step % 50 == 0 and len(rewards) > 10:
            self.log_episode_progress(rewards, int_lengths, step)
    
    def log_episode_progress(self, episode_rewards: List[float], episode_lengths: List[int], step: int) -> None:
        """
        Log episode progress visualization using native mlflow.log_figure
        
        Args:
            episode_rewards: List of episode rewards
            episode_lengths: List of episode lengths
            step: Current episode number
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle(f'Episode Progress - Step {step}', fontsize=16)
            
            # Plot rewards with moving average
            axes[0, 0].plot(_to_float_sequence(episode_rewards), alpha=0.6, color='blue', label='Episode Reward')
            if len(episode_rewards) >= 10:
                window_size = max(1, min(50, len(episode_rewards) // 5))
                moving_avg = np.convolve(episode_rewards, np.ones(window_size) / window_size, mode='valid')
                axes[0, 0].plot(range(window_size-1, len(episode_rewards)), moving_avg, 
                              color='red', linewidth=2, label=f'MA {window_size}')
            axes[0, 0].axhline(y=195, color='green', linestyle='--', alpha=0.7, label='Target')
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot episode lengths
            axes[0, 1].plot(episode_lengths, alpha=0.6, color='orange')
            if len(episode_lengths) >= 10:
                window_size = min(50, len(episode_lengths) // 5)
                moving_avg_length = np.convolve(episode_lengths, np.ones(window_size)/window_size, mode='valid')
                axes[0, 1].plot(range(window_size-1, len(episode_lengths)), moving_avg_length, 
                              color='red', linewidth=2, label=f'MA {window_size}')
            axes[0, 1].set_title('Episode Lengths')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Steps')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot reward distribution (last 100 episodes)
            recent_rewards = episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards
            axes[1, 0].hist(recent_rewards, bins=20, alpha=0.7, edgecolor='black', color='skyblue')
            mean_recent = float(np.mean(recent_rewards)) if recent_rewards else 0.0
            axes[1, 0].axvline(x=mean_recent, color='red', linestyle='--', 
                              label=f'Mean: {mean_recent:.1f}')
            axes[1, 0].set_title(f'Reward Distribution (Last {len(recent_rewards)} episodes)')
            axes[1, 0].set_xlabel('Reward')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot performance metrics over time
            if len(episode_rewards) >= 20:
                window_sizes = [10, 50]
                colors = ['green', 'purple']
                for i, window in enumerate(window_sizes):
                    if len(episode_rewards) >= window:
                        rolling_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
                        axes[1, 1].plot(range(window-1, len(episode_rewards)), rolling_avg, 
                                      color=colors[i], linewidth=2, label=f'MA {window}')
                axes[1, 1].axhline(y=195, color='black', linestyle='--', alpha=0.7, label='Target')
                axes[1, 1].set_title('Performance Trends')
                axes[1, 1].set_xlabel('Episode')
                axes[1, 1].set_ylabel('Average Reward')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Log using mlflow.log_figure with step parameter
            mlflow.log_figure(fig, f"episode_progress/step_{step}.png")
            plt.close()
            
        except Exception as error:
            print(f"Warning: Could not log episode progress: {error}")
    
    def log_loss_curves(self, losses: Dict[str, List[float]], step: int) -> None:
        """
        Log loss curves visualization using native mlflow.log_figure
        
        Args:
            losses: Dictionary of loss values
            step: Current step
        """
        loss_keys = [key for key, values in losses.items() if 'loss' in key and values]
        if not loss_keys:
            return

        cols = min(3, len(loss_keys))
        rows = (len(loss_keys) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))

        axes_iter: List[Any]
        if rows == 1 and cols == 1:
            axes_iter = [axes]
        elif rows == 1:
            axes_iter = list(axes)
        else:
            axes_iter = [axes[row, col] for row in range(rows) for col in range(cols)]

        for idx, (loss_key, ax) in enumerate(zip(loss_keys, axes_iter)):
            series = _to_float_sequence(losses[loss_key])
            if not series:
                continue
            ax.plot(series, linewidth=1.5)
            ax.set_title(loss_key.replace('_', ' ').title())
            ax.set_xlabel('Update Step')
            ax.set_ylabel('Loss')
            ax.grid(True, alpha=0.3)
            if len(series) > 10:
                trend = np.poly1d(np.polyfit(range(len(series)), series, 1))
                ax.plot(range(len(series)), trend(range(len(series))), '--', alpha=0.5, color='red', label='Trend')
                ax.legend()

        for ax in axes_iter[len(loss_keys):]:
            ax.set_visible(False)

        fig.suptitle(f'Loss Curves - Step {step}', fontsize=16)
        plt.tight_layout()
        try:
            mlflow.log_figure(fig, f"loss_curves/step_{step}.png")
        except Exception as error:
            print(f"Warning: Could not log loss curves: {error}")
        finally:
            plt.close(fig)
    
    def log_evaluation_metrics(self, eval_metrics: Dict[str, float], step: int,
                              model_id: Optional[str] = None, dataset: Optional[Any] = None,
                              video_path: Optional[str] = None) -> None:
        """
        Log evaluation metrics with optional model linking and GIF
        
        Args:
            eval_metrics: Evaluation metrics dictionary
            step: Current step/episode number
            model_id: Optional model ID to link metrics to
            dataset: Optional dataset to link metrics to
            video_path: Optional path to evaluation GIF (kept for compatibility)
        """
        # Filter out None values from metrics to avoid conversion errors
        filtered_metrics = {key: value for key, value in eval_metrics.items() if value is not None}
        
        # Prefix evaluation metrics to distinguish from training metrics
        prefixed_metrics = {f"eval_{key}": value for key, value in filtered_metrics.items()}
        self.log_metrics(prefixed_metrics, step=step, model_id=model_id, dataset=dataset)
        
        # Log GIF if provided
        if video_path and os.path.exists(video_path):
            try:
                mlflow.log_artifact(video_path, f"evaluation_gifs/step_{step}")
            except Exception as e:
                print(f"Warning: Could not log evaluation GIF: {e}")
    
    def log_evaluation_gif(self, gif_path: str, step: int, episode_id: Optional[int] = None) -> None:
        """
        Log an evaluation GIF to MLflow
        
        Args:
            gif_path: Path to the GIF file
            step: Current training step
            episode_id: Optional episode identifier
        """
        if not os.path.exists(gif_path):
            print(f"Warning: GIF file not found: {gif_path}")
            return
        
        try:
            artifact_path = f"evaluation_gifs/step_{step}"
            if episode_id is not None:
                artifact_path += f"_episode_{episode_id}"
            
            mlflow.log_artifact(gif_path, artifact_path)
            print(f"Logged evaluation GIF: {gif_path}")
        except Exception as e:
            print(f"Warning: Could not log evaluation GIF: {e}")
    
    def log_evaluation_video(self, video_path: str, step: int, episode_id: Optional[int] = None) -> None:
        """
        Log an evaluation video to MLflow (deprecated, use log_evaluation_gif instead)
        
        Args:
            video_path: Path to the video file
            step: Current training step
            episode_id: Optional episode identifier
        """
        # For backward compatibility, treat as GIF
        self.log_evaluation_gif(video_path, step, episode_id)
    
    def log_incremental_analysis(self, episode_rewards: List[float], 
                                losses: Dict[str, List[float]], 
                                eval_rewards: Optional[List[float]] = None,
                                step: int = 0) -> None:
        """
        Log incremental analysis during training (more frequent than comprehensive analysis)
        
        Args:
            episode_rewards: List of episode rewards
            losses: Dictionary of loss values
            eval_rewards: Optional list of evaluation rewards
            step: Current step/episode number
        """
        rewards = _to_float_sequence(episode_rewards)
        if not rewards or len(rewards) < 10:
            return
        
        # Log incremental learning curves every 25 episodes
        if step % 25 == 0:
            self.log_advanced_learning_curves(rewards, losses, eval_rewards)
        
        # Log incremental stability analysis every 50 episodes  
        if step % 50 == 0:
            self.log_training_stability(rewards)
        
        # Log loss curves every 25 episodes
        if step % 25 == 0 and losses:
            self.log_loss_curves(losses, step)
    
    def log_model_checkpoint(self, model, step: int, metrics: Optional[Dict[str, float]] = None,
                           dataset: Optional[Any] = None) -> Optional[str]:
        """
        Log a model checkpoint with associated metrics (MLflow 3.1.3)
        
        Args:
            model: The model to log
            step: Training step
            metrics: Optional metrics to associate with this checkpoint
            dataset: Optional dataset to link to
            
        Returns:
            Model ID if successful, None otherwise
        """
        model_id = self.log_model(model, artifact_path=f"checkpoint_step_{step}", step=step)
        
        if metrics and model_id:
            self.log_metrics(metrics, step=step, model_id=model_id, dataset=dataset)
        
        return model_id
    
    def log_training_curves(self, episode_rewards: List[float], losses: Dict[str, List[float]]) -> None:
        """
        Log training curves as a matplotlib figure via MLflow.
        
        Args:
            episode_rewards: List of episode rewards
            losses: Dictionary of loss values over time
        """
        rewards = _to_float_sequence(episode_rewards)
        if not rewards:
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Training Progress', fontsize=16)

        axes[0, 0].plot(rewards)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True)

        if len(rewards) > 10:
            window_size = max(1, min(100, len(rewards) // 10))
            moving_avg = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
            axes[0, 1].plot(moving_avg)
            axes[0, 1].set_title(f'Moving Average Rewards (window={window_size})')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Average Reward')
            axes[0, 1].grid(True)
        else:
            axes[0, 1].set_visible(False)

        policy_series = _to_float_sequence(losses.get('policy_loss', []))
        if policy_series:
            axes[1, 0].plot(policy_series)
            axes[1, 0].set_title('Policy Loss')
            axes[1, 0].set_xlabel('Update Step')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True)
        else:
            axes[1, 0].set_visible(False)

        value_series = _to_float_sequence(losses.get('value_loss', []))
        if value_series:
            axes[1, 1].plot(value_series)
            axes[1, 1].set_title('Value Loss')
            axes[1, 1].set_xlabel('Update Step')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].grid(True)
        else:
            axes[1, 1].set_visible(False)

        plt.tight_layout()
        try:
            mlflow.log_figure(fig, "training_curves/training_progress.png")
        except Exception as error:
            print(f"Warning: Could not log training curves: {error}")
        finally:
            plt.close(fig)
    
    def setup_autologging(self) -> None:
        """Set up MLflow autologging for JAX/Flax"""
        try:
            # Manual logging since mlflow.flax doesn't exist
            print("Using manual logging for MLflow")
        except Exception as e:
            print(f"Warning: Could not set up autologging: {e}")
    
    def create_dashboard_data(self, training_results: Dict[str, Any]) -> None:
        """
        Create comprehensive dashboard data from training results
        
        Args:
            training_results: Dictionary containing all training results
        """
        summary = {
            'training_summary': {
                'total_episodes': training_results.get('total_episodes', 0),
                'total_steps': training_results.get('total_steps', 0),
                'training_time': training_results.get('training_time', 0),
                'final_avg_reward': training_results.get('final_avg_reward', 0),
                'final_std_reward': training_results.get('final_std_reward', 0),
                'best_avg_reward': training_results.get('best_avg_reward', 0),
                'converged': training_results.get('converged', False),
            },
            'final_evaluation': {
                'avg_reward': training_results.get('final_avg_reward', 0),
                'std_reward': training_results.get('final_std_reward', 0),
            },
            'convergence_info': {
                'target_reward': 195.0,
                'achieved': training_results.get('best_avg_reward', 0) >= 195.0,
                'best_reward': training_results.get('best_avg_reward', 0),
            },
        }

        try:
            mlflow.log_dict(summary, "dashboard/training_summary.json")
        except Exception as error:
            print(f"Warning: Could not create dashboard data: {error}")
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log a local file as an artifact"""
        try:
            mlflow.log_artifact(local_path, artifact_path)
        except Exception as e:
            print(f"Warning: Could not log artifact: {e}")
    
    def end_run(self) -> None:
        """End the current MLflow run"""
        try:
            mlflow.end_run()
        except Exception as e:
            print(f"Warning: Could not end run: {e}")
    
    def search_best_models(self, experiment_ids: Optional[List[str]] = None,
                          filter_string: str = "", max_results: int = 5,
                          order_by: Optional[List[Dict[str, Any]]] = None) -> List[Any]:
        """
        Search for best models using MLflow 3.1.3 search_logged_models API
        
        Args:
            experiment_ids: List of experiment IDs to search in
            filter_string: SQL-like filter string for searching models
            max_results: Maximum number of results to return
            order_by: List of ordering specifications
            
        Returns:
            List of model objects matching the search criteria
        """
        try:
            if experiment_ids is None and self.experiment_id:
                experiment_ids = [self.experiment_id]
            
            # Default ordering by accuracy if not specified
            if order_by is None:
                order_by = [{"field_name": "metrics.eval_avg_reward", "ascending": False}]
            
            # Use MLflow 3.1.3 search_logged_models API
            models = mlflow.search_logged_models(
                experiment_ids=experiment_ids,
                filter_string=filter_string,
                max_results=max_results,
                order_by=order_by,
                output_format="list"
            )
            
            return models
        except Exception as e:
            print(f"Warning: Could not search models: {e}")
            return []
    
    def get_model_uri(self, model_id: str) -> str:
        """
        Get MLflow 3.1.3 model URI format for a given model ID
        
        Args:
            model_id: The model ID
            
        Returns:
            Model URI in format models:/<model_id>
        """
        return f"models:/{model_id}"
    
    def register_model(self, model_uri: str, name: str, version: str = "Staging") -> Optional[str]:
        """
        Register a model in the MLflow Model Registry
        
        Args:
            model_uri: URI of the model to register (use get_model_uri() for MLflow 3.1.3 format)
            name: Name for the registered model
            version: Version stage (Staging, Production, Archived)
            
        Returns:
            Model version if successful, None otherwise
        """
        try:
            model_version = mlflow.register_model(model_uri, name)
            
            # Try to transition to the specified stage
            try:
                from mlflow.tracking import MlflowClient
                client = MlflowClient()
                client.transition_model_version_stage(
                    name=name,
                    version=model_version.version,
                    stage=version
                )
            except Exception as stage_error:
                print(f"Warning: Could not transition model stage: {stage_error}")
            
            return model_version.version
        except Exception as e:
            print(f"Warning: Could not register model: {e}")
        return None
    
    def log_policy_distribution(self, action_probs: List[Any], observations: List[Any], 
                               step: Optional[int] = None) -> None:
        """
        Log policy distribution visualization to MLflow using native mlflow.log_figure
        
        Args:
            action_probs: List of action probability arrays
            observations: List of observations
            step: Training step
        """
        if not action_probs:
            return

        action_probs_array = np.asarray(action_probs)
        if action_probs_array.ndim < 2:
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"Policy Distribution - Step {step if step is not None else 'Final'}", fontsize=16)

        axes[0, 0].plot(action_probs_array[:, 0], label='Action 0 (Left)', alpha=0.7)
        if action_probs_array.shape[1] > 1:
            axes[0, 0].plot(action_probs_array[:, 1], label='Action 1 (Right)', alpha=0.7)
        axes[0, 0].set_title('Action Probabilities Over Time')
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('Probability')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        axes[0, 0].set_ylim([0, 1])

        axes[0, 1].hist(action_probs_array[:, 0], bins=30, alpha=0.7, label='Action 0', edgecolor='black')
        if action_probs_array.shape[1] > 1:
            axes[0, 1].hist(action_probs_array[:, 1], bins=30, alpha=0.7, label='Action 1', edgecolor='black')
        axes[0, 1].set_title('Action Probability Distribution')
        axes[0, 1].set_xlabel('Probability')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        entropy = -np.sum(action_probs_array * np.log(action_probs_array + 1e-8), axis=1)
        axes[1, 0].plot(entropy, alpha=0.7)
        axes[1, 0].set_title('Policy Entropy Over Time')
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('Entropy')
        axes[1, 0].grid(True)

        confidence = np.max(action_probs_array, axis=1)
        axes[1, 1].plot(confidence, alpha=0.7, color='red')
        axes[1, 1].set_title('Policy Confidence Over Time')
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].set_ylabel('Max Probability')
        axes[1, 1].grid(True)
        axes[1, 1].set_ylim([0.5, 1.0])

        plt.tight_layout()
        artifact_file = f"policy_analysis/policy_distribution_step_{step if step is not None else 'final'}.png"
        try:
            mlflow.log_figure(fig, artifact_file)
        except Exception as error:
            print(f"Warning: Could not log policy distribution: {error}")
        finally:
            plt.close(fig)
    
    def log_advanced_learning_curves(self, episode_rewards: List[float], 
                                    losses: Dict[str, List[float]], 
                                    eval_rewards: Optional[List[float]] = None) -> None:
        """
        Log advanced learning curves to MLflow using native mlflow.log_figure
        
        Args:
            episode_rewards: List of episode rewards
            losses: Dictionary of loss values
            eval_rewards: Optional list of evaluation rewards
        """
        rewards = _to_float_sequence(episode_rewards)
        if not rewards:
            return

        loss_series = {
            key: _to_float_sequence(values)
            for key, values in losses.items()
            if 'loss' in key and values
        }

        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        ax_rewards = fig.add_subplot(gs[0, :])
        ax_rewards.plot(rewards, alpha=0.3, color='blue', label='Raw Rewards')
        ax_rewards.set_title('Episode Rewards Over Time', fontsize=14)
        ax_rewards.set_xlabel('Episode')
        ax_rewards.set_ylabel('Reward')
        ax_rewards.grid(True, alpha=0.3)

        for idx, window in enumerate((10, 50, 100)):
            if len(rewards) >= window:
                moving_avg = np.convolve(rewards, np.ones(window) / window, mode='valid')
                ax_rewards.plot(range(window - 1, len(rewards)), moving_avg,
                                linewidth=2, label=f'MA {window}')

        ax_rewards.axhline(y=195, color='black', linestyle='--', alpha=0.7, label='Target (195)')
        ax_rewards.legend()

        for idx, (loss_key, series) in enumerate(list(loss_series.items())[:3]):
            ax_loss = fig.add_subplot(gs[1, idx])
            ax_loss.plot(series, alpha=0.8)
            ax_loss.set_title(loss_key.replace('_', ' ').title())
            ax_loss.set_xlabel('Update Step')
            ax_loss.set_ylabel('Loss')
            ax_loss.grid(True, alpha=0.3)

        if eval_rewards:
            eval_series = _to_float_sequence(eval_rewards)
            if eval_series:
                ax_eval = fig.add_subplot(gs[2, :2])
                stride = max(1, len(rewards) // max(1, len(eval_series)))
                eval_episodes = list(range(0, stride * len(eval_series), stride))[:len(eval_series)]
                ax_eval.plot(eval_episodes, eval_series, 'o-', color='purple', linewidth=2, markersize=6)
                ax_eval.set_title('Evaluation Rewards', fontsize=14)
                ax_eval.set_xlabel('Episode')
                ax_eval.set_ylabel('Evaluation Reward')
                ax_eval.grid(True, alpha=0.3)
                ax_eval.axhline(y=195, color='black', linestyle='--', alpha=0.7, label='Target (195)')
                ax_eval.legend()

        ax_stats = fig.add_subplot(gs[2, 2])
        stats_text = [
            f"Total Episodes: {len(rewards)}",
            f"Final Reward: {rewards[-1]:.1f}",
            f"Best Reward: {max(rewards):.1f}",
            f"Mean Reward: {np.mean(rewards):.1f}",
            f"Std Reward: {np.std(rewards):.1f}",
        ]
        if len(rewards) >= 100:
            recent_avg = np.mean(rewards[-100:])
            stats_text.append(f"Last 100 Avg: {recent_avg:.1f}")
            stats_text.append(f"Converged: {recent_avg >= 195}")
        ax_stats.text(
            0.1,
            0.5,
            "\n".join(stats_text),
            transform=ax_stats.transAxes,
            fontsize=11,
            verticalalignment='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5),
        )
        ax_stats.set_title('Learning Statistics')
        ax_stats.axis('off')

        plt.suptitle('Advanced Learning Progress Analysis', fontsize=16, y=0.98)
        try:
            mlflow.log_figure(fig, "learning_analysis/advanced_learning_curves.png")
        except Exception as error:
            print(f"Warning: Could not log advanced learning curves: {error}")
        finally:
            plt.close(fig)
    
    def log_training_stability(self, episode_rewards: List[float], 
                              window_size: int = 50) -> None:
        """
        Log training stability analysis to MLflow using native mlflow.log_figure
        
        Args:
            episode_rewards: List of episode rewards
            window_size: Window size for rolling statistics
        """
        rewards = _to_float_sequence(episode_rewards)
        if len(rewards) < window_size:
            return

        rewards_array = np.asarray(rewards)
        indices = list(range(window_size - 1, len(rewards)))
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Training Stability Analysis', fontsize=16)

        rolling_mean = np.convolve(rewards_array, np.ones(window_size) / window_size, mode='valid')
        axes[0, 0].plot(indices, rolling_mean)
        axes[0, 0].set_title(f'Rolling Mean (window={window_size})')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Mean Reward')
        axes[0, 0].grid(True)
        axes[0, 0].axhline(y=195, color='red', linestyle='--', alpha=0.7)

        rolling_std = [float(np.std(rewards_array[i - window_size + 1:i + 1])) for i in indices]
        axes[0, 1].plot(indices, rolling_std, color='orange')
        axes[0, 1].set_title(f'Rolling Std Dev (window={window_size})')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Std Dev')
        axes[0, 1].grid(True)

        rolling_cv = []
        for i in indices:
            window_data = rewards_array[i - window_size + 1:i + 1]
            mean = window_data.mean()
            rolling_cv.append(float(window_data.std() / mean) if mean > 0 else 0.0)
        axes[1, 0].plot(indices, rolling_cv, color='green')
        axes[1, 0].set_title(f'Coefficient of Variation (window={window_size})')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('CV (Std/Mean)')
        axes[1, 0].grid(True)

        consistency = [
            float(np.sum(rewards_array[i - window_size + 1:i + 1] >= 195) / window_size * 100)
            for i in indices
        ]
        axes[1, 1].plot(indices, consistency, color='purple')
        axes[1, 1].set_title(f'Consistency (% episodes ≥ 195, window={window_size})')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Consistency (%)')
        axes[1, 1].grid(True)
        axes[1, 1].set_ylim([0, 100])

        plt.tight_layout()
        try:
            mlflow.log_figure(fig, "stability_analysis/training_stability.png")
        except Exception as error:
            print(f"Warning: Could not log training stability: {error}")
        finally:
            plt.close(fig)
    
    def log_hyperparameter_comparison(self, results: Dict[str, Dict[str, Any]], 
                                    metric: str = 'final_reward') -> None:
        """
        Log hyperparameter comparison heatmap to MLflow using native mlflow.log_figure
        
        Args:
            results: Dictionary of results for different hyperparameter combinations
            metric: Metric to visualize in the heatmap
        """
        if not results:
            return

        try:
            import seaborn as sns  # type: ignore
        except ImportError as error:
            print(f"Warning: Could not import seaborn for hyperparameter plotting: {error}")
            return

        param_names = {
            name
            for run in results.values()
            if isinstance(run, Mapping) and isinstance(run.get('hyperparams'), Mapping)
            for name in run['hyperparams']
        }
        main_params = list(param_names)[:2]
        if len(main_params) < 2:
            print("Need at least 2 hyperparameters for heatmap visualization")
            return

        param1_values = sorted({
            run['hyperparams'].get(main_params[0])
            for run in results.values()
            if isinstance(run, Mapping) and isinstance(run.get('hyperparams'), Mapping)
        })
        param2_values = sorted({
            run['hyperparams'].get(main_params[1])
            for run in results.values()
            if isinstance(run, Mapping) and isinstance(run.get('hyperparams'), Mapping)
        })

        heatmap_data = np.full((len(param1_values), len(param2_values)), np.nan)
        for run in results.values():
            if not (isinstance(run, Mapping) and isinstance(run.get('hyperparams'), Mapping)):
                continue
            if metric not in run:
                continue
            p1_val = run['hyperparams'].get(main_params[0])
            p2_val = run['hyperparams'].get(main_params[1])
            if p1_val in param1_values and p2_val in param2_values:
                i = param1_values.index(p1_val)
                j = param2_values.index(p2_val)
                heatmap_data[i, j] = _to_float(run[metric])

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            heatmap_data,
            xticklabels=[f"{v:.3f}" if isinstance(v, (int, float)) else str(v) for v in param2_values],
            yticklabels=[f"{v:.3f}" if isinstance(v, (int, float)) else str(v) for v in param1_values],
            annot=True,
            fmt=".1f",
            cmap="YlOrRd",
            cbar_kws={'label': metric},
            ax=ax,
        )
        ax.set_title(f'Hyperparameter Heatmap: {metric}')
        ax.set_xlabel(main_params[1])
        ax.set_ylabel(main_params[0])

        try:
            mlflow.log_figure(fig, f"hyperparameter_analysis/hyperparameter_heatmap_{metric}.png")
        except Exception as error:
            print(f"Warning: Could not log hyperparameter comparison: {error}")
        finally:
            plt.close(fig)
    
    def log_comprehensive_analysis(self, training_data: Dict[str, Any]) -> None:
        """
        Log comprehensive analysis plots to MLflow using native mlflow.log_figure
        
        Args:
            training_data: Dictionary containing all training data
        """
        rewards = training_data.get('episode_rewards', [])
        losses = training_data.get('losses', {})
        eval_rewards = training_data.get('eval_rewards', [])
        episode_lengths = training_data.get('episode_lengths', [])
        hyperparams = training_data.get('hyperparams', {})

        reward_series = _to_float_sequence(rewards)
        length_series = _to_float_sequence(episode_lengths) if episode_lengths else []

        if reward_series and length_series:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            fig.suptitle('Episode Statistics', fontsize=16)
            axes[0].plot(length_series)
            axes[0].set_title('Episode Lengths')
            axes[0].set_xlabel('Episode')
            axes[0].set_ylabel('Steps')
            axes[0].grid(True)
            axes[1].hist(reward_series, bins=30, alpha=0.7, edgecolor='black')
            axes[1].set_title('Reward Distribution')
            axes[1].set_xlabel('Reward')
            axes[1].set_ylabel('Frequency')
            axes[1].grid(True)
            plt.tight_layout()
            try:
                mlflow.log_figure(fig, "comprehensive_analysis/episode_statistics.png")
            except Exception as error:
                print(f"Warning: Could not log episode statistics: {error}")
            finally:
                plt.close(fig)

        if reward_series:
            final_losses: Dict[str, float] = {}
            for name, values in losses.items():
                sequence = _to_float_sequence(values if isinstance(values, Sequence) else [])
                if sequence:
                    final_losses[name] = sequence[-1]
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('Training Summary', fontsize=16)
            axes[0, 0].hist(reward_series, bins=20, alpha=0.7, edgecolor='black')
            axes[0, 0].set_title('Final Rewards Distribution')
            axes[0, 0].set_xlabel('Reward')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].grid(True)

            if final_losses:
                axes[0, 1].bar(final_losses.keys(), final_losses.values())
                axes[0, 1].set_title('Final Losses')
                axes[0, 1].set_ylabel('Loss Value')
                axes[0, 1].tick_params(axis='x', rotation=45)
            else:
                axes[0, 1].set_visible(False)

            if len(reward_series) > 100:
                axes[1, 0].plot(reward_series[-100:])
                axes[1, 0].set_title('Last 100 Episodes')
                axes[1, 0].set_xlabel('Episode')
                axes[1, 0].set_ylabel('Reward')
                axes[1, 0].grid(True)
            else:
                axes[1, 0].set_visible(False)

            if hyperparams:
                axes[1, 1].text(
                    0.1,
                    0.5,
                    "\n".join(f"{k}: {v}" for k, v in hyperparams.items()),
                    transform=axes[1, 1].transAxes,
                    fontsize=10,
                    verticalalignment='center',
                )
                axes[1, 1].set_title('Hyperparameters')
                axes[1, 1].axis('off')
            else:
                axes[1, 1].set_visible(False)

            plt.tight_layout()
            try:
                mlflow.log_figure(fig, "comprehensive_analysis/training_summary.png")
            except Exception as error:
                print(f"Warning: Could not log training summary: {error}")
            finally:
                plt.close(fig)

        # Delegate to specific visualizations for consistency.
        self.log_training_curves(rewards, losses)
        self.log_advanced_learning_curves(rewards, losses, eval_rewards or None)
        self.log_training_stability(rewards)
    
    def create_performance_report(self, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a detailed performance report with statistical analysis
        
        Args:
            training_results: Dictionary containing training results
            
        Returns:
            Performance report dictionary
        """
        episode_rewards = _to_float_sequence(training_results.get('episode_rewards', []))
        episode_lengths = _to_float_sequence(training_results.get('episode_lengths', []))
        losses = training_results.get('losses', {})
        
        if not episode_rewards:
            return {}

        rewards_array = np.asarray(episode_rewards)
        total_episodes = len(rewards_array)
        total_steps = training_results.get('total_steps', 0)
        training_time = training_results.get('training_time', 0)

        report = {
            'basic_stats': {
                'total_episodes': total_episodes,
                'total_steps': total_steps,
                'training_time': training_time,
                'final_reward': float(rewards_array[-1]),
                'best_reward': float(rewards_array.max()),
                'mean_reward': float(rewards_array.mean()),
                'std_reward': float(rewards_array.std()),
            },
            'convergence_analysis': {},
            'stability_analysis': {},
            'loss_analysis': {},
            'performance_benchmarks': {}
        }
        
        if total_episodes >= 100:
            recent_avg = float(rewards_array[-100:].mean())
            rolling_avg = np.array([
                rewards_array[max(0, i-99):i+1].mean()
                for i in range(99, total_episodes)
            ])
            report['convergence_analysis'] = {
                'last_100_avg': recent_avg,
                'converged': recent_avg >= 195.0,
                'convergence_episode': int(np.argmax(rolling_avg >= 195.0)) + 99
                if np.any(rolling_avg >= 195.0) else None,
            }
        
        if total_episodes >= 50:
            window_size = 50
            rolling_means = np.array([
                rewards_array[i - window_size + 1:i + 1].mean()
                for i in range(window_size - 1, total_episodes)
            ])
            if rolling_means.size:
                report['stability_analysis'] = {
                    'rolling_mean_final': float(rolling_means[-1]),
                    'rolling_mean_std': float(rolling_means.std()),
                    'consistency_195': float(np.mean(rolling_means >= 195.0) * 100),
                }
        
        for loss_name, loss_values in losses.items():
            series = _to_float_sequence(loss_values if isinstance(loss_values, Sequence) else [])
            if series:
                report['loss_analysis'][loss_name] = {
                    'final_loss': float(series[-1]),
                    'min_loss': float(min(series)),
                    'max_loss': float(max(series)),
                }
        
        report['performance_benchmarks'] = {
            'target_195': report['basic_stats']['mean_reward'] >= 195.0,
            'target_180': report['basic_stats']['mean_reward'] >= 180.0,
            'target_150': report['basic_stats']['mean_reward'] >= 150.0,
            'efficiency': (
                total_steps / report['basic_stats']['mean_reward']
                if report['basic_stats']['mean_reward'] > 0
                else 0.0
            ),
        }
        
        try:
            mlflow.log_dict(report, "performance_analysis/performance_report.json")
        except Exception as error:
            print(f"Warning: Could not log performance report: {error}")
        
        return report


def setup_mlflow_experiment(experiment_name: str = "cartpole-ppo") -> MLflowLogger:
    """
    Set up MLflow experiment tracking
    
    Args:
        experiment_name: Name of the experiment
        
    Returns:
        Configured MLflowLogger instance
    """
    return MLflowLogger(experiment_name)
