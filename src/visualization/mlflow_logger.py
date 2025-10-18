"""MLflow integration for experiment tracking and model registry"""

from typing import Dict, Any, Optional, List
import os
import json
import tempfile
from pathlib import Path

import mlflow  # type: ignore


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
        self.backend_store_uri = os.getenv("MLFLOW_BACKEND_STORE_URI", "sqlite:///data/mlflow.db")
        self.run_id = None
        
        # Set tracking URI and backend store
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Create or get experiment
        try:
            self.experiment = mlflow.get_experiment_by_name(experiment_name)
            if self.experiment is None:
                self.experiment_id = mlflow.create_experiment(
                    name=experiment_name,
                    artifact_location="./data/ml_artifacts"
                )
            else:
                self.experiment_id = self.experiment.experiment_id
        except Exception as e:
            print(f"Warning: Could not connect to MLflow server at {self.tracking_uri}")
            print(f"Backend store: {self.backend_store_uri}")
            print(f"Error: {e}")
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
        try:
            # Log metrics with step parameter
            mlflow.log_metrics(metrics, step=step)
            
            # In MLflow 3.1.3, metrics can be linked to models and datasets
            # For now, we'll store the linking information as tags
            if model_id:
                mlflow.set_tag(f"step_{step}_model_id", model_id)
            if dataset:
                mlflow.set_tag(f"step_{step}_dataset", str(dataset))
        except Exception as e:
            print(f"Warning: Could not log metrics: {e}")
    
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
            # Manual logging since mlflow.flax doesn't exist
            import tempfile
            import json
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
                
                # Log artifacts using the mlflow instance
                mlflow.log_artifact(model_state_path, artifact_path)
                mlflow.log_artifact(metadata_path, artifact_path)
                
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
        # Convert JAX arrays and other types to Python scalars
        processed_metrics = {}
        for key, value in metrics.items():
            if hasattr(value, 'item'):
                processed_metrics[key] = float(value.item())
            elif isinstance(value, (int, float)):
                processed_metrics[key] = float(value)
            elif isinstance(value, (list, tuple)) and len(value) > 0:
                # Handle lists of values (e.g., episode rewards)
                if hasattr(value[0], 'item'):
                    processed_metrics[key] = float(value[0].item())
                else:
                    processed_metrics[key] = float(value[0])
            else:
                # Skip non-numeric values
                continue
        
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
        Log episode-specific metrics
        
        Args:
            episode_rewards: List of episode rewards
            episode_lengths: List of episode lengths
            step: Current episode number
        """
        if len(episode_rewards) == 0:
            return
            
        # Calculate averages manually to avoid numpy dependency issues
        avg_reward_10 = sum(episode_rewards[-10:]) / min(10, len(episode_rewards))
        if len(episode_rewards) >= 50:
            avg_reward_50 = sum(episode_rewards[-50:]) / 50
        else:
            avg_reward_50 = sum(episode_rewards) / len(episode_rewards)
        avg_length_10 = sum(episode_lengths[-10:]) / min(10, len(episode_lengths))
        
        metrics = {
            'episode_reward': episode_rewards[-1],
            'episode_length': episode_lengths[-1],
            'avg_reward_10': float(avg_reward_10),
            'avg_reward_50': float(avg_reward_50),
            'avg_length_10': float(avg_length_10),
        }
        
        self.log_metrics(metrics, step=step)
    
    def log_evaluation_metrics(self, eval_metrics: Dict[str, float], step: int,
                              model_id: Optional[str] = None, dataset: Optional[Any] = None) -> None:
        """
        Log evaluation metrics with optional model linking
        
        Args:
            eval_metrics: Evaluation metrics dictionary
            step: Current step/episode number
            model_id: Optional model ID to link metrics to
            dataset: Optional dataset to link metrics to
        """
        # Prefix evaluation metrics to distinguish from training metrics
        prefixed_metrics = {f"eval_{key}": value for key, value in eval_metrics.items()}
        self.log_metrics(prefixed_metrics, step=step, model_id=model_id, dataset=dataset)
    
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
        Log training curves as artifacts
        
        Args:
            episode_rewards: List of episode rewards
            losses: Dictionary of loss values over time
        """
        try:
            # Create temporary directory for artifacts
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Save episode rewards
                rewards_file = temp_path / "episode_rewards.json"
                with open(rewards_file, 'w') as f:
                    json.dump(episode_rewards, f, indent=2)
                
                # Save losses
                losses_file = temp_path / "losses.json"
                with open(losses_file, 'w') as f:
                    # Convert any JAX arrays to Python lists
                    processed_losses = {}
                    for key, values in losses.items():
                        if isinstance(values, list):
                            processed_losses[key] = [float(v) if hasattr(v, 'item') else v for v in values]
                        else:
                            processed_losses[key] = values
                    json.dump(processed_losses, f, indent=2)
                
                # Log artifacts
                self.log_artifact(str(rewards_file), "training_curves")
                self.log_artifact(str(losses_file), "training_curves")
                
        except Exception as e:
            print(f"Warning: Could not log training curves: {e}")
    
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
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Create comprehensive training summary
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
                        'target_reward': 195.0,  # CartPole target
                        'achieved': training_results.get('best_avg_reward', 0) >= 195.0,
                        'best_reward': training_results.get('best_avg_reward', 0),
                    }
                }
                
                # Save summary
                summary_file = temp_path / "training_summary.json"
                with open(summary_file, 'w') as f:
                    json.dump(summary, f, indent=2)
                
                # Log summary as artifact
                self.log_artifact(str(summary_file), "dashboard")
                
        except Exception as e:
            print(f"Warning: Could not create dashboard data: {e}")
    
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
            
            # Transition to the specified stage
            mlflow.transition_model_version_stage(
                name=name,
                version=model_version.version,
                stage=version
            )
            
            return model_version.version
        except Exception as e:
            print(f"Warning: Could not register model: {e}")
        return None


def setup_mlflow_experiment(experiment_name: str = "cartpole-ppo") -> MLflowLogger:
    """
    Set up MLflow experiment tracking
    
    Args:
        experiment_name: Name of the experiment
        
    Returns:
        Configured MLflowLogger instance
    """
    return MLflowLogger(experiment_name)