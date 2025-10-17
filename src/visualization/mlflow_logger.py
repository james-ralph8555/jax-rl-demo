"""MLflow integration for experiment tracking and model registry"""

from typing import Dict, Any, Optional
import os

# MLflow imports - these will be available when the environment is set up
try:
    import mlflow  # type: ignore
    import mlflow.flax  # type: ignore
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None
    print("Warning: MLflow not available. Install with: pip install mlflow")


class MLflowLogger:
    """MLflow logger for tracking experiments and managing model registry"""
    
    def __init__(self, experiment_name: str = "cartpole-ppo", tracking_uri: Optional[str] = None):
        """
        Initialize MLflow logger
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI (defaults to localhost:5000)
        """
        if not MLFLOW_AVAILABLE:
            print("MLflow not available - logging will be disabled")
            self.experiment_id = None
            return
            
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        
        # Set tracking URI
        if mlflow is not None:
            mlflow.set_tracking_uri(self.tracking_uri)
        
        # Create or get experiment
        try:
            if mlflow is not None:
                self.experiment = mlflow.get_experiment_by_name(experiment_name)
                if self.experiment is None:
                    self.experiment_id = mlflow.create_experiment(experiment_name)
                else:
                    self.experiment_id = self.experiment.experiment_id
        except Exception as e:
            print(f"Warning: Could not connect to MLflow server at {self.tracking_uri}")
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
        if not MLFLOW_AVAILABLE or self.experiment_id is None:
            return None
            
        try:
            if mlflow is not None:
                run = mlflow.start_run(
                    experiment_id=self.experiment_id,
                    run_name=run_name
                )
                return run.info.run_id
        except Exception as e:
            print(f"Warning: Could not start MLflow run: {e}")
        return None
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters"""
        if not MLFLOW_AVAILABLE:
            return
            
        try:
            if mlflow is not None:
                mlflow.log_params(params)
        except Exception as e:
            print(f"Warning: Could not log parameters: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics"""
        if not MLFLOW_AVAILABLE:
            return
            
        try:
            if mlflow is not None:
                mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            print(f"Warning: Could not log metrics: {e}")
    
    def log_model(self, model, artifact_path: str = "model") -> None:
        """Log a Flax model to MLflow"""
        if not MLFLOW_AVAILABLE:
            return
            
        try:
            if mlflow is not None and hasattr(mlflow, 'flax'):
                mlflow.flax.log_model(model, artifact_path)
        except Exception as e:
            print(f"Warning: Could not log model: {e}")
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log a local file as an artifact"""
        if not MLFLOW_AVAILABLE:
            return
            
        try:
            if mlflow is not None:
                mlflow.log_artifact(local_path, artifact_path)
        except Exception as e:
            print(f"Warning: Could not log artifact: {e}")
    
    def end_run(self) -> None:
        """End the current MLflow run"""
        if not MLFLOW_AVAILABLE:
            return
            
        try:
            if mlflow is not None:
                mlflow.end_run()
        except Exception as e:
            print(f"Warning: Could not end run: {e}")
    
    def register_model(self, model_uri: str, name: str, version: str = "Staging") -> Optional[str]:
        """
        Register a model in the MLflow Model Registry
        
        Args:
            model_uri: URI of the model to register
            name: Name for the registered model
            version: Version stage (Staging, Production, Archived)
            
        Returns:
            Model version if successful, None otherwise
        """
        if not MLFLOW_AVAILABLE:
            return None
            
        try:
            if mlflow is not None:
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