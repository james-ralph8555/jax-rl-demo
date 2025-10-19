"""Tests for MLflow gradient flow logging with nested mappings (Flax FrozenDict)."""

import os
from typing import Dict, Any

import jax.numpy as jnp
from flax.core.frozen_dict import freeze
import mlflow

from src.visualization.mlflow_logger import MLflowLogger


def _mock_grad_norms() -> Dict[str, Any]:
    """Create a nested gradient norms structure using Flax FrozenDict-compatible mapping."""
    nested = {
        "params": {
            "actor": {
                "Dense_0": {"kernel": jnp.array(0.11), "bias": jnp.array(0.02)},
                "Dense_1": {"kernel": jnp.array(0.22), "bias": jnp.array(0.03)},
            },
            "critic": {
                "Dense_0": {"kernel": jnp.array(0.33), "bias": jnp.array(0.04)},
                "Dense_1": {"kernel": jnp.array(0.44), "bias": jnp.array(0.05)},
            },
        }
    }
    return freeze(nested)


def test_log_gradient_flow_local_file_store(tmp_path):
    """Ensure gradient flow artifacts are logged when passing a Mapping (FrozenDict)."""
    # Use a local file-based tracking URI to avoid requiring a server
    tracking_uri = f"file:{tmp_path}/mlruns"
    logger = MLflowLogger(experiment_name="test-gradient-flow", tracking_uri=tracking_uri)

    run_id = logger.start_run("test-gradient-flow-run")
    assert run_id is not None

    try:
        grad_norms = _mock_grad_norms()
        logger.log_gradient_flow(grad_norms, step=50)

        # Resolve artifact path via MLflow API and verify it exists
        artifact_uri = mlflow.get_artifact_uri("gradient_flow/step_50.png")
        # mlflow may return file:// URI; handle both URI and plain path
        if artifact_uri.startswith("file:"):
            # Strip scheme and possible triple slashes
            path = artifact_uri.split(":", 1)[1]
            # On some platforms it may include '///'
            path = path.lstrip("/") if os.name == "nt" else path
        else:
            path = artifact_uri
        assert os.path.exists(path)
    finally:
        logger.end_run()

