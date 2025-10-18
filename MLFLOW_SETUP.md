# MLflow Integration Setup

This project now uses MLflow for experiment tracking and model management. Here's how to get started:

## Automatic Setup (Recommended)

When you enter the development shell with `nix develop`, the MLflow server will automatically start in the background:

```bash
nix develop
```

The server will:
- Start on `http://localhost:5000`
- Use SQLite database at `./mlflow.db` for metadata
- Store artifacts in `./mlartifacts/`
- Run in the background with PID stored in `.mlflow_server.pid`

## Manual Setup

If you need to set up the MLflow environment manually:

1. **Setup virtual environment:**
   ```bash
   ./setup_mlflow.sh
   ```

2. **Start server manually:**
   ```bash
   source .venv/bin/activate
   mlflow server \
     --host 0.0.0.0 \
     --port 5000 \
     --backend-store-uri sqlite:///mlflow.db \
     --default-artifact-root ./mlartifacts \
     --serve-artifacts
   ```

## Cleanup

To stop the MLflow server and clean up:

```bash
./cleanup_mlflow.sh
```

## Directory Structure

- `.venv/` - Virtual environment for MLflow server
- `mlruns/` - MLflow experiment data (auto-created)
- `mlartifacts/` - Model artifacts and logs
- `mlflow.db` - SQLite database for experiment metadata

## Usage in Training

The training scripts now automatically log to MLflow:
- Hyperparameters and metrics
- Model checkpoints
- Training curves
- Evaluation results

View results at: http://localhost:5000