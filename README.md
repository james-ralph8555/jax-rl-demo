# CartPole PPO with JAX

A Proximal Policy Optimization (PPO) implementation for the CartPole environment using JAX and Flax, with MLflow integration for experiment tracking.

## Quick Start

```bash
# Clone and enter the project
cd cartpole-ppo-jax

# Enter development environment
nix develop

# Start MLflow server in background
just start-mlflow-bg

# Run training with MLflow tracking
MLFLOW_TRACKING_URI=http://localhost:5000 python scripts/demo_training.py

# View results at http://localhost:5000

# Stop MLflow server when done
just stop-mlflow
```

## Development Environment

This project uses Nix flakes for reproducible dependency management and `just` for task automation.

```bash
# Enter development environment
nix develop

# See all available commands
just --list
```

## Server Management

The project uses `just` commands to manage MLflow servers:

```bash
# Start MLflow tracking server (foreground)
just start-mlflow

# Start MLflow tracking server (background)
just start-mlflow-bg

# Start MLflow MCP server (background)
just start-mcp-bg

# Start both servers (background)
just start-all

# Stop servers
just stop-mlflow
just stop-mcp
just stop-all

# Check server status
just status

# Clean up PID files and logs
just clean
```

## Testing

```bash
# Run all tests
just test

# Run specific test file
just test-file test_environment.py
```

## Project Structure

- `src/`: Source code
  - `environment/`: Environment wrappers and utilities
  - `agent/`: PPO agent implementation
  - `training/`: Training loop and configuration
  - `visualization/`: MLflow integration and plots
- `tests/`: Unit and integration tests
- `scripts/`: Training, evaluation, and visualization scripts
- `justfile`: Task automation commands

## MLflow Integration

The project includes MLflow for experiment tracking:
- Automatic logging of hyperparameters and metrics
- Model checkpointing
- Training visualization
- Artifact storage

Use `uv run --with mlflow` for any MLflow commands, or the provided `just` recipes.

See [MLFLOW_SETUP.md](MLFLOW_SETUP.md) for detailed setup instructions.