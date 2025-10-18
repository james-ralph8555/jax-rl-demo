# CartPole PPO with JAX

A reinforcement learning implementation of Proximal Policy Optimization (PPO) for CartPole using JAX, with MLflow experiment tracking.

## Quick Start

```bash
# Enter development environment
nix develop

# Start MLflow tracking server
just start-mlflow

# Train a model
python scripts/train.py --episodes 1000

# Evaluate the trained model
python scripts/evaluate.py --checkpoint models/best_model.pkl
```

## Development

```bash
# Run tests
just test

# Check MLflow server status
just status

# Stop MLflow server
just stop-mlflow

# Clean up artifacts
just clean
```

## Project Structure

- `src/` - Core implementation (agents, environment, training, visualization)
- `scripts/` - Command-line tools for training and evaluation
- `tests/` - Unit and integration tests
- `data/` - MLflow artifacts and logs

## Documentation

- [Scripts usage](scripts/README.md) - Detailed training and evaluation options
- [Agent documentation](AGENTS.md) - PPO implementation details
- [Agent module](src/agent/README.md) - PPO agent implementation with JAX and Flax
- [Environment module](src/environment/README.md) - CartPole environment wrapper and utilities
- [Training module](src/training/README.md) - Training infrastructure and PPO trainer
- [Visualization module](src/visualization/README.md) - MLflow integration and plotting utilities