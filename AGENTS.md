# Agent Instructions for CartPole PPO Project

## Development Environment Setup

This project uses Nix flakes for reproducible dependency management. Before running any Python commands or tests, you must enter the development environment:

```bash
nix develop
```

Or run commands directly in the development environment:
```bash
nix develop --command <command>
```

## Testing Commands

To run tests, use:
```bash
nix develop --command python -m pytest tests/ -v
```

To run specific test files:
```bash
nix develop --command python -m pytest tests/test_environment.py -v
```

## Python Commands

Always use the nix development environment for Python commands:
```bash
nix develop --command python <script.py>
nix develop --command python -c "import jax; print('JAX version:', jax.__version__)"
```

## Environment Variables

The development shell automatically sets:
- `PYTHONPATH="${builtins.toString ./.}/src:$PYTHONPATH"`
- `MLFLOW_TRACKING_URI="http://localhost:5000"`

## Project Structure

- `src/`: Source code
  - `environment/`: Environment wrappers and utilities
  - `agent/`: PPO agent implementation
  - `training/`: Training loop and configuration
  - `visualization/`: MLflow integration and plots
- `tests/`: Unit and integration tests
- `scripts/`: Training, evaluation, and visualization scripts

## Implementation Status

- Phase 1: Environment Setup and Dependencies - ✅ COMPLETE
- Phase 2: Core Components Implementation
  - Step 1: Environment Wrapper - ✅ COMPLETE
    - ✅ CartPole environment wrapper using Gymnasium
    - ✅ Observation preprocessing and normalization utilities
    - ✅ Episode management functions
    - ✅ Comprehensive test suite (unit + integration tests)
  - Step 2: Neural Network Architecture - ⏳ TODO
  - Step 3: PPO Algorithm - ⏳ TODO