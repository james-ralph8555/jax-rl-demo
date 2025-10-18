# Agent Instructions for CartPole PPO Project

## Development Environment Setup

This project uses Nix flakes for reproducible dependency management. Before running any Python commands or tests, you must enter the development environment:

```bash
# Default environment (without MLflow server)
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
