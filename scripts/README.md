# CartPole PPO Scripts

This directory contains polished scripts for training and evaluating CartPole PPO agents with MLflow integration.

## Scripts

### `train.py`
Training script for CartPole PPO with comprehensive MLflow tracking.

**Basic Usage:**
```bash
nix develop --command python scripts/train.py
```

**With custom parameters:**
```bash
nix develop --command python scripts/train.py \
    --episodes 500 \
    --learning-rate 1e-4 \
    --experiment-name my-experiment
```

**Disable MLflow tracking:**
```bash
nix develop --command python scripts/train.py --disable-mlflow
```

### `evaluate.py`
Evaluation script for trained CartPole PPO models.

**Evaluate a local checkpoint:**
```bash
nix develop --command python scripts/evaluate.py \
    --checkpoint models/best_model.pkl \
    --episodes 50
```

**Evaluate an MLflow model:**
```bash
nix develop --command python scripts/evaluate.py \
    --run-id 1234567890abcdef \
    --episodes 100
```

**With video recording:**
```bash
nix develop --command python scripts/evaluate.py \
    --checkpoint models/best_model.pkl \
    --episodes 10 \
    --video-dir videos/
```

## Utility Scripts

### `cleanup_mlflow.sh`
Cleans up MLflow experiments and runs.

### `setup_mlflow.sh`
Sets up MLflow server and tracking.