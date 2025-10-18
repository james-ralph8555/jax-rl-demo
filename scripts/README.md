# CartPole PPO Scripts

This directory contains polished scripts for training, evaluating, and analyzing CartPole PPO agents with MLflow integration.

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

**Available Parameters:**
- `--episodes`: Maximum training episodes (default: 1000)
- `--max-steps`: Maximum steps per episode (default: 500)
- `--target-reward`: Target reward for convergence (default: 195.0)
- `--convergence-window`: Episodes to check for convergence (default: 20)
- `--early-stopping-patience`: Episodes to wait after convergence (default: 10)
- `--eval-frequency`: Frequency of evaluation episodes (default: 50)
- `--eval-episodes`: Number of evaluation episodes (default: 10)
- `--log-frequency`: Frequency of logging metrics (default: 10)
- `--learning-rate`: Learning rate (default: 3e-4)
- `--clip-epsilon`: PPO clipping parameter (default: 0.2)
- `--gamma`: Discount factor (default: 0.99)
- `--gae-lambda`: GAE lambda parameter (default: 0.95)
- `--entropy-coef`: Entropy coefficient (default: 0.01)
- `--value-coef`: Value function coefficient (default: 0.5)
- `--max-grad-norm`: Maximum gradient norm (default: 0.5)
- `--batch-size`: Batch size for PPO updates (default: 64)
- `--epochs-per-update`: Number of epochs per PPO update (default: 10)
- `--experiment-name`: MLflow experiment name (default: cartpole-ppo)
- `--disable-mlflow`: Disable MLflow tracking
- `--video-record-freq`: Frequency of video recording during evaluation (default: 200)
- `--seed`: Random seed (default: 42)

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

**Available Parameters:**
- `--run-id`: MLflow run ID (e.g., "ed91b0290af04f9b9fef07e6d72b44f6")
- `--model-uri`: MLflow model URI (e.g., "runs:/<run_id>/final_model")
- `--checkpoint`: Path to local checkpoint file (.pkl)
- `--episodes`: Number of evaluation episodes (default: 100)
- `--max-steps`: Maximum steps per episode (default: 500)
- `--render`: Render environment during evaluation
- `--video-dir`: Directory to save videos (default: evaluation_videos)
- `--output`: Output file for evaluation report (default: evaluation_report.json)
- `--experiment`: MLflow experiment name (default: cartpole-ppo)
- `--artifact-path`: MLflow artifact path within the run (default: final_model)

### `analyze_hyperparameters.py`
Hyperparameter analysis script for CartPole PPO.

**Basic Usage:**
```bash
nix develop --command python scripts/analyze_hyperparameters.py
```

**With custom experiment:**
```bash
nix develop --command python scripts/analyze_hyperparameters.py \
    --experiment-name my-experiment \
    --output-dir analysis_results
```

**Compare specific runs:**
```bash
nix develop --command python scripts/analyze_hyperparameters.py \
    --compare-runs run1 run2 run3
```

**Available Parameters:**
- `--experiment-name`: MLflow experiment name (default: cartpole-ppo)
- `--output-dir`: Output directory for analysis results (default: hyperparameter_analysis)
- `--compare-runs`: Specific run names to compare
- `--load-from-file`: Load results from JSON file instead of MLflow

### `benchmark_performance.py`
Performance benchmarking script for CartPole PPO.

**Basic Usage:**
```bash
nix develop --command python scripts/benchmark_performance.py
```

**With custom configuration:**
```bash
nix develop --command python scripts/benchmark_performance.py \
    --max-configs 50 \
    --experiment-name my-benchmark
```

**Load existing results:**
```bash
nix develop --command python scripts/benchmark_performance.py \
    --load-results existing_results.json
```

**Available Parameters:**
- `--experiment-name`: MLflow experiment name (default: cartpole-ppo-benchmark)
- `--max-configs`: Maximum number of configurations to test (default: 20)
- `--output-file`: Output file for benchmark results (default: benchmark_results.json)
- `--analysis-dir`: Directory for analysis results (default: benchmark_analysis)
- `--load-results`: Load existing results from file instead of running benchmarks
- `--custom-grid`: JSON file with custom hyperparameter grid

## Utility Scripts

### `cleanup_mlflow.sh`
Cleans up MLflow experiments and runs.

### `setup_mlflow.sh`
Sets up MLflow server and tracking.