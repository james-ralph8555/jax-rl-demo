# Model Evaluation Guide

This guide explains how to use the evaluation script to assess trained PPO models on the CartPole environment.

## Overview

The evaluation script (`scripts/evaluate_model.py`) allows you to:
- Load trained models from MLflow or local checkpoints
- Evaluate model performance without training
- Generate comprehensive performance metrics
- Create video recordings of policy behavior
- Export detailed evaluation reports

## Quick Start

### Basic Evaluation

```bash
# Evaluate a local checkpoint
nix develop --command python scripts/evaluate_model.py --checkpoint models/best_model.pkl

# Evaluate using run ID (recommended)
nix develop --command python scripts/evaluate_model.py --run-id ed91b0290af04f9b9fef07e6d72b44f6

# Evaluate an MLflow model (alternative)
nix develop --command python scripts/evaluate_model.py --model-uri "runs:/1234567890abcdef/final_model"
```

### Evaluation with Video Recording

```bash
# Record videos of the policy in action
nix develop --command python scripts/evaluate_model.py \
    --checkpoint models/best_model.pkl \
    --episodes 10 \
    --record-video \
    --video-dir evaluation_videos
```

## Command Line Options

### Required Arguments
- `--run-id ID`: MLflow run ID (e.g., "ed91b0290af04f9b9fef07e6d72b44f6")
- `--checkpoint PATH`: Path to local model checkpoint file (.pkl)
- `--model-uri URI`: MLflow model URI (e.g., "runs:/<run_id>/final_model")

Note: Exactly one of these must be specified.

### Optional Arguments
- `--artifact-path PATH`: MLflow artifact path within the run (default: final_model)
- `--episodes N`: Number of evaluation episodes (default: 100)
- `--max-steps N`: Maximum steps per episode (default: 500)
- `--render`: Render environment during evaluation
- `--record-video`: Record video of evaluation episodes
- `--video-dir DIR`: Directory to save videos (default: evaluation_videos)
- `--output FILE`: Output file for evaluation report (default: evaluation_report.json)
- `--experiment NAME`: MLflow experiment name (default: cartpole-ppo)

## Examples

### 1. Standard Evaluation

Evaluate a trained model for 100 episodes:

```bash
nix develop --command python scripts/evaluate_model.py \
    --checkpoint models/trained_model.pkl \
    --episodes 100 \
    --output results.json
```

### 2. Quick Evaluation with Visualization

Quick test with rendering:

```bash
nix develop --command python scripts/evaluate_model.py \
    --checkpoint models/trained_model.pkl \
    --episodes 5 \
    --render
```

### 3. Comprehensive Evaluation with Videos

Full evaluation with video recordings:

```bash
nix develop --command python scripts/evaluate_model.py \
    --model-uri "runs:/your-run-id/final_model" \
    --episodes 50 \
    --record-video \
    --video-dir policy_videos \
    --output comprehensive_evaluation.json
```

### 4. MLflow Model Evaluation

Evaluate the best model from MLflow:

```bash
# First, find the best run
nix develop --command python -c "
import mlflow
client = mlflow.tracking.MlflowClient()
runs = client.search_runs(
    experiment_ids=['your-experiment-id'],
    order_by=['metrics.eval_avg_reward DESC'],
    max_results=1
)
if runs:
    print(f'Best run: {runs[0].info.run_id}')
    print(f'Best reward: {runs[0].data.metrics.get(\"eval_avg_reward\", \"N/A\")}')
"

# Then evaluate the best run using run ID (recommended)
nix develop --command python scripts/evaluate_model.py \
    --run-id best-run-id \
    --episodes 100

# Or using model URI (alternative)
nix develop --command python scripts/evaluate_model.py \
    --model-uri "runs:/best-run-id/final_model" \
    --episodes 100
```

## Output and Metrics

### Console Output
The script provides real-time progress:
```
Episode   1/100 | Reward:  487.0 | Length: 487
Episode   2/100 | Reward:  500.0 | Length: 500
...
Episode 100/100 | Reward:  495.0 | Length: 495

Evaluation Results:
Episodes evaluated: 100
Mean reward: 492.34 ± 15.67
Min/Max reward: 445.00 / 500.00
Mean episode length: 492.3 ± 15.7
Success rate (reward ≥ 195): 100.00%
Converged: True
```

### JSON Report
The evaluation report includes:
```json
{
  "evaluation_summary": {
    "total_episodes": 100,
    "mean_reward": 492.34,
    "std_reward": 15.67,
    "success_rate": 1.0,
    "converged": true
  },
  "detailed_metrics": {
    "reward_statistics": {
      "mean": 492.34,
      "std": 15.67,
      "min": 445.0,
      "max": 500.0,
      "median": 495.0
    },
    "length_statistics": {
      "mean": 492.3,
      "std": 15.7,
      "min": 445,
      "max": 500
    }
  },
  "episode_data": {
    "rewards": [487, 500, 495, ...],
    "lengths": [487, 500, 495, ...]
  }
}
```

### Video Recordings
When `--record-video` is specified:
- Videos are saved in MP4 format
- First 10 episodes are recorded by default
- Files named: `cartpole_evaluation-episode-0.mp4`, etc.

## Performance Metrics

### Key Metrics
- **Mean Reward**: Average episode reward across all episodes
- **Success Rate**: Percentage of episodes achieving reward ≥ 195
- **Convergence**: Whether mean reward ≥ 195 (CartPole solving criterion)
- **Episode Length**: Average number of steps per episode

### Statistical Measures
- Standard deviation of rewards and lengths
- Min/Max values for performance range
- Median values for central tendency

## Best Practices

### 1. Evaluation Episodes
- Use at least 100 episodes for reliable statistics
- For quick checks, 10-20 episodes may suffice
- More episodes provide more stable metrics

### 2. Video Recording
- Record 5-10 episodes for policy visualization
- Use consistent episode numbers for comparison
- Videos help identify policy behaviors and failure modes

### 3. Model Selection
- Always evaluate the final model and best checkpoint
- Compare multiple models to select the best performer
- Use MLflow to track and compare different runs

### 4. Reproducibility
- Use fixed random seeds for consistent evaluation
- Document evaluation parameters
- Save evaluation reports for future reference

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   ```
   Error: Could not load model: FileNotFoundError
   ```
   - Check checkpoint file path
   - Verify MLflow URI format
   - Ensure model artifacts exist

2. **Import Errors**
   ```
   ImportError: No module named 'src.agent.ppo'
   ```
   - Run with `nix develop`
   - Check PYTHONPATH includes src directory

3. **Video Recording Issues**
   ```
   Error: No display available
   ```
   - Use headless mode for servers
   - Install required display dependencies

### Solutions

1. **Environment Setup**
   ```bash
   # Always use the Nix development environment
   nix develop --command python scripts/evaluate_model.py [args]
   ```

2. **MLflow Connection**
   ```bash
   # Set MLflow tracking URI
   export MLFLOW_TRACKING_URI="http://localhost:5000"
   ```

3. **Display Issues**
   ```bash
   # For headless environments
   export DISPLAY=""
   # Or use virtual display
   xvfb-run -a python scripts/evaluate_model.py [args]
   ```

## Integration with Training

### During Training
The training script automatically evaluates models:
- Every 50 episodes by default
- Results logged to MLflow
- Best models saved to registry

### Post-Training
Use the evaluation script for:
- Final model assessment
- Comparison between different runs
- Generating reports and visualizations
- Preparing models for deployment

## Advanced Usage

### Custom Evaluation Metrics
Extend the evaluation script to include:
- Custom reward functions
- Domain-specific metrics
- Behavioral analysis
- Performance profiling

### Batch Evaluation
Evaluate multiple models:
```bash
for model in models/*.pkl; do
    nix develop --command python scripts/evaluate_model.py \
        --checkpoint "$model" \
        --output "eval_$(basename "$model" .pkl).json"
done
```

### Automated Reporting
Generate comprehensive reports:
```python
import json
import pandas as pd

# Load multiple evaluation reports
reports = []
for file in eval_reports/*.json:
    with open(file) as f:
        reports.append(json.load(f))

# Create comparison table
df = pd.DataFrame([r['evaluation_summary'] for r in reports])
print(df.sort_values('mean_reward', ascending=False))
```

## Next Steps

After evaluation:
1. Analyze the metrics to assess model quality
2. Review videos to understand policy behavior
3. Compare with baseline models
4. Prepare best models for deployment
5. Document findings for future reference

For more advanced analysis, see the visualization and analysis tools in Phase 4 of the implementation plan.