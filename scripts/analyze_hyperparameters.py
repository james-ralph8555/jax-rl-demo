#!/usr/bin/env python3
"""
Hyperparameter analysis script for CartPole PPO.

This script analyzes training results across different hyperparameter configurations
and creates comprehensive visualizations and statistical reports.
"""

import argparse
import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from src.visualization.mlflow_logger import MLflowLogger
from src.visualization.plots import (
    plot_hyperparameter_heatmap,
    plot_hyperparameter_comparison,
    create_comprehensive_analysis
)


def load_results_from_mlflow(experiment_name: str) -> Dict[str, Dict[str, Any]]:
    """
    Load training results from MLflow experiment
    
    Args:
        experiment_name: Name of the MLflow experiment
        
    Returns:
        Dictionary of results for different runs
    """
    logger = MLflowLogger(experiment_name)
    
    try:
        # Get all runs from the experiment
        import mlflow
        runs = mlflow.search_runs(experiment_ids=[logger.experiment_id])
        
        results = {}
        for _, run in runs.iterrows():
            run_id = run['run_id']
            run_name = run.get('tags', {}).get('mlflow.runName', run_id)
            
            # Get parameters
            params = dict(run.get('params', {}))
            
            # Get metrics
            metrics = {}
            for col in run.columns:
                if col.startswith('metrics.'):
                    metric_name = col[8:]  # Remove 'metrics.' prefix
                    metric_value = run[col]
                    if not np.isnan(metric_value):
                        metrics[metric_name] = float(metric_value)
            
            # Get artifacts
            artifacts = {}
            try:
                artifact_uri = mlflow.get_run(run_id).info.artifact_uri
                local_path = f"/tmp/mlflow_artifacts_{run_id}"
                mlflow.artifacts.download_artifacts(artifact_uri, local_path)
                
                # Load training data if available
                training_data_file = os.path.join(local_path, "training_curves", "episode_rewards.json")
                if os.path.exists(training_data_file):
                    with open(training_data_file, 'r') as f:
                        artifacts['episode_rewards'] = json.load(f)
                
                losses_file = os.path.join(local_path, "training_curves", "losses.json")
                if os.path.exists(losses_file):
                    with open(losses_file, 'r') as f:
                        artifacts['losses'] = json.load(f)
                        
            except Exception as e:
                print(f"Warning: Could not load artifacts for run {run_id}: {e}")
            
            results[run_name] = {
                'run_id': run_id,
                'hyperparams': params,
                'metrics': metrics,
                'artifacts': artifacts
            }
        
        return results
        
    except Exception as e:
        print(f"Error loading results from MLflow: {e}")
        return {}


def analyze_hyperparameter_sensitivity(results: Dict[str, Dict[str, Any]], 
                                     target_param: str,
                                     metric: str = 'final_avg_reward') -> Dict[str, Any]:
    """
    Analyze sensitivity of a specific hyperparameter
    
    Args:
        results: Dictionary of training results
        target_param: Parameter to analyze sensitivity for
        metric: Metric to analyze
        
    Returns:
        Sensitivity analysis results
    """
    param_values = []
    metric_values = []
    
    for run_name, run_data in results.items():
        if 'hyperparams' in run_data and 'metrics' in run_data:
            hyperparams = run_data['hyperparams']
            metrics = run_data['metrics']
            
            if target_param in hyperparams and metric in metrics:
                try:
                    param_val = float(hyperparams[target_param])
                    metric_val = float(metrics[metric])
                    param_values.append(param_val)
                    metric_values.append(metric_val)
                except (ValueError, TypeError):
                    continue
    
    if not param_values:
        return {}
    
    # Sort by parameter values
    sorted_pairs = sorted(zip(param_values, metric_values))
    param_values, metric_values = zip(*sorted_pairs)
    
    # Calculate correlation
    correlation = np.corrcoef(param_values, metric_values)[0, 1]
    
    # Find best parameter value
    best_idx = np.argmax(metric_values)
    best_param_value = param_values[best_idx]
    best_metric_value = metric_values[best_idx]
    
    return {
        'parameter': target_param,
        'metric': metric,
        'correlation': float(correlation) if not np.isnan(correlation) else 0.0,
        'best_param_value': float(best_param_value),
        'best_metric_value': float(best_metric_value),
        'param_values': list(param_values),
        'metric_values': list(metric_values)
    }


def create_hyperparameter_report(results: Dict[str, Dict[str, Any]], 
                               output_dir: str) -> None:
    """
    Create comprehensive hyperparameter analysis report
    
    Args:
        results: Dictionary of training results
        output_dir: Directory to save the report
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all hyperparameters
    all_params = set()
    for run_data in results.values():
        if 'hyperparams' in run_data:
            all_params.update(run_data['hyperparams'].keys())
    
    # Analyze each parameter
    sensitivity_analyses = {}
    for param in all_params:
        if param != 'mlflow.runName':  # Skip non-hyperparameter
            analysis = analyze_hyperparameter_sensitivity(results, param)
            if analysis:
                sensitivity_analyses[param] = analysis
    
    # Create summary report
    report = {
        'summary': {
            'total_runs': len(results),
            'analyzed_parameters': len(sensitivity_analyses),
            'parameters_analyzed': list(sensitivity_analyses.keys())
        },
        'sensitivity_analyses': sensitivity_analyses,
        'best_configurations': {}
    }
    
    # Find best configurations for different metrics
    metrics_to_analyze = ['final_avg_reward', 'best_avg_reward', 'converged']
    for metric in metrics_to_analyze:
        best_run = None
        best_value = float('-inf')
        
        for run_name, run_data in results.items():
            if 'metrics' in run_data and metric in run_data['metrics']:
                value = float(run_data['metrics'][metric])
                if value > best_value:
                    best_value = value
                    best_run = run_name
        
        if best_run:
            report['best_configurations'][metric] = {
                'run_name': best_run,
                'value': best_value,
                'hyperparams': results[best_run].get('hyperparams', {})
            }
    
    # Save report
    report_path = os.path.join(output_dir, 'hyperparameter_analysis_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Create visualizations
    logger = MLflowLogger("hyperparameter-analysis")
    
    # Create heatmaps for key metrics
    for metric in ['final_avg_reward', 'best_avg_reward']:
        if any(metric in run_data.get('metrics', {}) for run_data in results.values()):
            logger.log_hyperparameter_comparison(results, metric)
    
    print(f"Hyperparameter analysis report saved to {report_path}")
    print(f"Visualizations logged to MLflow experiment 'hyperparameter-analysis'")


def compare_specific_runs(results: Dict[str, Dict[str, Any]], 
                         run_names: List[str],
                         output_dir: str) -> None:
    """
    Compare specific runs side-by-side
    
    Args:
        results: Dictionary of training results
        run_names: List of run names to compare
        output_dir: Directory to save comparison plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter results to only include specified runs
    comparison_results = {}
    for run_name in run_names:
        if run_name in results:
            comparison_results[run_name] = results[run_name]
        else:
            print(f"Warning: Run '{run_name}' not found in results")
    
    if len(comparison_results) < 2:
        print("Need at least 2 runs for comparison")
        return
    
    # Create comparison data for plotting
    plot_data = {}
    for run_name, run_data in comparison_results.items():
        artifacts = run_data.get('artifacts', {})
        
        if 'episode_rewards' in artifacts:
            plot_data[run_name] = {
                'rewards': artifacts['episode_rewards'],
                'losses': artifacts.get('losses', {})
            }
    
    if plot_data:
        # Create comparison plots
        from src.visualization.plots import plot_hyperparameter_comparison
        
        # Compare rewards
        plot_hyperparameter_comparison(
            plot_data, 
            metric='rewards',
            save_path=os.path.join(output_dir, 'reward_comparison.png'),
            show=False
        )
        
        # Compare losses if available
        if any('losses' in data and data['losses'] for data in plot_data.values()):
            plot_hyperparameter_comparison(
                plot_data,
                metric='losses',
                save_path=os.path.join(output_dir, 'loss_comparison.png'),
                show=False
            )
        
        print(f"Comparison plots saved to {output_dir}")


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(
        description="Analyze hyperparameter sensitivity and create comparison plots"
    )
    
    parser.add_argument("--experiment-name", type=str, default="cartpole-ppo",
                        help="MLflow experiment name (default: cartpole-ppo)")
    parser.add_argument("--output-dir", type=str, default="hyperparameter_analysis",
                        help="Output directory for analysis results (default: hyperparameter_analysis)")
    parser.add_argument("--compare-runs", nargs='+', default=[],
                        help="Specific run names to compare")
    parser.add_argument("--load-from-file", type=str,
                        help="Load results from JSON file instead of MLflow")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("HYPERPARAMETER ANALYSIS FOR CARTPOLE PPO")
    print("=" * 60)
    
    # Load results
    if args.load_from_file and os.path.exists(args.load_from_file):
        print(f"Loading results from file: {args.load_from_file}")
        with open(args.load_from_file, 'r') as f:
            results = json.load(f)
    else:
        print(f"Loading results from MLflow experiment: {args.experiment_name}")
        results = load_results_from_mlflow(args.experiment_name)
    
    if not results:
        print("No results found!")
        return
    
    print(f"Found {len(results)} training runs")
    
    # Create comprehensive analysis
    print("\nCreating hyperparameter analysis report...")
    create_hyperparameter_report(results, args.output_dir)
    
    # Compare specific runs if requested
    if args.compare_runs:
        print(f"\nComparing specific runs: {args.compare_runs}")
        compare_dir = os.path.join(args.output_dir, "run_comparison")
        compare_specific_runs(results, args.compare_runs, compare_dir)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {args.output_dir}")
    print("Check MLflow UI for additional visualizations")


if __name__ == "__main__":
    main()