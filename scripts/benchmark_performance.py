#!/usr/bin/env python3
"""
Performance benchmarking script for CartPole PPO.

This script runs comprehensive benchmarks to evaluate the performance
of different hyperparameter configurations and training strategies.
"""

import argparse
import os
import sys
import json
import time
import itertools
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax
import numpy as np
from src.agent.ppo import PPOAgent
from src.environment.cartpole import CartPoleWrapper
from src.training.trainer import PPOTrainer
from src.visualization.mlflow_logger import MLflowLogger


def define_hyperparameter_grid() -> Dict[str, List[Any]]:
    """
    Define the hyperparameter grid for benchmarking
    
    Returns:
        Dictionary of hyperparameters and their values to test
    """
    return {
        'learning_rate': [1e-4, 3e-4, 1e-3],
        'clip_epsilon': [0.1, 0.2, 0.3],
        'gamma': [0.95, 0.99, 0.995],
        'gae_lambda': [0.9, 0.95, 0.99],
        'entropy_coef': [0.001, 0.01, 0.1],
        'value_coef': [0.25, 0.5, 1.0],
        'batch_size': [32, 64, 128],
        'epochs_per_update': [5, 10, 20]
    }


def generate_configurations(param_grid: Dict[str, List[Any]], 
                          max_configs: int = 50) -> List[Dict[str, Any]]:
    """
    Generate hyperparameter configurations for benchmarking
    
    Args:
        param_grid: Dictionary of hyperparameters and their values
        max_configs: Maximum number of configurations to generate
        
    Returns:
        List of hyperparameter configurations
    """
    # Generate all possible combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    all_combinations = list(itertools.product(*values))
    
    # Limit number of configurations if needed
    if len(all_combinations) > max_configs:
        # Randomly sample configurations
        np.random.shuffle(all_combinations)
        all_combinations = all_combinations[:max_configs]
    
    configurations = []
    for combination in all_combinations:
        config = dict(zip(keys, combination))
        configurations.append(config)
    
    return configurations


def run_single_benchmark(config: Dict[str, Any], 
                        experiment_name: str,
                        seed: int = 42) -> Dict[str, Any]:
    """
    Run a single benchmark configuration
    
    Args:
        config: Hyperparameter configuration
        experiment_name: MLflow experiment name
        seed: Random seed
        
    Returns:
        Benchmark results
    """
    print(f"Running benchmark with config: {config}")
    
    # Create random key
    key = jax.random.PRNGKey(seed)
    
    # Create environment
    env = CartPoleWrapper(max_episode_steps=500)
    
    # Create agent with benchmark configuration
    agent = PPOAgent(
        observation_dim=env.observation_dim,
        action_dim=env.action_dim,
        learning_rate=config['learning_rate'],
        clip_epsilon=config['clip_epsilon'],
        gamma=config['gamma'],
        gae_lambda=config['gae_lambda'],
        entropy_coef=config['entropy_coef'],
        value_coef=config['value_coef'],
        batch_size=config['batch_size'],
        epochs_per_update=config['epochs_per_update'],
        key=key
    )
    
    # Create trainer with reduced episodes for benchmarking
    trainer = PPOTrainer(
        agent=agent,
        env=env,
        max_episodes=300,  # Reduced for benchmarking
        max_steps_per_episode=500,
        target_reward=195.0,
        convergence_window=20,
        early_stopping_patience=5,
        eval_frequency=25,
        eval_episodes=5,
        log_frequency=10,
        enable_mlflow=True,
        mlflow_experiment_name=experiment_name,
        key=key
    )
    
    # Run training
    start_time = time.time()
    results = trainer.train()
    end_time = time.time()
    
    # Add configuration and timing info
    results['config'] = config
    results['wall_time'] = end_time - start_time
    results['seed'] = seed
    
    # Calculate additional metrics
    if results['episode_rewards']:
        results['benchmark_metrics'] = {
            'convergence_speed': len(results['episode_rewards']) if results['converged'] else 300,
            'stability_score': np.std(results['episode_rewards'][-50:]) if len(results['episode_rewards']) >= 50 else np.std(results['episode_rewards']),
            'efficiency': results['total_steps'] / max(results['final_avg_reward'], 1.0),
            'sample_efficiency': results['total_steps'] / max(results['best_avg_reward'], 1.0)
        }
    
    env.close()
    return results


def run_benchmark_suite(param_grid: Dict[str, List[Any]], 
                       experiment_name: str = "cartpole-ppo-benchmark",
                       max_configs: int = 20,
                       output_file: str = "benchmark_results.json") -> List[Dict[str, Any]]:
    """
    Run a comprehensive benchmark suite
    
    Args:
        param_grid: Dictionary of hyperparameters and their values
        experiment_name: MLflow experiment name
        max_configs: Maximum number of configurations to test
        output_file: File to save benchmark results
        
    Returns:
        List of benchmark results
    """
    print("=" * 60)
    print("CARTPOLE PPO BENCHMARK SUITE")
    print("=" * 60)
    
    # Generate configurations
    configurations = generate_configurations(param_grid, max_configs)
    print(f"Generated {len(configurations)} configurations for benchmarking")
    
    # Run benchmarks
    all_results = []
    for i, config in enumerate(configurations):
        print(f"\nBenchmark {i+1}/{len(configurations)}")
        print("-" * 40)
        
        try:
            result = run_single_benchmark(
                config, 
                f"{experiment_name}_config_{i+1}",
                seed=42 + i
            )
            all_results.append(result)
            
            # Print summary
            print(f"Results: Reward={result['final_avg_reward']:.1f}, "
                  f"Converged={result['converged']}, "
                  f"Time={result['wall_time']:.1f}s")
            
        except Exception as e:
            print(f"Error running benchmark: {e}")
            continue
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nBenchmark results saved to {output_file}")
    return all_results


def analyze_benchmark_results(results: List[Dict[str, Any]], 
                            output_dir: str = "benchmark_analysis") -> Dict[str, Any]:
    """
    Analyze benchmark results and create summary report
    
    Args:
        results: List of benchmark results
        output_dir: Directory to save analysis
        
    Returns:
        Analysis summary
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if not results:
        return {}
    
    # Extract metrics
    converged_runs = [r for r in results if r.get('converged', False)]
    
    analysis = {
        'summary': {
            'total_runs': len(results),
            'converged_runs': len(converged_runs),
            'convergence_rate': len(converged_runs) / len(results) * 100,
            'avg_final_reward': np.mean([r['final_avg_reward'] for r in results]),
            'best_final_reward': max([r['final_avg_reward'] for r in results]),
            'avg_training_time': np.mean([r['wall_time'] for r in results]),
        },
        'converged_summary': {},
        'best_configurations': {},
        'parameter_analysis': {}
    }
    
    if converged_runs:
        analysis['converged_summary'] = {
            'avg_convergence_episodes': np.mean([r['total_episodes'] for r in converged_runs]),
            'min_convergence_episodes': min([r['total_episodes'] for r in converged_runs]),
            'avg_convergence_time': np.mean([r['wall_time'] for r in converged_runs]),
        }
    
    # Find best configurations for different metrics
    metrics = ['final_avg_reward', 'best_avg_reward', 'wall_time']
    for metric in metrics:
        if metric == 'wall_time':
            best_run = min(results, key=lambda x: x[metric])
        else:
            best_run = max(results, key=lambda x: x[metric])
        
        analysis['best_configurations'][metric] = {
            'value': best_run[metric],
            'config': best_run.get('config', {}),
            'converged': best_run.get('converged', False)
        }
    
    # Analyze parameter importance
    if results and 'config' in results[0]:
        param_names = list(results[0]['config'].keys())
        
        for param in param_names:
            param_values = []
            success_rates = []
            
            # Group by parameter value
            value_groups = {}
            for result in results:
                if 'config' in result and param in result['config']:
                    value = result['config'][param]
                    if value not in value_groups:
                        value_groups[value] = []
                    value_groups[value].append(result)
            
            # Calculate success rate for each value
            for value, group in value_groups.items():
                converged_count = sum(1 for r in group if r.get('converged', False))
                success_rate = converged_count / len(group) * 100
                param_values.append(value)
                success_rates.append(success_rate)
            
            if param_values:
                analysis['parameter_analysis'][param] = {
                    'values': param_values,
                    'success_rates': success_rates,
                    'best_value': param_values[np.argmax(success_rates)],
                    'best_success_rate': max(success_rates)
                }
    
    # Save analysis
    analysis_path = os.path.join(output_dir, 'benchmark_analysis.json')
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    # Create visualizations
    logger = MLflowLogger("benchmark-analysis")
    
    # Log parameter analysis
    if analysis['parameter_analysis']:
        for param, param_analysis in analysis['parameter_analysis'].items():
            logger.log_metrics({
                f'{param}_best_success_rate': param_analysis['best_success_rate']
            })
    
    print(f"Benchmark analysis saved to {output_dir}")
    return analysis


def main():
    """Main benchmarking function."""
    parser = argparse.ArgumentParser(
        description="Run performance benchmarks for CartPole PPO"
    )
    
    parser.add_argument("--experiment-name", type=str, default="cartpole-ppo-benchmark",
                        help="MLflow experiment name (default: cartpole-ppo-benchmark)")
    parser.add_argument("--max-configs", type=int, default=20,
                        help="Maximum number of configurations to test (default: 20)")
    parser.add_argument("--output-file", type=str, default="benchmark_results.json",
                        help="Output file for benchmark results (default: benchmark_results.json)")
    parser.add_argument("--analysis-dir", type=str, default="benchmark_analysis",
                        help="Directory for analysis results (default: benchmark_analysis)")
    parser.add_argument("--load-results", type=str,
                        help="Load existing results from file instead of running benchmarks")
    parser.add_argument("--custom-grid", type=str,
                        help="JSON file with custom hyperparameter grid")
    
    args = parser.parse_args()
    
    # Load hyperparameter grid
    if args.custom_grid and os.path.exists(args.custom_grid):
        print(f"Loading custom hyperparameter grid from {args.custom_grid}")
        with open(args.custom_grid, 'r') as f:
            param_grid = json.load(f)
    else:
        param_grid = define_hyperparameter_grid()
    
    # Run or load benchmarks
    if args.load_results and os.path.exists(args.load_results):
        print(f"Loading benchmark results from {args.load_results}")
        with open(args.load_results, 'r') as f:
            results = json.load(f)
    else:
        results = run_benchmark_suite(
            param_grid,
            args.experiment_name,
            args.max_configs,
            args.output_file
        )
    
    # Analyze results
    if results:
        print("\n" + "=" * 60)
        print("ANALYZING BENCHMARK RESULTS")
        print("=" * 60)
        
        analysis = analyze_benchmark_results(results, args.analysis_dir)
        
        # Print summary
        print("\nBenchmark Summary:")
        print(f"Total runs: {analysis['summary']['total_runs']}")
        print(f"Converged runs: {analysis['summary']['converged_runs']}")
        print(f"Convergence rate: {analysis['summary']['convergence_rate']:.1f}%")
        print(f"Best final reward: {analysis['summary']['best_final_reward']:.1f}")
        print(f"Average training time: {analysis['summary']['avg_training_time']:.1f}s")
        
        if analysis['best_configurations']:
            best_config = analysis['best_configurations']['final_avg_reward']
            print(f"\nBest configuration for reward:")
            print(f"  Reward: {best_config['value']:.1f}")
            print(f"  Config: {best_config['config']}")
    
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()