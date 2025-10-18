#!/usr/bin/env python3
"""
Test script for the evaluation script with run ID functionality.
"""

import subprocess
import sys

def test_run_id_evaluation():
    """Test evaluation using run ID directly."""
    
    # Use one of the run IDs from the database
    run_id = "ed91b0290af04f9b9fef07e6d72b44f6"
    
    print(f"Testing evaluation with run ID: {run_id}")
    print("Command: python scripts/evaluate_model.py --run-id ed91b0290af04f9b9fef07e6d72b44f6 --episodes 5")
    
    result = subprocess.run([
        sys.executable, "scripts/evaluate_model.py",
        "--run-id", run_id,
        "--episodes", "5",
        "--output", "test_run_id_evaluation.json"
    ], capture_output=True, text=True)
    
    print("STDOUT:")
    print(result.stdout)
    
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    
    if result.returncode == 0:
        print("✓ Run ID evaluation test passed!")
    else:
        print("✗ Run ID evaluation test failed!")
    
    return result.returncode == 0

def test_help():
    """Test help output."""
    print("\nTesting help output:")
    result = subprocess.run([
        sys.executable, "scripts/evaluate_model.py", "--help"
    ], capture_output=True, text=True)
    
    print(result.stdout)
    return result.returncode == 0

if __name__ == "__main__":
    print("Testing evaluation script with run ID functionality")
    print("=" * 60)
    
    # Test help first
    help_success = test_help()
    
    # Test run ID evaluation
    run_id_success = test_run_id_evaluation()
    
    print("\n" + "=" * 60)
    if help_success and run_id_success:
        print("All tests passed! ✅")
    else:
        print("Some tests failed! ❌")