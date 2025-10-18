#!/bin/bash
# Cleanup script for MLflow server

set -e

echo "🧹 Cleaning up MLflow server..."

# Stop MLflow server if running
if [ -f ".mlflow_server.pid" ]; then
    echo "Stopping MLflow server..."
    PID=$(cat .mlflow_server.pid)
    
    # Kill the process group to ensure all children are killed
    kill -- -$PID 2>/dev/null || kill $PID 2>/dev/null || echo "Process $PID not found"
    
    # Also kill any remaining mlflow processes on port 5000
    pkill -f "mlflow server.*port 5000" 2>/dev/null || echo "No additional MLflow processes found"
    
    # Kill anything using port 5000 as a fallback
    lsof -ti:5000 | xargs kill 2>/dev/null || echo "No processes using port 5000"
    
    rm -f .mlflow_server.pid
    echo "✅ MLflow server stopped"
else
    echo "No MLflow server PID file found"
    # Try to kill any MLflow processes anyway
    pkill -f "mlflow server.*port 5000" 2>/dev/null || echo "No MLflow processes found"
    lsof -ti:5000 | xargs kill 2>/dev/null || echo "No processes using port 5000"
fi

# Optional: Clean up MLflow data (uncomment if desired)
# echo "Cleaning up MLflow data..."
# rm -rf mlruns mlartifacts mlflow.db
# echo "✅ MLflow data cleaned up"

echo "🏁 Cleanup complete!"