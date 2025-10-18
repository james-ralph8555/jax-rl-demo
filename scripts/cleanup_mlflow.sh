#!/bin/bash
# Cleanup script for MLflow server

set -e

echo "🧹 Cleaning up MLflow server..."

# Stop MLflow server if running
if [ -f ".mlflow_server.pid" ]; then
    echo "Stopping MLflow server..."
    kill $(cat .mlflow_server.pid) 2>/dev/null || echo "Server already stopped"
    rm -f .mlflow_server.pid
    echo "✅ MLflow server stopped"
else
    echo "No MLflow server PID file found"
fi

# Optional: Clean up MLflow data (uncomment if desired)
# echo "Cleaning up MLflow data..."
# rm -rf mlruns mlartifacts mlflow.db
# echo "✅ MLflow data cleaned up"

echo "🏁 Cleanup complete!"