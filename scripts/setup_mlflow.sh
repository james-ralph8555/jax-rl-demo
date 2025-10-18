#!/bin/bash
# Setup script for MLflow virtual environment

set -e

echo "🔧 Setting up MLflow virtual environment..."

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install mlflow
echo "Installing mlflow..."
pip install mlflow

echo "✅ MLflow virtual environment setup complete!"
echo "To activate: source .venv/bin/activate"
echo "To start server: mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --serve-artifacts"