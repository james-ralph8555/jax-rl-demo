#!/bin/bash
# Test script to verify MLflow server PID handling

set -e

echo "🧪 Testing MLflow server PID handling..."

# Clean up any existing server
bash scripts/cleanup_mlflow.sh

# Start a shell with MLflow server in background
echo "Starting MLflow server..."
(
    # Source the same commands as in flake.nix
    export PYTHONPATH="${PWD}/src:$PYTHONPATH"
    export MLFLOW_TRACKING_URI="http://localhost:5000"
    export MLFLOW_BACKEND_STORE_URI="sqlite:///mlflow.db"
    
    # Create and activate virtual environment for MLflow server if it doesn't exist
    if [ ! -d ".venv" ]; then
        echo "Creating virtual environment for MLflow server..."
        python -m venv .venv
    fi
    
    # Install mlflow in the virtual environment if not already installed
    if [ ! -f ".venv/bin/mlflow" ]; then
        echo "Installing mlflow in virtual environment..."
        .venv/bin/pip install --upgrade mlflow
    fi
    
    # Start MLflow server in background
    echo "Starting MLflow tracking server..."
    .venv/bin/mlflow server \
        --host 0.0.0.0 \
        --port 5000 \
        --backend-store-uri $MLFLOW_BACKEND_STORE_URI \
        --default-artifact-root ./mlartifacts \
        --serve-artifacts > /dev/null 2>&1 &
    
    # Store the parent shell PID
    PARENT_PID=$!
    echo $PARENT_PID > .mlflow_server.pid
    
    # Wait a moment for the server to start and get the actual MLflow process
    sleep 2
    
    # Find the actual MLflow server process (child of the parent)
    MLFLOW_PID=$(pgrep -P $PARENT_PID 2>/dev/null || echo "")
    if [ -n "$MLFLOW_PID" ]; then
        echo $MLFLOW_PID > .mlflow_server.pid
    fi
) &

# Wait for server to start
sleep 3

# Check if PID file was created
if [ -f ".mlflow_server.pid" ]; then
    PID=$(cat .mlflow_server.pid)
    echo "✅ PID file created with PID: $PID"
    
    # Check if the process is actually running
    if ps -p $PID > /dev/null 2>&1; then
        echo "✅ Process $PID is running"
        
        # Check if it's actually MLflow
        CMD=$(ps -p $PID -o cmd= 2>/dev/null || echo "")
        echo "Process command: $CMD"
        
        # Check if port 5000 is being used
        if lsof -ti:5000 >/dev/null 2>&1; then
            echo "✅ Port 5000 is in use"
        else
            echo "⚠️  Port 5000 is not in use"
        fi
    else
        echo "❌ Process $PID is not running"
    fi
else
    echo "❌ PID file not created"
fi

# Test cleanup
echo ""
echo "Testing cleanup..."
bash scripts/cleanup_mlflow.sh

# Verify server is stopped
sleep 2
if lsof -ti:5000 >/dev/null 2>&1; then
    echo "❌ Server still running on port 5000"
else
    echo "✅ Server successfully stopped"
fi

echo ""
echo "🏁 Test complete!"