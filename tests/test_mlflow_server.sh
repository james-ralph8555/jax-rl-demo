#!/bin/bash
# Test script to verify MLflow server PID handling

set -e

echo "🧪 Testing MLflow server PID handling..."

# Clean up any existing server
bash scripts/cleanup_mlflow.sh

# Start the server in the background using nix develop
echo "Starting MLflow server with nix develop .#with-mlflow..."
nix develop .#with-mlflow --command bash -c 'sleep 5' &
NIX_PID=$!

# Wait for the server to start
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
        if [[ $CMD == *"mlflow"* ]] || [[ $CMD == *"uvicorn"* ]]; then
            echo "✅ Process is MLflow/uvicorn server"
        else
            echo "⚠️  Process might not be MLflow server: $CMD"
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