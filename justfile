# CartPole PPO JAX Project Justfile

# Default recipe
default:
    @just --list

# Start MLflow tracking server
start-mlflow:
    @echo "Starting MLflow tracking server..."
    @mkdir -p ./data/mlartifacts
    @uv run --with mlflow mlflow server \
        --host 0.0.0.0 \
        --port 5000 \
        --backend-store-uri sqlite:///./data/mlflow.db \
        --default-artifact-root ./data/mlartifacts \
        --serve-artifacts

# Start MLflow tracking server in background
start-mlflow-bg:
    @echo "Starting MLflow tracking server in background..."
    @mkdir -p ./data/mlartifacts
    @nohup uv run --with mlflow mlflow server \
        --host 0.0.0.0 \
        --port 5000 \
        --backend-store-uri sqlite:///./data/mlflow.db \
        --default-artifact-root ./data/mlartifacts \
        --serve-artifacts > ./data/mlflow.log 2>&1 &
    @PID=$!; echo $$PID > ./data/.mlflow_server.pid; echo "MLflow server started with PID $$PID"
    @echo "Logs available in ./data/mlflow.log"

# Stop MLflow tracking server
stop-mlflow:
    @echo "Stopping MLflow server..."
    @if [ -f ./data/.mlflow_server.pid ]; then \
        PID=$(cat ./data/.mlflow_server.pid); \
        kill $$PID 2>/dev/null || true; \
        pkill -f "mlflow server.*port 5000" 2>/dev/null || true; \
        rm -f ./data/.mlflow_server.pid; \
        echo "✅ MLflow server stopped"; \
    else \
        echo "No MLflow server PID file found"; \
    fi

# Start MLflow MCP server
start-mcp:
    @uv run --with mlflow mlflow mcp run

# Show server status
status:
    @echo "MLflow server status:"
    @if [ -f ./data/.mlflow_server.pid ]; then \
        PID=$(cat ./data/.mlflow_server.pid); \
        if ps -p $$PID > /dev/null 2>&1; then \
            echo "  ✅ MLflow tracking server running (PID: $$PID)"; \
        else \
            echo "  ❌ MLflow tracking server PID file exists but process not running"; \
            rm -f ./data/.mlflow_server.pid; \
        fi; \
    else \
        echo "  ❌ MLflow tracking server not running"; \
    fi

# Run tests
test:
    @python -m pytest tests/ -v

# Run specific test file
test-file file:
    @pytest python -m pytest tests/{{file}} -v

# Clean up PID files and logs
clean:
    @echo "Cleaning up PID files and logs..."
    @rm -f ./data/.mlflow_server.pid ./data/.mcp_server.pid ./data/mlflow.log ./data/mcp.log
    @echo "✅ Cleanup complete"