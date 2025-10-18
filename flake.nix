{
  description = "CartPole PPO with JAX and MLflow";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        pythonEnv = pkgs.python313.withPackages (ps: with ps; [
          jax
          jaxlib
          flax
          gymnasium
          numpy
          matplotlib
          pytest
          sqlalchemy
          mlflow
        ]);
      in {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            pythonEnv
          ];
          shellHook = ''
            export PYTHONPATH="${builtins.toString ./.}/src:$PYTHONPATH"
            export MLFLOW_TRACKING_URI="http://localhost:5000"
            
            # Set up MLflow backend store URI
            export MLFLOW_BACKEND_STORE_URI="sqlite:///mlflow.db"
            
            echo "Development environment ready."
            echo "To start MLflow server, run: nix develop .#with-mlflow"
          '';
        };

        devShells.with-mlflow = pkgs.mkShell {
          buildInputs = with pkgs; [
            pythonEnv
          ];
          shellHook = ''
            export PYTHONPATH="${builtins.toString ./.}/src:$PYTHONPATH"
            export MLFLOW_TRACKING_URI="http://localhost:5000"
            
            # Set up MLflow backend store URI
            export MLFLOW_BACKEND_STORE_URI="sqlite:///mlflow.db"
            
            # Create and activate virtual environment for MLflow server if it doesn't exist
            if [ ! -d ".venv" ]; then
              echo "Creating virtual environment for MLflow server..."
              python -m venv .venv
            fi
            
            # Install mlflow in the virtual environment if not already installed
            if [ ! -f ".venv/bin/mlflow" ]; then
              echo "Installing mlflow in virtual environment..."
              .venv/bin/pip install mlflow
            fi
            
            # Start MLflow server in background
            echo "Starting MLflow tracking server..."
            .venv/bin/mlflow server \
              --host 0.0.0.0 \
              --port 5000 \
              --backend-store-uri $MLFLOW_BACKEND_STORE_URI \
              --default-artifact-root ./mlartifacts \
              --serve-artifacts &
            
            # Store the server PID to kill it later
            echo $! > .mlflow_server.pid
            
            echo "MLflow tracking server started at: $MLFLOW_TRACKING_URI"
            echo "Backend store: $MLFLOW_BACKEND_STORE_URI"
            echo "Artifacts stored in: ./mlartifacts"
            echo ""
            echo "To stop the server when done: kill \$(cat .mlflow_server.pid)"
            echo ""
          '';
        };
      });
}
