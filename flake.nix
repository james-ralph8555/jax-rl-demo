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
          numpy
          pytest
          mlflow
          # mlflow db
          sqlalchemy
          #gym classic control
          gymnasium
          pygame
          # mujoco gym env
          mujoco
          #evaluation/visualization
          moviepy
          matplotlib
          imageio
          seaborn
          plotly
          # mlflow system metrics
          psutil
        ]);
      in {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            pythonEnv
            just
            uv
          ];
          shellHook = ''
            export PYTHONPATH="${builtins.toString ./.}/src:$PYTHONPATH"
            export MLFLOW_TRACKING_URI="http://localhost:5000"
            
            echo "Development environment ready."
            echo ""
            echo "Available commands:"
            echo "  just --list              Show all available commands"
            echo ""
            echo "Use 'just <command>' to run any command"
          '';
        };
      });
}
