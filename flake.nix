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
          mlflow
          numpy
          matplotlib
          pytest
        ]);
      in {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            pythonEnv
          ];
          shellHook = ''
            export PYTHONPATH="${builtins.toString ./.}/src:$PYTHONPATH"
            export MLFLOW_TRACKING_URI="http://localhost:5000"
            echo "MLflow tracking server will be available at: $MLFLOW_TRACKING_URI"
            echo "Start with: mlflow ui"
          '';
        };
      });
}
