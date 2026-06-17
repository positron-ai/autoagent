{
  description = "Development environment for the AutoAgent Harbor harness";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  };

  outputs =
    { self, nixpkgs, ... }:
    let
      systems = [
        "aarch64-darwin"
        "aarch64-linux"
        "x86_64-darwin"
        "x86_64-linux"
      ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
    in
    {
      packages = forAllSystems (
        system:
        let
          pkgs = import nixpkgs { inherit system; };
          python = pkgs.python312;

          autoagent-sync = pkgs.writeShellApplication {
            name = "autoagent-sync";
            runtimeInputs = [
              python
              pkgs.uv
            ];
            text = ''
              export UV_PYTHON="${python}/bin/python"
              export UV_PYTHON_DOWNLOADS=never
              exec uv sync --python "$UV_PYTHON" "$@"
            '';
          };

          autoagent-check = pkgs.writeShellApplication {
            name = "autoagent-check";
            runtimeInputs = [
              python
              pkgs.uv
            ];
            text = ''
              export UV_PYTHON="${python}/bin/python"
              export UV_PYTHON_DOWNLOADS=never

              uv sync --python "$UV_PYTHON"
              uv run python -m py_compile agent.py agent-claude.py
              uv run python - <<'PY'
              import importlib.metadata as metadata
              import agent

              for package in ("openai-agents", "harbor", "pandas", "openpyxl", "numpy"):
                  print(f"{package} {metadata.version(package)}")

              print(f"agent import ok: {agent.AutoAgent.name()} model={agent.MODEL}")
              PY
            '';
          };

          autoagent-build-base = pkgs.writeShellApplication {
            name = "autoagent-build-base";
            runtimeInputs = [ pkgs.docker-client ];
            text = ''
              exec docker build -f Dockerfile.base -t autoagent-base .
            '';
          };

          autoagent-run = pkgs.writeShellApplication {
            name = "autoagent-run";
            runtimeInputs = [
              pkgs.coreutils
              python
              pkgs.uv
            ];
            text = ''
              export UV_PYTHON="${python}/bin/python"
              export UV_PYTHON_DOWNLOADS=never

              mkdir -p jobs
              exec uv run harbor run \
                -p tasks/ \
                -n "''${AUTOAGENT_CONCURRENCY:-100}" \
                --agent-import-path agent:AutoAgent \
                -o jobs \
                --job-name "''${AUTOAGENT_JOB_NAME:-latest}" \
                "$@"
            '';
          };

          autoagent-ingest = pkgs.writeShellApplication {
            name = "ingest";
            runtimeInputs = [
              pkgs.coreutils
              pkgs.git
              pkgs.nix
              python
              pkgs.uv
            ];
            text = ''
              export UV_PYTHON="${python}/bin/python"
              export UV_PYTHON_DOWNLOADS=never

              if [ -n "''${AUTOAGENT_HOME:-}" ]; then
                autoagent_home="$AUTOAGENT_HOME"
              elif [ -e /home/jwiegley/autoagent/pyproject.toml ]; then
                autoagent_home=/home/jwiegley/autoagent
              else
                autoagent_home="${self}"
              fi

              cache_root="''${XDG_CACHE_HOME:-$HOME/.cache}"
              mkdir -p "$cache_root/autoagent-ingest"
              export UV_PROJECT_ENVIRONMENT="''${UV_PROJECT_ENVIRONMENT:-$cache_root/autoagent-ingest/.venv}"

              caller_cwd="$PWD"
              exec uv run --project "$autoagent_home" --directory "$caller_cwd" ingest "$@"
            '';
          };
        in
        {
          default = autoagent-sync;
          inherit
            autoagent-ingest
            autoagent-build-base
            autoagent-check
            autoagent-run
            autoagent-sync
            ;
        }
      );

      devShells = forAllSystems (
        system:
        let
          pkgs = import nixpkgs { inherit system; };
          python = pkgs.python312;
        in
        {
          default = pkgs.mkShell {
            packages = [
              pkgs.bashInteractive
              pkgs.cacert
              pkgs.coreutils
              pkgs.curl
              pkgs.docker-client
              pkgs.git
              python
              pkgs.uv
              self.packages.${system}.autoagent-build-base
              self.packages.${system}.autoagent-check
              self.packages.${system}.autoagent-ingest
              self.packages.${system}.autoagent-run
              self.packages.${system}.autoagent-sync
            ];

            shellHook = ''
              export UV_PYTHON="${python}/bin/python"
              export UV_PYTHON_DOWNLOADS=never
              export SSL_CERT_FILE="${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
              export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
              export DOCKER_BUILDKIT=1

              cat <<'EOF'
              AutoAgent dev shell
                autoagent-sync        install/update Python dependencies with uv
                autoagent-check       sync deps and verify the harness imports
                autoagent-build-base  build the Harbor task base image
                ingest                run Tron ingest AutoAgent loop from a Tron worktree
                autoagent-run         run Harbor against tasks/
              EOF
            '';
          };
        }
      );

      formatter = forAllSystems (
        system:
        let
          pkgs = import nixpkgs { inherit system; };
        in
        pkgs.nixfmt
      );
    };
}
