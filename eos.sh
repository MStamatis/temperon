#!/usr/bin/env bash
# eos.sh -- run a command inside the eos-switch dev container, starting the
# container first if it is stopped (and creating it via compose if missing).
# POSIX counterpart of eos.ps1; both do exactly the same thing.
# Requires a running Docker daemon with the NVIDIA container runtime.
#
# Usage (from any directory):
#   ./eos.sh python experiments/check_env.py
#   ./eos.sh python -m pytest tests/ -q
#   ./eos.sh python experiments/launch_grid.py --configs "configs/bench/c100_*.yaml" --seeds "42,1181241943" --output results/bench/c100
#   ./eos.sh bash            # interactive shell inside the container
set -euo pipefail

# Stop Git Bash / MSYS from rewriting "/workspace" into a Windows path.
export MSYS_NO_PATHCONV=1

name="eos-switch-dev"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! docker info >/dev/null 2>&1; then
    echo "Docker daemon not reachable. Start Docker (or Docker Desktop) and retry." >&2
    exit 1
fi

# Ensure the container exists and is running.
if [ "$(docker ps -a --filter "name=^/${name}$" --format '{{.Names}}')" = "$name" ]; then
    docker start "$name" >/dev/null
else
    echo "Container $name not found; creating it via docker compose..."
    (cd "$here" && docker compose up -d dev)
fi

if [ "$#" -eq 0 ]; then
    echo "Container is up. Pass a command, e.g.: ./eos.sh python experiments/check_env.py"
    exit 0
fi

# Allocate a TTY only when we actually have one, so CI and piped output work.
tty_flags=()
if [ -t 0 ] && [ -t 1 ]; then
    tty_flags=(-it)
fi

exec docker exec "${tty_flags[@]}" -w /workspace "$name" "$@"
