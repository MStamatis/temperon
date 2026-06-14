# eos.ps1 — run a command inside the eos-switch dev container, starting the
# container first if it is stopped (and creating it via compose if missing).
# Requires Docker Desktop to be running.
#
# Usage (from any directory):
#   .\eos.ps1 python experiments/check_env.py
#   .\eos.ps1 pytest -q
#   .\eos.ps1 python experiments/launch_grid.py --configs "configs/arm_*.yaml" --seeds 42 --extra "--set epochs=1" --output results/bench_clean
#   .\eos.ps1 bash            # interactive shell inside the container

$ErrorActionPreference = "Stop"
$name = "eos-switch-dev"

# Is the Docker daemon reachable?
try { docker info *> $null } catch {
    Write-Host "Docker daemon not reachable. Start Docker Desktop and wait for it to go green." -ForegroundColor Yellow
    exit 1
}

# Ensure the container exists and is running.
$exists = docker ps -a --filter "name=$name" --format "{{.Names}}"
if ($exists -eq $name) {
    docker start $name | Out-Null
} else {
    Write-Host "Container $name not found; creating it via docker compose..." -ForegroundColor Cyan
    Push-Location -LiteralPath $PSScriptRoot
    try { docker compose up -d dev } finally { Pop-Location }
}

if (-not $args -or $args.Count -eq 0) {
    Write-Host "Container is up. Pass a command, e.g.: .\eos.ps1 python experiments/check_env.py"
    exit 0
}

# Forward all arguments verbatim to a command inside the container.
docker exec -w /workspace $name @args
exit $LASTEXITCODE
