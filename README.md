# eos-switch

Research codebase extending the **OptiRoulette** meta-optimizer
(arXiv 2603.06613) with **Edge-of-Stability (EoS)**-aware switching
mechanisms, targeting controlled, measurable super-convergence.

Everything runs inside a Docker container on a single NVIDIA RTX 5090
(Blackwell, sm_120, 32 GB VRAM).

## Setup

Requirements on the host: Docker (Desktop) with the NVIDIA container runtime.
Nothing else — Python, uv and all dependencies live in the image.

```bash
# 1. (only after changing pyproject.toml) regenerate the lock file:
docker compose run --rm lock

# 2. build the image — installs PyTorch >= 2.7 with CUDA 12.8 wheels,
#    which is REQUIRED for the RTX 5090 (sm_120). The effective install is:
#    uv sync against pyproject/uv.lock with torch pinned to the
#    https://download.pytorch.org/whl/cu128 index.
docker compose build dev

# 3. start the long-lived dev container and install the project (editable):
docker compose up -d dev
docker compose exec dev uv sync --frozen

# 4. verify the GPU is usable (prints torch/CUDA versions, GPU name,
#    compute capability, runs a CUDA matmul):
docker compose exec dev python experiments/check_env.py
```

If you ever see `no kernel image is available for execution on the device`,
the installed torch wheels lack sm_120 support — rebuild against the cu128
index (step 2).

## OptiRoulette acquisition

Route **(a)** succeeded: `optiroulette` 0.1.0 is installed **from PyPI** as a
regular locked dependency, and its sdist source is vendored under
[third_party/optiroulette](third_party/optiroulette) for study. The rest of
the codebase talks to it only through
[src/eos_switch/optimizers/optiroulette_adapter.py](src/eos_switch/optimizers/optiroulette_adapter.py).
(Route (b), cloning `MStamatis/OptiRoulette` from GitHub, was not needed.)

## Layout

```
src/eos_switch/
  probes/        # curvature & stability measurement (fp32, no autocast/TF32)
  optimizers/    # builders, OptiRoulette adapter, muon, state transfer
  controllers/   # fixed / sequential / optiroulette / eos_switch / mpc
  data/          # GPU-resident CIFAR with on-GPU augmentation
  train/         # models, seeding, training loop
  report/        # run logging, plots, markdown report
configs/         # one YAML per experiment arm
tests/           # pytest, CPU-only
experiments/     # run.py, check_env.py, launch_grid.py
third_party/     # vendored OptiRoulette source (study copy)
```

## Running

```bash
# Smoke run (<5 min, also works on CPU): CIFAR-10 subset 5k, 3 epochs
docker compose exec dev python experiments/run.py --config configs/smoke.yaml --smoke

# Any experiment arm
docker compose exec dev python experiments/run.py --config configs/<arm>.yaml

# Tests (CPU)
docker compose exec dev pytest
```

Each run writes to `results/<run_name>/seed<seed>/`:
`steps.jsonl` (per-step loss/lr/optimizer/sharpness/wall-clock),
`probes.jsonl`, `epochs.csv`, `summary.csv` (first-hit milestones included),
`meta.json` (config + switch events), `nvidia_smi.txt`.

Default seeds: `42, 1181241943, 958682846`. Batch size is fixed at 128 in
all comparative arms (Edge-of-Stochastic-Stability depends on it).

## Phases

- **Phase 0**: scaffolding, Docker, OptiRoulette adapter — done.
- **Phase 1**: probes library (lambda_max, batch sharpness, preconditioned
  sharpness, stability margin) — see `src/eos_switch/probes/`.
- **Phase 2**: ablation harness, arms A-E (schedule effect vs switching
  effect) — configs `arm_*.yaml`.
- **Phase 3+**: EoS-aware switching controller, per-layer Muon pool,
  sharpness-MPC. Not yet run.
