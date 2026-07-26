# Reproducing the results

Every command runs inside the container. On Windows use `./eos.ps1 <cmd>`; on
Linux use `docker compose exec dev <cmd>`. All commands below assume the
repository root as the working directory.

Canonical seeds: `42, 1181241943, 958682846, 271828, 314159`.
Run **sequentially** (`--parallel 1`, the default) — the wall-clock numbers are
part of the claim, and concurrent runs inflate them by 20-30%.

## 0. Environment

```bash
docker compose build dev
docker compose up -d dev
docker compose exec dev uv sync --frozen
./eos.ps1 python experiments/check_env.py     # GPU sanity
./eos.ps1 python -m pytest tests/ -q          # 141 tests, CPU-only
```

## 1. Vision baselines (the 4x5 benchmark matrix)

The per-dataset x per-arm configs are generated, not hand-written:

```bash
./eos.ps1 python experiments/gen_bench_configs.py
```

Then, per dataset (`c100`, `c10`, `tiny`, `svhn`) — this is the bulk of the
GPU time, roughly 40 hours in total for all four:

```bash
./eos.ps1 python experiments/launch_grid.py --configs "configs/bench/c100_*.yaml" --seeds "42,1181241943,958682846,271828,314159" --continue --output results/bench/c100
```

Outputs: `results/bench/<dataset>/<dataset>_<arm>/seed<seed>/`.

## 2. Phase 6 — the paper's arms

```bash
./eos.ps1 python experiments/launch_grid.py --configs configs/arm_P_handoff.yaml configs/arm_P_handoff_e80.yaml configs/arm_P_tiny.yaml configs/arm_P_c10.yaml configs/arm_P_svhn.yaml configs/c100_sammuon_p2.yaml --seeds "42,1181241943,958682846,271828,314159" --continue --output results/phase6
```

| config | role in the paper |
|---|---|
| `arm_P_handoff.yaml` | the method, CIFAR-100, 100 epochs |
| `arm_P_handoff_e80.yaml` | the time/accuracy dial (80 epochs) |
| `arm_P_tiny.yaml` / `arm_P_c10.yaml` / `arm_P_svhn.yaml` | transfer, incl. a SAM+SGD tail on Tiny (refiner-agnostic) |
| `c100_sammuon_p2.yaml` | **money control**: uniform periodic SAM at equal-or-greater budget |
| `arm_O_tailsam_f{15,25,40}.yaml` | the *failing* allocation (tail on a catapult continuation) |
| `arm_Q_msam_r{005,030}.yaml` | MSAM literature rival (eliminated in-pipeline) |

Statistics for every table in the README:

```bash
./eos.ps1 python experiments/analyze_5seed.py results
```

This prints per-seed values, 5-seed mean±sd, median first-hit wall-clock per
target, and Welch two-sided tests. Saved output:
`results/phase6/analysis_5seed.txt`.

## 3. Phase 7 — GPT-2-class language model

One-time data preparation (CPU, ~5 min, writes `data/lm/wt103/`):

```bash
./eos.ps1 python experiments/lm/prepare_data.py
```

Throughput check before committing GPU hours (the protocol used throughout —
measure, then decide):

```bash
./eos.ps1 python experiments/lm/train_lm.py --config configs/lm_muon.yaml --bench 40
```

The three arms (~3.4 h total: 0.86 + 1.06 + 1.48 h):

```bash
foreach ($c in 'lm_muon','lm_handoff','lm_sammuon') { .\eos.ps1 python experiments/lm/train_lm.py --config "configs/$c.yaml" --continue }
```

The 20-pass boundary experiment (same budget, 20M-token slice) uses the
`configs/lm_*_r20.yaml` variants with the same loop.

Outputs: `results/phase7/<run_name>/seed42/{evals.csv,metrics.json}`.
`evals.csv` carries `sam_on` and `step_ms`, which is how the gate and the
1-pass/2-pass cost split are verified.

**Noise floor**: `lm_muon` and `lm_handoff` are bit-identical before the switch,
so their pre-switch divergence measures run-to-run nondeterminism (bf16
atomics + compile autotuning). Measured: mean 0.0031, max 0.0092 nats. Use it
as the single-seed significance yardstick.

## 4. Phase 8 — GLUE fine-tuning

RoBERTa-base on the four small GLUE tasks, three arms, 5 seeds (~6.5-7 h; the
launcher skips finished runs, so re-running the same command resumes):

```bash
./eos.ps1 python experiments/glue/launch_glue.py
```

Single run:

```bash
./eos.ps1 python experiments/glue/train_glue.py --task rte --arm tail --seed 42
```

Outputs: `results/phase8/<task>/<arm>/seed<seed>/result.json`.

**Pre-registered**: the primary metric is the **best dev score** across epochs
(standard GLUE practice); `final_score` is reported as secondary. This was
fixed before any Phase-8 result was inspected.

## Notes on timing hygiene

- Epoch 0 is always 3-4x slower (autotune); first-hit medians and per-epoch
  medians exclude it.
- Any background GPU load inflates wall-clock. Two runs in this repository were
  affected and are flagged in the analysis output; they do not change rankings.
- `torch.compile` needs ~20 warmup steps to settle. The LM bench mode uses 20
  warmup steps for exactly this reason — a shorter warmup biases the 1-pass arm
  slow and can make SAM look faster than no-SAM, which is impossible.
