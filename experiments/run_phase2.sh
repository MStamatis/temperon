#!/bin/sh
# Full Phase 2 campaign on one GPU.
#   Fast track (ResNet-18): arms A, C, D, E + arm B LR-sweep {3e-4,1e-3,3e-3}
#   Paper track (ResNet-110): arms A, C, D
#   3 seeds each, parallel=3 (measured ~1.76x throughput vs sequential).
# No `set -e`: a single failed run must not abort the rest of the campaign;
# launch_grid already isolates per-run failures and reports them.
#
# Run foreground (watch live):   .\eos.ps1 sh experiments/run_phase2.sh
# Run detached (overnight):
#   docker exec -d eos-switch-dev sh -lc "sh experiments/run_phase2.sh > campaign_phase2.log 2>&1"
SEEDS=42,1181241943,958682846
OUT=results/phase2
mkdir -p "$OUT"

echo "=== Fast track (ResNet-18) + Paper track (ResNet-110) ==="
python experiments/launch_grid.py \
  --configs configs/arm_A_paper_baseline.yaml \
            configs/arm_C_warmup_only.yaml \
            configs/arm_D_optiroulette.yaml \
            configs/arm_E_optiroulette_no_warmup.yaml \
            configs/paper_A_resnet110.yaml \
            configs/paper_C_resnet110.yaml \
            configs/paper_D_resnet110.yaml \
  --seeds "$SEEDS" --parallel 3 --output "$OUT"

echo "=== Arm B LR sweep (ResNet-18) ==="
python experiments/launch_grid.py \
  --configs configs/arm_B_tuned_baseline.yaml \
  --sweep controller.lr=3e-4,1e-3,3e-3 \
  --seeds "$SEEDS" --parallel 3 --output "$OUT"

echo "=== Building report ==="
python -m eos_switch.report.make_report "$OUT"
echo "=== CAMPAIGN COMPLETE ==="
