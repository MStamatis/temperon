#!/bin/sh
# Re-run the OptiRoulette arms (D, E, paper_D) with the FIXED LR-scaling
# package, to compare against the broken-scaling Phase 2 baselines.
# Probes off (this comparison is about accuracy / milestones / switch LRs,
# not sharpness), 3 seeds, parallel=3. 100 epochs to match Phase 2.
SEEDS=42,1181241943,958682846
OUT=results/phase2_lrfix
mkdir -p "$OUT"
python experiments/launch_grid.py \
  --configs configs/arm_D_optiroulette.yaml \
            configs/arm_E_optiroulette_no_warmup.yaml \
            configs/paper_D_resnet110.yaml \
  --seeds "$SEEDS" --parallel 3 --extra "--set probes.enabled=false" --output "$OUT"
echo "=== LRFIX RE-RUN COMPLETE ==="
