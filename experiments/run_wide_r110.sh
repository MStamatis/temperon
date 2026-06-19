#!/bin/sh
# Port the OptiRoulette-matching setup (WIDE ResNet-110: initial_channels=64,
# BN momentum 0.01, matched aug + 90/10 val + grad_clip 2.0) to the new arms.
# Run cyclic-J and strong-SGD on the SAME wide net, head-to-head vs the
# OptiRoulette repro (results/repro). 3 seeds; parallel 2 (wide net ~27.6M).
S=42,1181241943,958682846
OUT=results/wide_r110
mkdir -p "$OUT"
WIDE="--set model=resnet110 --set initial_channels=64 --set bn_momentum=0.01"
echo "=== cyclic-J wide-R110 ==="
python experiments/launch_grid.py --configs configs/match_cyclic.yaml --seeds "$S" --parallel 2 \
  --extra "$WIDE --set run_name=cyclicJ_wr110" --output "$OUT"
echo "=== strong-SGD wide-R110 ==="
python experiments/launch_grid.py --configs configs/match_strong.yaml --seeds "$S" --parallel 2 \
  --extra "$WIDE --set run_name=strong_wr110" --output "$OUT"
echo "=== WIDE R110 DONE ==="
