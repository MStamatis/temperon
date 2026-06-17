#!/bin/sh
# OptiRoulette-matched comparison: cyclic-J vs strong-SGD, on ResNet-18 AND
# ResNet-110, with the framework's exact data pipeline (crop+flip+cutout8+
# color_jitter0.1 + 90/10 held-out val + grad-clip 2.0). 3 seeds each.
S=42,1181241943,958682846
OUT=results/matched
mkdir -p "$OUT"
for M in resnet18 resnet110; do
  echo "=== cyclic-J $M ==="
  python experiments/launch_grid.py --configs configs/match_cyclic.yaml --seeds "$S" \
    --parallel 3 --extra "--set model=$M --set run_name=cyclicJ_$M" --output "$OUT"
  echo "=== strong-SGD $M ==="
  python experiments/launch_grid.py --configs configs/match_strong.yaml --seeds "$S" \
    --parallel 3 --extra "--set model=$M --set run_name=strong_$M" --output "$OUT"
done
echo "=== MATCHED DONE ==="
