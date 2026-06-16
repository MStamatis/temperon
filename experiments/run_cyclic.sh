#!/bin/sh
# arm J cyclic-catapult (SGDR) at a short horizon (50 epochs), base optimizer
# sgd / adamw / adam, 3 seeds. Compare to strong-SGD @50 and @100 for
# super-convergence (high acc in few epochs).
SEEDS=42,1181241943,958682846
OUT=results/cyclic50
mkdir -p "$OUT"
echo "=== J cyclic SGD ==="
python experiments/launch_grid.py --configs configs/arm_J_cyclic.yaml \
  --seeds "$SEEDS" --epochs 50 --parallel 3 --output "$OUT"
echo "=== J cyclic AdamW ==="
python experiments/launch_grid.py --configs configs/arm_J_cyclic.yaml \
  --seeds "$SEEDS" --epochs 50 --parallel 3 \
  --extra "--set controller.base_optimizer=adamw --set controller.lr=1.0e-3 --set run_name=arm_J_adamw" \
  --output "$OUT"
echo "=== J cyclic Adam ==="
python experiments/launch_grid.py --configs configs/arm_J_cyclic.yaml \
  --seeds "$SEEDS" --epochs 50 --parallel 3 \
  --extra "--set controller.base_optimizer=adam --set controller.lr=1.0e-3 --set run_name=arm_J_adam" \
  --output "$OUT"
echo "=== CYCLIC DONE ==="
