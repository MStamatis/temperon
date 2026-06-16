#!/bin/sh
# Tune arm J (cyclic SGD) to reach 0.75 by ~epoch 30 (OptiRoulette level).
# Growing cycles (t_mult=2 -> long deep-annealing final cycle) at short horizons,
# plus a pure-cosine-30 baseline (n_cycles=1). 3 seeds each.
S=42,1181241943,958682846
OUT=results/cyclic_tune
mkdir -p "$OUT"
run() {  # $1 run_name  $2 epochs  $3 n_cycles  $4 t_mult
  python experiments/launch_grid.py --configs configs/arm_J_cyclic.yaml --seeds "$S" \
    --epochs "$2" --parallel 3 \
    --extra "--set controller.n_cycles=$3 --set controller.t_mult=$4 --set run_name=$1" \
    --output "$OUT"
}
echo "=== J e30 c3 t2 ===";        run J_e30_c3_t2 30 3 2
echo "=== J e35 c3 t2 ===";        run J_e35_c3_t2 35 3 2
echo "=== J e40 c4 t2 ===";        run J_e40_c4_t2 40 4 2
echo "=== J e30 c1 (cosine) ===";  run J_e30_c1 30 1 1
echo "=== TUNE DONE ==="
