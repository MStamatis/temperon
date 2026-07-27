"""Measure per-epoch cost of each optimizer on a quiet machine.

Wall-clock recorded during a 100-epoch grid is only as trustworthy as the
machine was idle, and ours was not: the same seeds are the fast ones in every
arm, so the spread tracks background load rather than the method. Re-running
the grids would cost ~40h per dataset and would still be a single machine's
numbers, which no reader can reproduce on their own GPU anyway.

The transferable decomposition is

    time to target = (epochs to target) x (seconds per epoch)

where the left factor is algorithmic -- already measured, immune to load -- and
the right factor is a property of the optimizer. This script measures the right
factor directly: a few epochs of each bench config, which between them cover
every optimizer the paper reports (sgd_momentum, muon, sam:sgd_momentum,
sam:muon).

RUN IT ON AN IDLE GPU. The desktop counts: if the GPU is driving a monitor,
compositing and a browser cost 20-28% of every epoch. Detach the display (any
second adapter will do) or leave the machine alone. Other containers must be
stopped either way -- `nvidia-smi` should show this process alone.

Finished measurements are kept, so datasets can be calibrated one at a time as
the machine frees up; delete the output directory to force a fresh one.

    python experiments/calibrate_cost.py --datasets c100
    python experiments/calibrate_cost.py --datasets tiny c10 svhn
    python experiments/calibrate_cost.py --dry-run

Writes results/costmodel/calibration.json, which analyze_costmodel.py picks up
automatically in preference to its min-observed fallback.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics as st
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# One bench config per optimizer. These run the optimizer for the whole run, so
# every timed epoch is that optimizer -- no phase boundary to reason about.
CONFIGS = ["strongsgd", "muon", "samsgd", "sammuon"]
DATASETS = ["c100", "tiny", "c10", "svhn"]


def run_dir(out: str, ds: str, name: str) -> str:
    return os.path.join(out, f"{ds}_{name}", "seed42")


def epoch_times(path: str) -> dict[str, list[float]]:
    """Per-optimizer epoch times, dropping epoch 0 (includes autotune)."""
    per: dict[str, list[float]] = {}
    with open(os.path.join(path, "epochs.csv"), newline="") as f:
        for row in csv.DictReader(f):
            if int(row["epoch"]) == 0:
                continue
            per.setdefault(row["optimizer"], []).append(float(row["epoch_time_s"]))
    return per


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["c100", "tiny"],
                    choices=DATASETS)
    ap.add_argument("--epochs", type=int, default=6,
                    help="epochs per config; epoch 0 is discarded, so this "
                         "many minus one are timed")
    ap.add_argument("--output", default="results/costmodel")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    jobs = [(ds, name) for ds in args.datasets for name in CONFIGS]
    print(f"{len(jobs)} calibration runs x {args.epochs} epochs")
    if args.dry_run:
        for ds, name in jobs:
            print(f"  configs/bench/{ds}_{name}.yaml")
        return

    t0 = time.perf_counter()
    for i, (ds, name) in enumerate(jobs, 1):
        cfg = os.path.join(ROOT, "configs", "bench", f"{ds}_{name}.yaml")
        print(f"\n=== [{i}/{len(jobs)}] {ds}_{name} "
              f"(elapsed {(time.perf_counter()-t0)/60:.0f} min) ===", flush=True)
        r = subprocess.run([
            sys.executable, os.path.join(HERE, "run.py"),
            "--config", cfg, "--seed", "42",
            "--set", f"epochs={args.epochs}",
            "--output", args.output,
            # Calibrating one dataset at a time is the normal way to use this,
            # so finished measurements must survive the next invocation. Delete
            # the output directory to force a fresh measurement.
            "--continue",
        ])
        if r.returncode != 0:
            print(f"FAILED: {ds}_{name} (exit {r.returncode})")
            sys.exit(r.returncode)

    # --- collect ------------------------------------------------------------
    # Scan every measurement on disk, not just this invocation's jobs: writing
    # the file from `jobs` alone would drop the datasets calibrated in earlier
    # calls, which is the normal way to use this on a machine that is only free
    # in short windows.
    table: dict[str, dict[str, dict]] = {}
    for ds in DATASETS:
        for name in CONFIGS:
            d = run_dir(args.output, ds, name)
            if not os.path.exists(os.path.join(d, "epochs.csv")):
                continue
            for opt, times in epoch_times(d).items():
                prev = table.setdefault(ds, {}).get(opt)
                entry = {
                    "median_s": st.median(times),
                    "min_s": min(times),
                    "max_s": max(times),
                    "n": len(times),
                    "source": f"{ds}_{name}",
                }
                # Two configs can share an optimizer; keep the quieter one.
                if prev is None or entry["median_s"] < prev["median_s"]:
                    table[ds][opt] = entry

    os.makedirs(args.output, exist_ok=True)
    path = os.path.join(args.output, "calibration.json")
    with open(path, "w") as f:
        json.dump(table, f, indent=2, sort_keys=True)

    print(f"\nwrote {path}")
    print(f"{'dataset':>7} {'optimizer':<18} {'median':>8} {'min':>8} {'max':>8}"
          f"  spread")
    for ds in sorted(table):
        for opt in sorted(table[ds]):
            e = table[ds][opt]
            spread = 100 * (e["max_s"] / e["min_s"] - 1)
            # A handful of epochs jitter a few percent on an idle GPU; real
            # contention showed up as 30-50% in the grids that provoked this
            # script, so flag well above the noise rather than inside it.
            flag = "  <- NOISY, machine was not idle" if spread > 12 else ""
            print(f"{ds:>7} {opt:<18} {e['median_s']:7.1f}s {e['min_s']:7.1f}s "
                  f"{e['max_s']:7.1f}s  {spread:4.1f}%{flag}")
    print(f"\ntotal {(time.perf_counter()-t0)/60:.0f} min")


if __name__ == "__main__":
    main()
