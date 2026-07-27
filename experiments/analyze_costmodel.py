"""Contention-robust time-to-target: epochs-to-target x a clean cost model.

Raw wall-clock across our grids is contaminated by background GPU load: the
same two seeds are the fast ones in EVERY arm, including bench runs recorded on
a different day, so the fast/slow split tracks machine state and not the
method. Epochs-to-target is immune to that, and the per-epoch cost of each
optimizer is a property of the optimizer, not of the run.

So we report target x arm as:

    predicted_seconds = sum over phases of (epochs in phase) x (clean s/epoch)

where the clean s/epoch is the MINIMUM observed epoch time for that optimizer
across every seed of every arm on that dataset -- the closest thing we have to
an uncontended measurement. Raw medians are printed alongside, and the ratio
between them is the contention factor each arm actually suffered.
"""

from __future__ import annotations

import csv
import glob
import json
import os
import statistics as st
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "results"

TARGETS = {
    "c100": [0.78, 0.80, 0.82, 0.83],
    "tiny": [0.65, 0.68, 0.69, 0.70],
    "c10": [0.96, 0.965, 0.968, 0.97],
    "svhn": [0.96, 0.97, 0.975, 0.98],
}

# display name -> (path under ROOT, dataset)
ARMS = {
    "Temperon (arm P)":   ("phase6/arm_P_handoff", "c100"),
    "arm S (SGD tail)":   ("phase6/arm_P_sgdtail", "c100"),
    "late-phase SAM":     ("latesam/c100_latesam", "c100"),
    "full SAM+Muon":      ("bench/c100/c100_sammuon", "c100"),
    "full SAM+SGD":       ("bench/c100/c100_samsgd", "c100"),
    "tuned SGD":          ("bench/c100/c100_strongsgd", "c100"),
    "Temperon tiny":      ("phase6/arm_P_tiny", "tiny"),
    "late-phase SAM tiny": ("latesam/tiny_latesam", "tiny"),
    "full SAM+Muon tiny": ("bench/tiny/tiny_sammuon", "tiny"),
    "full SAM+SGD tiny":  ("bench/tiny/tiny_samsgd", "tiny"),
    "tuned SGD tiny":     ("bench/tiny/tiny_strongsgd", "tiny"),
    "Temperon c10":       ("phase6/arm_P_c10", "c10"),
    "full SAM+Muon c10":  ("bench/c10/c10_sammuon", "c10"),
    "full SAM+SGD c10":   ("bench/c10/c10_samsgd", "c10"),
    "tuned SGD c10":      ("bench/c10/c10_strongsgd", "c10"),
    "Temperon svhn":      ("phase6/arm_P_svhn", "svhn"),
    "full SAM+Muon svhn": ("bench/svhn/svhn_sammuon", "svhn"),
    "full SAM+SGD svhn":  ("bench/svhn/svhn_samsgd", "svhn"),
    "tuned SGD svhn":     ("bench/svhn/svhn_strongsgd", "svhn"),
}


def read_epochs(run_dir):
    out = []
    with open(os.path.join(run_dir, "epochs.csv"), newline="") as f:
        for row in csv.DictReader(f):
            out.append((int(row["epoch"]), float(row["val_acc"]),
                        row["optimizer"], float(row["epoch_time_s"]),
                        float(row["wall_clock_s"])))
    return out


def load(sub):
    pat = os.path.join(glob.escape(os.path.join(ROOT, *sub.split("/"))), "seed*")
    return {os.path.basename(d): read_epochs(d) for d in sorted(glob.glob(pat))}


def main():
    data = {name: (load(sub), ds) for name, (sub, ds) in ARMS.items()}
    data = {k: v for k, v in data.items() if v[0]}

    # --- clean cost model ----------------------------------------------------
    # Preferred: a calibration measured on an idle GPU (calibrate_cost.py).
    # Fallback: the minimum epoch time observed anywhere, which is the closest
    # thing the existing grids offer to an uncontended measurement.
    clean: dict[tuple[str, str], float] = {}
    calib_path = os.path.join(ROOT, "costmodel", "calibration.json")
    calibrated = os.path.exists(calib_path)
    if calibrated:
        with open(calib_path) as f:
            for ds, opts in json.load(f).items():
                for opt, e in opts.items():
                    clean[(ds, opt)] = e["median_s"]

    # Calibrated entries are authoritative; everything else falls back to the
    # minimum observed. Guard on the calibrated key set, not on `key in clean`
    # -- the latter accepts the FIRST fallback value and then blocks its own
    # minimum from ever being taken.
    measured = set(clean)
    for _name, (runs, ds) in data.items():
        for epochs in runs.values():
            for ep, _acc, opt, s, _w in epochs:
                if ep == 0:  # epoch 0 includes autotune
                    continue
                key = (ds, opt)
                if key not in measured:
                    clean[key] = min(clean.get(key, s), s)

    print("=" * 74)
    if calibrated:
        print(f"COST MODEL (calibrated on an idle GPU: {calib_path})")
    else:
        print("COST MODEL (min observed s/epoch per optimizer, all arms)")
        print("  -- fallback estimate; run calibrate_cost.py for measured costs")
    print("=" * 74)
    for (ds, opt), s in sorted(clean.items()):
        print(f"  {ds:>5} {opt:<18} {s:6.1f} s/epoch")

    # --- contention factor actually suffered, per arm ------------------------
    print()
    print("=" * 74)
    print("CONTENTION FACTOR (median observed s/epoch / clean, per arm)")
    print("=" * 74)
    for name, (runs, ds) in data.items():
        obs: dict[str, list[float]] = {}
        for epochs in runs.values():
            for ep, _acc, opt, s, _w in epochs:
                if ep:
                    obs.setdefault(opt, []).append(s)
        cells = "  ".join(
            f"{opt}={st.median(v)/clean[(ds, opt)]:.2f}x" for opt, v in obs.items())
        print(f"  {name:<22} {cells}")

    # --- epochs to target + predicted clean seconds ---------------------------
    for ds in ["c100", "tiny", "c10", "svhn"]:
        arms = [(n, r) for n, (r, d) in data.items() if d == ds]
        if not arms:
            continue
        print()
        print("=" * 74)
        print(f"TIME TO TARGET -- {ds}   (epochs; predicted s at clean cost; raw s)")
        print("=" * 74)
        for t in TARGETS[ds]:
            print(f"\n  target {t}")
            for name, runs in arms:
                eps, pred, raw = [], [], []
                for epochs in runs.values():
                    hit = next((e for e in epochs if e[1] >= t), None)
                    if hit is None:
                        continue
                    eps.append(hit[0])
                    raw.append(hit[4])
                    pred.append(sum(clean[(ds, opt)]
                                    for ep, _a, opt, _s, _w in epochs
                                    if 0 < ep <= hit[0]))
                n = len(runs)
                if not eps:
                    print(f"    {name:<22}  --  0/{n} seeds")
                    continue
                print(f"    {name:<22} ep{st.median(eps):5.1f}  "
                      f"{st.median(pred):6.0f}s clean  "
                      f"{st.median(raw):6.0f}s raw   {len(eps)}/{n} seeds")


if __name__ == "__main__":
    main()
