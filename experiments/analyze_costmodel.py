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
import os
import statistics as st
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "results"

TARGETS = {
    "c100": [0.78, 0.80, 0.82, 0.83],
    "tiny": [0.65, 0.68, 0.69, 0.70],
}

# display name -> (path under ROOT, dataset)
ARMS = {
    "Temperon (arm P)":   ("phase6/arm_P_handoff", "c100"),
    "late-phase SAM":     ("latesam/c100_latesam", "c100"),
    "full SAM+Muon":      ("bench/c100/c100_sammuon", "c100"),
    "full SAM+SGD":       ("bench/c100/c100_samsgd", "c100"),
    "tuned SGD":          ("bench/c100/c100_strongsgd", "c100"),
    "Temperon tiny":      ("phase6/arm_P_tiny", "tiny"),
    "late-phase SAM tiny": ("latesam/tiny_latesam", "tiny"),
    "full SAM+Muon tiny": ("bench/tiny/tiny_sammuon", "tiny"),
    "full SAM+SGD tiny":  ("bench/tiny/tiny_samsgd", "tiny"),
    "tuned SGD tiny":     ("bench/tiny/tiny_strongsgd", "tiny"),
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

    # --- clean cost model: min epoch time per (dataset, optimizer) ------------
    clean: dict[tuple[str, str], float] = {}
    for _name, (runs, ds) in data.items():
        for epochs in runs.values():
            for ep, _acc, opt, s, _w in epochs:
                if ep == 0:  # epoch 0 includes autotune
                    continue
                key = (ds, opt)
                clean[key] = min(clean.get(key, s), s)

    print("=" * 74)
    print("CLEAN COST MODEL (min observed s/epoch per optimizer, all arms)")
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
    for ds in ["c100", "tiny"]:
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
