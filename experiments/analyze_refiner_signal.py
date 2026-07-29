"""Phase 9b replay: can early observables predict the refiner choice?

Pre-registered in phase9b_refiner_prereg.md (commit d3746e9). Signals, labels
and read-out are locked there; this script executes them.

Usage:
    python experiments/analyze_refiner_signal.py > results/analysis_refiner.txt
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ROOT = Path("results/bench")
SEEDS = ["42", "1181241943", "958682846", "271828", "314159"]
DATASETS = ["c100", "tiny", "c10", "svhn"]
LABELS = {"c100": "+0.60pp YES", "c10": "+0.26pp YES",
          "svhn": "+0.06pp YES", "tiny": "-1.89pp NO"}
H_FRAC, H2_FRAC = 0.43, 0.21


def envelope(dataset: str, arm: str, seed: str):
    f = ROOT / dataset / f"{dataset}_{arm}" / f"seed{seed}" / "epochs.csv"
    if not f.exists():
        return None
    acc = [float(r["val_acc"]) for r in csv.DictReader(open(f))]
    return np.maximum.accumulate(acc)


def main():
    print("PHASE 9B REPLAY -- pre-registered in phase9b_refiner_prereg.md")
    values: dict[str, dict[str, list[float]]] = {
        s: {ds: [] for ds in DATASETS} for s in ("S1", "S2", "S3", "D")
    }
    for ds in DATASETS:
        for seed in SEEDS:
            mu, sg = envelope(ds, "muon", seed), envelope(ds, "strongsgd", seed)
            if mu is None or sg is None:
                continue
            n = min(len(mu), len(sg))
            h, h2 = int(H_FRAC * n), int(H2_FRAC * n)
            values["S1"][ds].append(mu[h] - sg[h])
            values["S2"][ds].append(mu[h] / sg[h])
            values["S3"][ds].append((mu[h] - mu[h2]) - (sg[h] - sg[h2]))
            values["D"][ds].append(mu[h] / sg[-1])

    for sig in ("S1", "S2", "S3", "D"):
        print(f"\n-- {sig} " + ("(diagnostic, not observable online)" if sig == "D" else ""))
        for ds in DATASETS:
            v = np.array(values[sig][ds])
            print(f"  {ds:>5} ({LABELS[ds]:>11}): "
                  f"mean {v.mean():+.4f}  range [{v.min():+.4f}, {v.max():+.4f}]  n={len(v)}")
        tiny = np.array(values[sig]["tiny"])
        rest = np.concatenate([values[sig][d] for d in DATASETS if d != "tiny"])
        if tiny.max() < rest.min():
            verdict = "SEPARATES, tiny LOW -> required direction"
        elif tiny.min() > rest.max():
            verdict = "SEPARATES, tiny HIGH -> INVERTED (not usable per prereg)"
        else:
            verdict = "OVERLAPS -> no signal"
        print(f"  verdict: {verdict}")

    print("\nRead-out per prereg: SIGNAL-CANDIDATE requires an observable signal")
    print("(S1-S3) separating with tiny LOW. See verdicts above.")


if __name__ == "__main__":
    main()
