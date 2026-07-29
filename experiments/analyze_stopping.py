"""Phase 9 offline replay: the calibrated stopping rule on logged trajectories.

Pre-registered in phase9_stopping_prereg.md (commit 3637e95) BEFORE this tool
ran. The rule, grid, substrates and read-out criteria are locked there; this
script only executes them.

Usage:
    python experiments/analyze_stopping.py > results/analysis_stopping.txt
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ROOT = Path("results")

# Locked grid (prereg).
WINDOWS = [5, 8]
PATIENCES = [2, 3]
CS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
T_MIN_FRAC, T_MAX_FRAC = 0.15, 0.85
THETA_CREDIT = 0.46 / 37 / 100.0  # 0.0125pp/epoch, in accuracy units

# Hard constraints (prereg).
C100_WINDOW = (35, 55)
LM_WINDOW = (0.55, 0.80)
SEED_SPREAD_MAX = 12  # epochs

SEEDS = ["42", "1181241943", "958682846", "271828", "314159"]
DATASETS = ["c100", "tiny", "c10", "svhn"]
VISION_ARMS = {"strongsgd": "primary", "cyclicj": "secondary"}


def load_vision(dataset: str, arm: str, seed: str):
    """Return (epochs, val_acc) arrays or None if the run is absent."""
    f = ROOT / "bench" / dataset / f"{dataset}_{arm}" / f"seed{seed}" / "epochs.csv"
    if not f.exists():
        return None
    rows = list(csv.DictReader(open(f)))
    ep = np.array([int(r["epoch"]) for r in rows])
    acc = np.array([float(r["val_acc"]) for r in rows])
    return ep, acc


def load_lm():
    """Return (budget_frac, val_loss) from the pure-Muon WSD run."""
    f = ROOT / "phase7" / "lm_muon" / "seed42" / "evals.csv"
    rows = list(csv.DictReader(open(f)))
    tok = np.array([int(r["tokens"]) for r in rows])
    loss = np.array([float(r["val_loss"]) for r in rows])
    return tok / tok[-1], loss


def replay(x: np.ndarray, metric: np.ndarray, W: int, K: int, c: float,
           maximize: bool, theta_abs: float | None = None):
    """Fire index under the pre-registered rule, or None.

    x is the budget axis (epochs or budget fraction), metric the val curve.
    The envelope (running max/min) absorbs cyclic oscillation. rate and
    avg_rate are slopes on that envelope; the rule fires after K consecutive
    evals with rate < c * avg_rate inside the guardrails. When theta_abs is
    given it replaces c * avg_rate (the vision-only credit form).
    """
    env = np.maximum.accumulate(metric) if maximize else np.minimum.accumulate(metric)
    sign = 1.0 if maximize else -1.0
    t_min = x[0] + T_MIN_FRAC * (x[-1] - x[0])
    t_max = x[0] + T_MAX_FRAC * (x[-1] - x[0])
    i_min = int(np.searchsorted(x, t_min))
    streak = 0
    for i in range(max(i_min, W), len(x)):
        if x[i] > t_max:
            break
        w_x, w_y = x[i - W + 1 : i + 1], env[i - W + 1 : i + 1]
        rate = sign * np.polyfit(w_x, w_y, 1)[0]
        span = x[i] - x[i_min]
        avg_rate = sign * (env[i] - env[i_min]) / span if span > 0 else np.inf
        threshold = theta_abs if theta_abs is not None else c * max(avg_rate, 0.0)
        streak = streak + 1 if rate < threshold else 0
        if streak >= K:
            return i
    return None


def vision_cell(arm: str, W: int, K: int, c: float, theta_abs=None):
    """Median fire epoch and seed spread per dataset for one grid cell."""
    out = {}
    for ds in DATASETS:
        fires = []
        for seed in SEEDS:
            run = load_vision(ds, arm, seed)
            if run is None:
                continue
            ep, acc = run
            i = replay(ep.astype(float), acc, W, K, c, maximize=True,
                       theta_abs=theta_abs)
            fires.append(ep[i] if i is not None else None)
        hit = [f for f in fires if f is not None]
        out[ds] = {
            "n": len(fires), "fired": len(hit),
            "median": float(np.median(hit)) if hit else None,
            "spread": float(max(hit) - min(hit)) if len(hit) > 1 else 0.0,
        }
    return out


def main():
    print("=" * 72)
    print("PHASE 9 OFFLINE REPLAY -- pre-registered in phase9_stopping_prereg.md")
    print("=" * 72)

    lm_x, lm_loss = load_lm()

    passing = []
    for arm, role in VISION_ARMS.items():
        print(f"\n---- substrate: {arm} ({role}) " + "-" * 30)
        header = f"{'W':>3} {'K':>3} {'c':>5} |"
        for ds in DATASETS:
            header += f" {ds:>13}"
        header += f" {'LM fire':>9} | verdict"
        print(header)
        for W in WINDOWS:
            for K in PATIENCES:
                for c in CS:
                    cells = vision_cell(arm, W, K, c)
                    i_lm = replay(lm_x, lm_loss, W, K, c, maximize=False)
                    lm_fire = lm_x[i_lm] if i_lm is not None else None

                    m100 = cells["c100"]["median"]
                    ok_c100 = m100 is not None and C100_WINDOW[0] <= m100 <= C100_WINDOW[1]
                    ok_lm = lm_fire is not None and LM_WINDOW[0] <= lm_fire <= LM_WINDOW[1]
                    ok_spread = all(
                        v["spread"] <= SEED_SPREAD_MAX for v in cells.values() if v["fired"]
                    )
                    verdict = ("PASS" if ok_c100 and ok_lm and ok_spread else
                               "c100" * (not ok_c100) + " lm" * (not ok_lm) +
                               " spread" * (not ok_spread))

                    row = f"{W:>3} {K:>3} {c:>5.2f} |"
                    for ds in DATASETS:
                        v = cells[ds]
                        cell = ("--" if v["median"] is None
                                else f"{v['median']:.0f} (+-{v['spread']:.0f},{v['fired']}/{v['n']})")
                        row += f" {cell:>13}"
                    lm_cell = "--" if lm_fire is None else f"{lm_fire:.2f}"
                    row += f" {lm_cell:>9} | {verdict}"
                    print(row)
                    if verdict == "PASS" and role == "primary":
                        passing.append((W, K, c, cells, lm_fire))

    print(f"\n---- credit form (vision-only, theta = {THETA_CREDIT*100:.4f}pp/ep, strongsgd)")
    for W in WINDOWS:
        for K in PATIENCES:
            cells = vision_cell("strongsgd", W, K, 0.0, theta_abs=THETA_CREDIT)
            row = f"{W:>3} {K:>3}      |"
            for ds in DATASETS:
                v = cells[ds]
                cell = ("--" if v["median"] is None
                        else f"{v['median']:.0f} (+-{v['spread']:.0f},{v['fired']}/{v['n']})")
                row += f" {cell:>13}"
            print(row)

    print("\n" + "=" * 72)
    if passing:
        print(f"OUTCOME: SUPPORTED -- {len(passing)} primary-substrate cell(s) pass all")
        print("hard constraints. c10/svhn/tiny medians of those cells are PREDICTIONS")
        print("for a future live run, not results.")
        for W, K, c, cells, lm_fire in passing:
            preds = ", ".join(
                f"{ds}@{cells[ds]['median']:.0f}" for ds in ("c10", "svhn", "tiny")
                if cells[ds]["median"] is not None
            )
            print(f"  W={W} K={K} c={c:.2f}: c100@{cells['c100']['median']:.0f}, "
                  f"LM@{lm_fire:.2f}, predicts {preds}")
    else:
        print("OUTCOME: no primary cell passes both hard constraints -->")
        print("PREDICTIVE-ONLY or FAILED per prereg; see rows above.")


if __name__ == "__main__":
    main()
