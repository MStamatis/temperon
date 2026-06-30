"""Aggregate a benchmark results dir into a comparison CSV + plots.

Reads results/<dir>/<arm>/seed*/{summary.csv, metrics.json, epochs.csv}, groups
by arm, aggregates mean +/- std across seeds, and writes:
  - comparison.csv        (arm x metric, mean & std)
  - <ds>_accuracy_f1.png  (grouped bar chart, error bars)
  - <ds>_curves.png       (val/test accuracy vs epoch, one line per arm)

Usage:  python experiments/benchmark_report.py results/bench/c100
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

METRICS = ["test_accuracy", "f1_macro", "precision_macro", "recall_macro",
           "roc_auc_macro_ovr", "best_val_acc"]
ARM_ORDER = ["sammuon", "samsgd", "muon", "cyclicj", "strongsgd"]
ARM_LABEL = {"sammuon": "SAM+Muon (ours)", "samsgd": "SAM+SGD [Foret'21]",
             "muon": "Muon [Jordan'24]", "cyclicj": "cyclic-J [SGDR'17]",
             "strongsgd": "strong-SGD"}


def _fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _arm_key(run_name: str) -> str:
    return run_name.split("_", 1)[1] if "_" in run_name else run_name


def load_runs(results_dir: str) -> dict:
    runs: dict[str, list] = {}
    for summ in glob.glob(os.path.join(results_dir, "*", "seed*", "summary.csv")):
        d = os.path.dirname(summ)
        row = list(csv.DictReader(open(summ)))[0]
        rec = dict(row)
        mp = os.path.join(d, "metrics.json")
        if os.path.exists(mp):
            rec.update(json.load(open(mp)))
        ep = os.path.join(d, "epochs.csv")
        if os.path.exists(ep):
            rec["_curve"] = [(int(float(r["epoch"])), float(r["val_acc"]))
                             for r in csv.DictReader(open(ep))]
        runs.setdefault(_arm_key(row["run_name"]), []).append(rec)
    return runs


def aggregate(runs: dict) -> list[dict]:
    rows = []
    for arm in [a for a in ARM_ORDER if a in runs] + [a for a in runs if a not in ARM_ORDER]:
        recs = runs[arm]
        agg = {"arm": arm, "label": ARM_LABEL.get(arm, arm), "n_seeds": len(recs)}
        for m in METRICS:
            vals = [v for v in (_fnum(r.get(m)) for r in recs) if v is not None]
            if vals:
                agg[f"{m}_mean"] = round(st.mean(vals), 4)
                agg[f"{m}_std"] = round(st.pstdev(vals), 4) if len(vals) > 1 else 0.0
        rows.append(agg)
    return rows


def write_csv(rows: list[dict], path: str) -> None:
    cols = ["arm", "label", "n_seeds"] + [f"{m}_{s}" for m in METRICS for s in ("mean", "std")]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot_bars(rows: list[dict], path: str, ds: str) -> None:
    arms = [r["label"] for r in rows]
    x = range(len(arms))
    fig, ax = plt.subplots(figsize=(max(7, len(arms) * 1.6), 4.5))
    for i, (m, off, col) in enumerate([("test_accuracy", -0.2, "#4C72B0"), ("f1_macro", 0.2, "#DD8452")]):
        means = [r.get(f"{m}_mean", 0) for r in rows]
        stds = [r.get(f"{m}_std", 0) for r in rows]
        ax.bar([xi + off for xi in x], means, width=0.38, yerr=stds, capsize=3,
               label=m, color=col)
    ax.set_xticks(list(x))
    ax.set_xticklabels(arms, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("score")
    ax.set_title(f"{ds}: test accuracy & macro-F1 (mean +/- std)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_curves(runs: dict, path: str, ds: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for arm in [a for a in ARM_ORDER if a in runs] + [a for a in runs if a not in ARM_ORDER]:
        curves = [r["_curve"] for r in runs[arm] if "_curve" in r]
        if not curves:
            continue
        n = min(len(c) for c in curves)
        mean = [st.mean(c[e][1] for c in curves) for e in range(n)]
        ax.plot(range(n), mean, label=ARM_LABEL.get(arm, arm), linewidth=1.6)
    ax.set_xlabel("epoch")
    ax.set_ylabel("eval accuracy")
    ax.set_title(f"{ds}: accuracy vs epoch (mean over seeds)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    args = ap.parse_args()
    ds = os.path.basename(os.path.normpath(args.results_dir))
    runs = load_runs(args.results_dir)
    if not runs:
        raise SystemExit(f"no runs found under {args.results_dir!r}")
    rows = aggregate(runs)
    write_csv(rows, os.path.join(args.results_dir, "comparison.csv"))
    plot_bars(rows, os.path.join(args.results_dir, f"{ds}_accuracy_f1.png"), ds)
    plot_curves(runs, os.path.join(args.results_dir, f"{ds}_curves.png"), ds)
    print(f"[{ds}] {len(runs)} arms, {sum(len(v) for v in runs.values())} runs")
    print(f"  -> comparison.csv, {ds}_accuracy_f1.png, {ds}_curves.png")
    for r in rows:
        print(f"  {r['label']:<22} acc={r.get('test_accuracy_mean','?')}"
              f"+/-{r.get('test_accuracy_std','?')}  f1={r.get('f1_macro_mean','?')}"
              f"  auc={r.get('roc_auc_macro_ovr_mean','?')}  (n={r['n_seeds']})")


if __name__ == "__main__":
    main()
