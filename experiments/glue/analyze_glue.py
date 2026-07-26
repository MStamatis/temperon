"""Phase 8 statistics: 5-seed GLUE results per task and arm.

Primary metric is the PRE-REGISTERED one: best dev score across epochs
(standard GLUE practice), fixed before any result was inspected. `final_score`
is reported alongside as secondary.

Also computes the free noise floor: `off` and `tail` are bit-identical before
the switch, so on any seed where the tail changes nothing the two arms differ
only by run-to-run nondeterminism.
"""

from __future__ import annotations

import glob
import json
import math
import os
import statistics as st
import sys

TASKS = ["rte", "mrpc", "stsb", "cola"]
ARMS = ["off", "tail", "full"]
METRIC_NAME = {"rte": "acc", "mrpc": "acc+F1/2", "stsb": "pearson", "cola": "mcc"}


def betacf(a, b, x):
    MAXIT, EPS, FPMIN = 200, 3e-12, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (FPMIN if abs(d) < FPMIN else d)
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (FPMIN if abs(d) < FPMIN else d)
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (FPMIN if abs(d) < FPMIN else d)
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            break
    return h


def betai(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                  + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * betacf(a, b, x) / a
    return 1.0 - bt * betacf(b, a, 1.0 - x) / b


def welch(x, y):
    nx, ny = len(x), len(y)
    vx, vy = st.variance(x), st.variance(y)
    se2 = vx / nx + vy / ny
    if se2 == 0:
        return 0.0, float(nx + ny - 2), 1.0
    t = (st.mean(x) - st.mean(y)) / math.sqrt(se2)
    df = se2 ** 2 / ((vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1))
    return t, df, betai(df / 2.0, 0.5, df / (df + t * t))


def load(root):
    data = {}
    for path in sorted(glob.glob(os.path.join(
            glob.escape(root), "*", "*", "*", "result.json"))):
        d = json.load(open(path))
        data.setdefault((d["task"], d["arm"]), []).append(d)
    return data


def main(root):
    data = load(root)
    report(data)


def report(data):
    print("#" * 78)
    print("# PHASE 8 -- GLUE fine-tuning, RoBERTa-base, 5 seeds")
    print("# Primary metric: BEST DEV score across epochs (pre-registered)")
    print("#" * 78)

    for task in TASKS:
        print(f"\n=== {task.upper()}  ({METRIC_NAME[task]}) ===")
        print(f"{'arm':>6} | {'best dev':>17} | {'final':>17} | {'wall':>7} | acc/min")
        for arm in ARMS:
            runs = data.get((task, arm), [])
            if not runs:
                continue
            best = [r["best_score"] for r in runs]
            final = [r["final_score"] for r in runs]
            wall = st.median([r["wall_clock_s"] for r in runs])
            print(f"{arm:>6} | {st.mean(best):.4f} +/- {st.stdev(best):.4f} | "
                  f"{st.mean(final):.4f} +/- {st.stdev(final):.4f} | "
                  f"{wall:6.0f}s | {st.mean(best)/(wall/60):.4f}")

        for a, b in [("tail", "full"), ("tail", "off"), ("full", "off")]:
            if (task, a) in data and (task, b) in data:
                xa = [r["best_score"] for r in data[(task, a)]]
                xb = [r["best_score"] for r in data[(task, b)]]
                t, df, p = welch(xa, xb)
                d = 100 * (st.mean(xa) - st.mean(xb))
                print(f"   {a} vs {b}: {d:+.2f}pp  t={t:+.2f} df={df:.1f} p={p:.3f}")

    # --- aggregate + cost ---------------------------------------------------------
    print("\n" + "#" * 78)
    print("# GLUE AVERAGE (mean over the four tasks of the per-task 5-seed mean)")
    print("#" * 78)
    totals = {}
    for arm in ARMS:
        means, walls = [], []
        for task in TASKS:
            runs = data.get((task, arm), [])
            if runs:
                means.append(st.mean([r["best_score"] for r in runs]))
                walls.append(st.median([r["wall_clock_s"] for r in runs]))
        totals[arm] = (st.mean(means), sum(walls))
        print(f"  {arm:>5}: avg {st.mean(means):.4f}   total wall (4 tasks, 1 seed) "
              f"{sum(walls):5.0f}s")

    if "full" in totals and "tail" in totals:
        dt = 100 * (totals["tail"][1] / totals["full"][1] - 1)
        print(f"\n  tail vs full : {100*(totals['tail'][0]-totals['full'][0]):+.2f}pp "
              f"at {dt:+.0f}% wall-clock")
        dt = 100 * (totals["tail"][1] / totals["off"][1] - 1)
        print(f"  tail vs off  : {100*(totals['tail'][0]-totals['off'][0]):+.2f}pp "
              f"at {dt:+.0f}% wall-clock")

    # --- noise floor --------------------------------------------------------------
    print("\n" + "#" * 78)
    print("# NOISE FLOOR: off vs tail are bit-identical before the switch, so the")
    print("# spread of |off - tail| per seed bounds run-to-run nondeterminism")
    print("#" * 78)
    for task in TASKS:
        off = {r["seed"]: r for r in data.get((task, "off"), [])}
        tail = {r["seed"]: r for r in data.get((task, "tail"), [])}
        pre = []
        for seed in sorted(set(off) & set(tail)):
            start = tail[seed]["sam_start_step"]
            ho = [h for h in off[seed]["history"] if h["step"] <= start]
            ht = [h for h in tail[seed]["history"] if h["step"] <= start]
            for a, b in zip(ho, ht):
                pre.append(abs(a["score"] - b["score"]))
        if pre:
            print(f"  {task:>5}: n={len(pre):3d} pre-switch evals, "
                  f"mean |delta| {st.mean(pre):.4f}, max {max(pre):.4f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/phase8")
