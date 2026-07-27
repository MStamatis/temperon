# 5-seed preprint statistics for phase6 arms vs bench baselines.
# Pure stdlib: csv, json, glob, statistics, math.
import csv, json, glob, os, math, sys, statistics as st

ROOT = sys.argv[1] if len(sys.argv) > 1 else "results"

TARGETS = {
    "c100": [0.75, 0.78, 0.80, 0.82, 0.83],
    "tiny": [0.60, 0.65, 0.68, 0.69, 0.70],
    "c10":  [0.94, 0.96, 0.965, 0.968, 0.97],
    "svhn": [0.95, 0.96, 0.97, 0.975, 0.98],
}

ARMS = {  # display name -> (path under ROOT, dataset)
    "arm_P_handoff":     ("phase6/arm_P_handoff", "c100"),
    "arm_P_handoff_e80": ("phase6/arm_P_handoff_e80", "c100"),
    "c100_sammuon_p2":   ("phase6/c100_sammuon_p2", "c100"),
    # arm S: arm P with an SGD refiner -- isolates refiner from allocation shape
    "arm_S_sgdtail":     ("phase6/arm_P_sgdtail", "c100"),
    "arm_P_tiny":        ("phase6/arm_P_tiny", "tiny"),
    "arm_P_c10":         ("phase6/arm_P_c10", "c10"),
    "arm_P_svhn":        ("phase6/arm_P_svhn", "svhn"),
    # arm R: late-phase SAM (2410.10373) re-run in this pipeline. Absent dirs
    # are skipped, so this stays correct while the sweep is still running.
    "latesam_c100":      ("latesam/c100_latesam", "c100"),
    "latesam_tiny":      ("latesam/tiny_latesam", "tiny"),
    "latesam_c10":       ("latesam/c10_latesam", "c10"),
    "latesam_svhn":      ("latesam/svhn_latesam", "svhn"),
}
BENCH_ARMS = ["sammuon", "samsgd", "cyclicj", "muon", "strongsgd"]


def load_run(run_dir):
    epochs = []
    with open(os.path.join(run_dir, "epochs.csv"), newline="") as f:
        for row in csv.DictReader(f):
            epochs.append({
                "epoch": int(row["epoch"]),
                "val_acc": float(row["val_acc"]),
                "opt": row["optimizer"],
                "ep_s": float(row["epoch_time_s"]),
                "wall": float(row["wall_clock_s"]),
            })
    final = None
    mpath = os.path.join(run_dir, "metrics.json")
    if os.path.exists(mpath):
        with open(mpath) as f:
            final = json.load(f).get("accuracy")
    return epochs, final


def first_hit(epochs, target):
    for e in epochs:
        if e["val_acc"] >= target:
            return e["epoch"], e["wall"]
    return None, None


def per_opt_median_s(epochs):
    """Median epoch_time_s per optimizer, excluding epoch 0 (autotune)."""
    d = {}
    for e in epochs:
        if e["epoch"] == 0:
            continue
        d.setdefault(e["opt"], []).append(e["ep_s"])
    return {k: st.median(v) for k, v in d.items()}


def betacf(a, b, x):
    MAXIT, EPS, FPMIN = 200, 3e-12, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < FPMIN: d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN: d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN: c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN: d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN: c = FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            break
    return h


def betai(a, b, x):
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                  + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * betacf(a, b, x) / a
    return 1.0 - bt * betacf(b, a, 1.0 - x) / b


def welch(x, y):
    nx, ny = len(x), len(y)
    mx, my = st.mean(x), st.mean(y)
    vx, vy = st.variance(x), st.variance(y)
    se2 = vx / nx + vy / ny
    t = (mx - my) / math.sqrt(se2)
    df = se2 ** 2 / ((vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1))
    p = betai(df / 2.0, 0.5, df / (df + t * t))  # two-sided
    return t, df, p


def collect(pattern):
    # ROOT contains "[D]" which glob would eat as a character class.
    base, tail = os.path.split(pattern)
    pattern = os.path.join(glob.escape(base), tail)
    runs = {}
    for run_dir in sorted(glob.glob(pattern)):
        seed = os.path.basename(run_dir).replace("seed", "")
        try:
            runs[seed] = load_run(run_dir)
        except FileNotFoundError:
            print(f"  !! missing files in {run_dir}")
    return runs


def summarize(name, runs, ds):
    targets = TARGETS[ds]
    finals, bests, totals = [], [], []
    hits = {t: [] for t in targets}       # wall-clock of first hit
    hit_eps = {t: [] for t in targets}    # epoch of first hit
    opt_s_all = {}
    per_seed_lines = []
    for seed, (epochs, final) in sorted(runs.items()):
        best = max(e["val_acc"] for e in epochs)
        total = epochs[-1]["wall"]
        finals.append(final if final is not None else best)
        bests.append(best)
        totals.append(total)
        hstr = []
        for t in targets:
            ep, w = first_hit(epochs, t)
            if w is not None:
                hits[t].append(w)
                hit_eps[t].append(ep)
                hstr.append(f"{t}@{w:.0f}s/ep{ep}")
        for k, v in per_opt_median_s(epochs).items():
            opt_s_all.setdefault(k, []).append(v)
        per_seed_lines.append(
            f"    seed{seed:>10}: final={finals[-1]:.4f} best={best:.4f} "
            f"total={total:.0f}s | " + " ".join(hstr))
    n = len(finals)
    print(f"\n== {name} ({ds}, n={n}) ==")
    for line in per_seed_lines:
        print(line)
    mu, sd = st.mean(finals), (st.stdev(finals) if n > 1 else 0.0)
    bmu, bsd = st.mean(bests), (st.stdev(bests) if n > 1 else 0.0)
    print(f"  FINAL test: {mu:.4f} +/- {sd:.4f}   BEST val: {bmu:.4f} +/- {bsd:.4f}"
          f"   total med {st.median(totals):.0f}s")
    for t in targets:
        k = len(hits[t])
        if k:
            print(f"  hit {t}: {k}/{n} seeds, median {st.median(hits[t]):.0f}s"
                  f" / ep{st.median(hit_eps[t]):.1f}")
        else:
            print(f"  hit {t}: 0/{n} seeds")
    med_opt = {k: st.median(v) for k, v in opt_s_all.items()}
    print("  s/ep medians: " + "  ".join(f"{k}={v:.1f}" for k, v in med_opt.items()))
    return finals, totals, hits


all_stats = {}

print("#" * 70)
print("# CONTROLLED ARMS (5-seed)")
print("#" * 70)
for arm, (sub, ds) in ARMS.items():
    runs = collect(os.path.join(ROOT, *sub.split("/"), "seed*"))
    if not runs:
        continue
    all_stats[arm] = summarize(arm, runs, ds) + (ds,)

print()
print("#" * 70)
print("# BENCH BASELINES (5-seed)")
print("#" * 70)
for ds in ["c100", "tiny", "c10", "svhn"]:
    for b in BENCH_ARMS:
        pat = os.path.join(ROOT, "bench", ds, f"{ds}_{b}", "seed*")
        runs = collect(pat)
        if runs:
            all_stats[f"{ds}_{b}"] = summarize(f"{ds}_{b}", runs, ds) + (ds,)

print()
print("#" * 70)
print("# HEAD-TO-HEAD (Welch two-sided on FINAL test acc)")
print("#" * 70)
pairs = [
    ("arm_P_handoff", "c100_sammuon"),
    ("arm_P_handoff", "c100_samsgd"),
    ("arm_P_handoff", "c100_sammuon_p2"),
    ("arm_P_handoff_e80", "c100_sammuon"),
    ("arm_P_handoff_e80", "c100_samsgd"),
    ("arm_P_tiny", "tiny_samsgd"),
    ("arm_P_tiny", "tiny_sammuon"),
    ("arm_P_c10", "c10_sammuon"),
    ("arm_P_svhn", "svhn_sammuon"),
    # arm R: the published allocation rival, same pipeline and same HPs
    ("arm_P_handoff", "latesam_c100"),
    ("latesam_c100", "c100_sammuon"),
    ("latesam_c100", "c100_samsgd"),
    ("latesam_c100", "c100_strongsgd"),
    ("arm_P_tiny", "latesam_tiny"),
    ("latesam_tiny", "tiny_samsgd"),
    # vs the published full-time SAM recipe on every dataset
    ("arm_P_c10", "c10_samsgd"),
    ("arm_P_svhn", "svhn_samsgd"),
    ("arm_P_handoff", "c100_strongsgd"),
    # arm S: the 2x2's fourth cell. vs arm P the ONLY change is the refiner;
    # vs latesam the only change is the allocation shape.
    ("arm_P_handoff", "arm_S_sgdtail"),
    ("arm_S_sgdtail", "latesam_c100"),
    ("arm_S_sgdtail", "c100_samsgd"),
    ("arm_S_sgdtail", "c100_sammuon"),
]
for a, b in pairs:
    if a not in all_stats or b not in all_stats:
        continue
    fa, fb = all_stats[a][0], all_stats[b][0]
    t, df, p = welch(fa, fb)
    print(f"{a:>20} {st.mean(fa):.4f} vs {b:<16} {st.mean(fb):.4f} : "
          f"diff {100*(st.mean(fa)-st.mean(fb)):+.2f}pp  t={t:+.2f} df={df:.1f} p={p:.3f}")

print()
print("#" * 70)
print("# TIME-TO-TARGET (median first-hit wall-clock, phase6 arm vs baselines)")
print("#" * 70)
for arm, (_sub, ds) in ARMS.items():
    if arm not in all_stats:
        continue
    targets = TARGETS[ds]
    print(f"\n-- {arm} ({ds}) --")
    header = f"{'target':>8} | {arm[:18]:>18} | " + " | ".join(
        f"{ds}_{b}"[:14].rjust(14) for b in BENCH_ARMS)
    print(header)
    for t in targets:
        cells = []
        h = all_stats[arm][2][t]
        cells.append(f"{st.median(h):.0f}s({len(h)}/5)" if h else "--")
        for b in BENCH_ARMS:
            key = f"{ds}_{b}"
            if key in all_stats and all_stats[key][2].get(t):
                hb = all_stats[key][2][t]
                cells.append(f"{st.median(hb):.0f}s({len(hb)}/5)")
            else:
                cells.append("--")
        print(f"{t:>8} | {cells[0]:>18} | " + " | ".join(c.rjust(14) for c in cells[1:]))
