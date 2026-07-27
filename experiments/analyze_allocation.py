"""Where does the allocation saving go? Explorer credit vs refiner premium.

Skipping SAM for the first 43 epochs buys a fixed credit:

    credit = 43 x (cost of a SAM epoch - cost of a cheap epoch)

What happens to that credit depends on what the tail then spends it on. Our
tiny-ImageNet arm keeps an SGD tail, so the credit survives as wall-clock. The
other three arms hand the tail to Muon, which costs more per epoch than the SAM
baseline it is being compared against, and the premium eats the credit -- the
run comes out even on time and ahead on accuracy instead.

That is one law with two ways of cashing it out, not an inconsistency, and this
script shows the arithmetic per dataset so the claim is checkable.

    python experiments/analyze_allocation.py results
"""

from __future__ import annotations

import csv
import glob
import json
import os
import statistics as st
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "results"

# dataset -> (target, our arm, the full-time SAM+SGD baseline, switch epoch)
CASES = {
    "c100": (0.82, "phase6/arm_P_handoff", "bench/c100/c100_samsgd", 43),
    "tiny": (0.69, "phase6/arm_P_tiny", "bench/tiny/tiny_samsgd", 43),
    "c10": (0.965, "phase6/arm_P_c10", "bench/c10/c10_samsgd", 43),
    "svhn": (0.98, "phase6/arm_P_svhn", "bench/svhn/svhn_samsgd", 43),
}
ACC = {  # 5-seed final test mean, from analyze_5seed
    "c100": (0.8295, 0.8232), "tiny": (0.7003, 0.7027),
    "c10": (0.9695, 0.9668), "svhn": (0.9807, 0.9798),
}


def runs(sub):
    pat = os.path.join(glob.escape(os.path.join(ROOT, *sub.split("/"))), "seed*")
    for d in sorted(glob.glob(pat)):
        rows = []
        with open(os.path.join(d, "epochs.csv"), newline="") as f:
            for r in csv.DictReader(f):
                rows.append((int(r["epoch"]), float(r["val_acc"]),
                             r["optimizer"], float(r["epoch_time_s"])))
        yield rows


def cost_table():
    """Clean s/epoch: calibration if present, else min observed."""
    clean, calibrated = {}, False
    p = os.path.join(ROOT, "costmodel", "calibration.json")
    if os.path.exists(p):
        calibrated = True
        for ds, opts in json.load(open(p)).items():
            for opt, e in opts.items():
                clean[(ds, opt)] = e["median_s"]
    measured = set(clean)  # calibrated keys win; the rest fall back to min
    for ds, (_t, ours, base, _s) in CASES.items():
        for sub in (ours, base):
            for rows in runs(sub):
                for ep, _a, opt, s in rows:
                    if ep and (ds, opt) not in measured:
                        clean[(ds, opt)] = min(clean.get((ds, opt), s), s)
    return clean, calibrated


def main():
    clean, calibrated = cost_table()
    print("cost basis:", "calibrated" if calibrated else "min-observed estimate")
    print()
    print(f"{'dataset':>7} {'tail':>10} | {'credit':>8} {'premium':>8} {'net':>8} "
          f"| {'time':>8} {'accuracy':>10}")
    print("-" * 74)

    for ds, (target, ours, base, switch) in CASES.items():
        ours_rows = list(runs(ours))
        base_rows = list(runs(base))
        if not ours_rows or not base_rows:
            continue

        def hit(rs):
            e = [next((r[0] for r in x if r[1] >= target), None) for x in rs]
            e = [v for v in e if v is not None]
            return st.median(e) if e else None

        eo, eb = hit(ours_rows), hit(base_rows)
        if eo is None or eb is None:
            print(f"{ds:>7} : target {target} not reached by both arms")
            continue

        # which optimizer owns each phase
        tail_opt = next(r[2] for r in ours_rows[0] if r[0] > switch)
        cheap_opt = next(r[2] for r in ours_rows[0] if 0 < r[0] <= switch)
        sam_opt = next(r[2] for r in base_rows[0] if r[0] > 0)

        c_cheap = clean[(ds, cheap_opt)]
        c_tail = clean[(ds, tail_opt)]
        c_sam = clean[(ds, sam_opt)]

        credit = switch * (c_sam - c_cheap)          # epochs we did not pay SAM for
        premium = (eo - switch) * (c_tail - c_sam)   # extra the tail costs per epoch
        drift = (eo - eb) * c_sam                    # different epochs to target
        ours_t = switch * c_cheap + (eo - switch) * c_tail
        base_t = eb * c_sam
        da = 100 * (ACC[ds][0] - ACC[ds][1])

        print(f"{ds:>7} {tail_opt.replace('sam:', ''):>10} | "
              f"{-credit:+8.0f} {premium:+8.0f} {premium - credit + drift:+8.0f} "
              f"| {100*(ours_t/base_t-1):+7.1f}% {da:+9.2f}pp")

    print("-" * 74)
    print("credit  = seconds saved by not running SAM before the switch")
    print("premium = seconds the tail optimizer costs above the SAM baseline")
    print("net     = credit + premium + epochs-to-target drift")
    print("time/accuracy are versus the full-time SAM+SGD baseline of that dataset")


if __name__ == "__main__":
    main()
