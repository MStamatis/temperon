"""Phase 8 diagnostic: does full-time SAM stop hurting at a smaller rho?

The 5-seed Phase-8 run found the tail arm beating full-time SAM on all four
tasks, but also found full-time SAM *below* the no-SAM baseline everywhere --
which contradicts the published fine-tuning gains that motivated the
experiment. The leading suspect is rho: 0.05 may simply be too large for
fine-tuning, in which case the tail wins by harming less rather than by
helping.

This script compares the `full` arm across rho settings against the *same*
no-SAM baseline (the `off` arm does not use rho, so it is reused unchanged).

    python experiments/glue/analyze_rho.py results/phase8 \
        0.05:results/phase8 0.02:results/phase8_rho0.02 0.01:results/phase8_rho0.01

The first argument is the root holding the `off` baseline; the rest are
`rho:root` pairs. A rho at which `full` climbs above `off` is the one to re-run
the tail arm at.
"""

from __future__ import annotations

import glob
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_glue import TASKS, welch  # noqa: E402


def scores(root: str, task: str, arm: str) -> list[float]:
    out = []
    for p in sorted(glob.glob(os.path.join(
            glob.escape(root), task, arm, "*", "result.json"))):
        out.append(json.load(open(p))["best_score"])
    return out


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    base_root = sys.argv[1]
    pairs = []
    for spec in sys.argv[2:]:
        rho, root = spec.split(":", 1)
        pairs.append((rho, root))

    print("full-time SAM vs the no-SAM baseline, by rho (best dev, 5 seeds)")
    print(f"baseline `off` read from: {base_root}\n")
    for task in TASKS:
        off = scores(base_root, task, "off")
        if not off:
            continue
        print(f"=== {task.upper()} ===   off {st.mean(off):.4f} +/- "
              f"{st.stdev(off):.4f}  (n={len(off)})")
        for rho, root in pairs:
            full = scores(root, task, "full")
            if not full:
                print(f"   rho {rho:>5}: (no runs)")
                continue
            t, df, p = welch(full, off)
            delta = 100 * (st.mean(full) - st.mean(off))
            verdict = "HELPS" if delta > 0 and p < 0.05 else (
                "hurts" if delta < 0 and p < 0.05 else "n.s.")
            print(f"   rho {rho:>5}: {st.mean(full):.4f} +/- {st.stdev(full):.4f}"
                  f"  {delta:+.2f}pp vs off  p={p:.3f}  {verdict}  (n={len(full)})")
        print()


if __name__ == "__main__":
    main()
