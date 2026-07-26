"""Phase 8 grid launcher: tasks x arms x seeds, skipping finished runs.

Sequential on one GPU (parallel runs would corrupt the wall-clock numbers the
allocation claim rests on). Re-running the same command resumes: any
combination whose result.json already exists is skipped.

  python experiments/glue/launch_glue.py                       # full grid
  python experiments/glue/launch_glue.py --tasks rte mrpc      # subset
  python experiments/glue/launch_glue.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [42, 1181241943, 958682846, 271828, 314159]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["rte", "mrpc", "stsb", "cola"])
    ap.add_argument("--arms", nargs="+", default=["off", "tail", "full"])
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--output", default="results/phase8")
    ap.add_argument("--config", default="configs/glue_base.yaml")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    jobs, skipped = [], 0
    for task in args.tasks:
        for seed in args.seeds:
            for arm in args.arms:  # off/tail share a prefix -> keep them adjacent
                done = os.path.join(args.output, task, arm, f"seed{seed}",
                                    "result.json")
                if os.path.exists(done):
                    skipped += 1
                    continue
                jobs.append((task, arm, seed))

    print(f"{len(jobs)} runs to do, {skipped} already finished")
    if args.dry_run:
        for t, a, s in jobs:
            print(f"  {t:>5} {a:>4} seed{s}")
        return

    t0 = time.perf_counter()
    for i, (task, arm, seed) in enumerate(jobs, 1):
        cmd = [sys.executable, os.path.join(HERE, "train_glue.py"),
               "--task", task, "--arm", arm, "--seed", str(seed),
               "--config", args.config, "--output", args.output]
        el = time.perf_counter() - t0
        print(f"\n=== [{i}/{len(jobs)}] {task} {arm} seed{seed} "
              f"(elapsed {el/60:.0f} min) ===", flush=True)
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"FAILED: {task} {arm} seed{seed} (exit {r.returncode})")
            sys.exit(r.returncode)
    print(f"\nall done in {(time.perf_counter()-t0)/60:.0f} min")


if __name__ == "__main__":
    main()
