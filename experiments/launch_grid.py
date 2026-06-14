"""Simple job queue: run experiment grids with bounded concurrency on 1 GPU.

Examples:
    # all five arms, three seeds, sequential
    python experiments/launch_grid.py --configs "configs/arm_*.yaml"

    # arm B LR sweep, 2 jobs in parallel on the same GPU
    python experiments/launch_grid.py --configs configs/arm_B_tuned_baseline.yaml \
        --sweep controller.lr=3e-4,1e-3,3e-3 --parallel 2

    # 1-epoch ETA benchmark
    python experiments/launch_grid.py --configs "configs/arm_*.yaml" --seeds 42 \
        --extra "--set epochs=1"

Before adopting --parallel > 1 as a default, verify with a 1-epoch benchmark
that total throughput actually improves on this GPU.
"""

from __future__ import annotations

import argparse
import glob as globmod
import shlex
import subprocess
import sys
import time
from pathlib import Path


def build_jobs(args) -> list[dict]:
    configs: list[str] = []
    for pattern in args.configs:
        matches = sorted(globmod.glob(pattern))
        if not matches:
            raise SystemExit(f"no configs match {pattern!r}")
        configs.extend(matches)

    sweep_key, sweep_vals = None, [None]
    if args.sweep:
        sweep_key, _, raw = args.sweep.partition("=")
        sweep_vals = raw.split(",")

    seeds = [int(s) for s in args.seeds.split(",")]
    jobs = []
    for cfg in configs:
        for sweep_val in sweep_vals:
            for seed in seeds:
                name = Path(cfg).stem
                cmd = [
                    sys.executable,
                    "experiments/run.py",
                    "--config",
                    cfg,
                    "--seed",
                    str(seed),
                    "--output",
                    args.output,
                ]
                if sweep_val is not None:
                    leaf = sweep_key.split(".")[-1]
                    name = f"{name}_{leaf}{sweep_val}"
                    cmd += ["--set", f"{sweep_key}={sweep_val}", "--set", f"run_name={name}"]
                if args.epochs is not None:
                    cmd += ["--set", f"epochs={args.epochs}"]
                if args.smoke:
                    cmd.append("--smoke")
                if args.extra:
                    cmd += shlex.split(args.extra)
                jobs.append({"name": f"{name}_seed{seed}", "cmd": cmd})
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True, help="config paths or globs")
    parser.add_argument("--seeds", default="42,1181241943,958682846")
    parser.add_argument("--parallel", type=int, default=1, help="max concurrent runs on the GPU")
    parser.add_argument("--sweep", default=None, metavar="KEY=V1,V2,...")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=None, help="override epochs for every run")
    parser.add_argument("--extra", default="", help="extra args appended to every run.py call")
    parser.add_argument("--output", default="results")
    args = parser.parse_args()

    jobs = build_jobs(args)
    log_dir = Path(args.output) / "grid_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} jobs, parallel={args.parallel}")

    pending = list(jobs)
    running: list[dict] = []
    done: list[dict] = []
    t0 = time.perf_counter()
    while pending or running:
        while pending and len(running) < args.parallel:
            job = pending.pop(0)
            log_path = log_dir / f"{job['name']}.log"
            handle = open(log_path, "w", encoding="utf-8")
            job["log"] = handle
            job["t_start"] = time.perf_counter()
            job["proc"] = subprocess.Popen(job["cmd"], stdout=handle, stderr=subprocess.STDOUT)
            print(f"[start] {job['name']}  (log: {log_path})", flush=True)
            running.append(job)
        time.sleep(2)
        for job in list(running):
            code = job["proc"].poll()
            if code is None:
                continue
            job["log"].close()
            job["duration_s"] = time.perf_counter() - job["t_start"]
            job["exit_code"] = code
            status = "ok" if code == 0 else f"FAILED({code})"
            print(f"[done ] {job['name']}  {status}  {job['duration_s']:.1f}s", flush=True)
            running.remove(job)
            done.append(job)

    total = time.perf_counter() - t0
    failed = [j for j in done if j["exit_code"] != 0]
    print(f"\ngrid finished in {total:.1f}s — {len(done) - len(failed)} ok, {len(failed)} failed")
    for job in done:
        print(f"  {job['name']:<50} exit={job['exit_code']}  {job['duration_s']:.1f}s")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
