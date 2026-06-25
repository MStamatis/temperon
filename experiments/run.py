"""Unified experiment entry point.

    python experiments/run.py --config configs/smoke.yaml [--smoke] [--seed N]
                              [--device cpu|cuda] [--compile] [--output results]

--smoke shrinks ANY config to a <5-minute CPU-friendly sanity run:
CIFAR-10 subset of 5k samples, 3 epochs, and phase/warmup boundaries pulled
inside the shortened horizon so switching logic still gets exercised.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from eos_switch.train.loop import run_training


def apply_smoke_overrides(cfg: dict) -> dict:
    cfg["run_name"] = cfg.get("run_name", "run") + "_smoke"
    cfg["dataset"]["name"] = "cifar10"
    cfg["dataset"]["smoke_subset"] = 5000
    cfg["epochs"] = 3
    ctrl = cfg.get("controller", {})
    # Pull switching boundaries into the 3-epoch horizon.
    if ctrl.get("type") == "sequential":
        for phase in ctrl.get("phases", [])[:-1]:
            phase["epochs"] = 1
    if ctrl.get("type") in ("optiroulette", "eos_switch") and ctrl.get("warmup", True):
        ctrl["warmup_epochs"] = 1
    if ctrl.get("type") == "perlayer":
        ctrl["warmup_epochs"] = 1
        ctrl["select_every"] = min(int(ctrl.get("select_every", 200)), 20)
    if ctrl.get("type") == "edge_lr":
        ctrl["warmup_steps"] = min(int(ctrl.get("warmup_steps", 200)), 20)
        ctrl["check_every"] = min(int(ctrl.get("check_every", 50)), 10)
    if ctrl.get("type") in ("eos_restart", "sam_eos_catapult"):
        ctrl["warmup_steps"] = min(int(ctrl.get("warmup_steps", 200)), 20)
        ctrl["check_every"] = min(int(ctrl.get("check_every", 50)), 10)
        ctrl["cycle_epochs"] = min(float(ctrl.get("cycle_epochs", 20)), 1.0)
    if ctrl.get("type") in ("sam_catapult", "sharp_muon_catapult"):
        # Pull warmup inside the 3-epoch horizon so warm restarts (and the SAM
        # two-pass / sharpness correction) get exercised in the smoke run.
        ctrl["warmup_steps"] = min(int(ctrl.get("warmup_steps", 200)), 20)
        # Shrink the EoS-rho probe cadence so stage-2 rho adaptation fires too.
        if ctrl.get("eos_rho", False):
            ctrl["check_every"] = min(int(ctrl.get("check_every", 50)), 10)
    probes = cfg.get("probes", {})
    if probes.get("enabled", False):
        probes["probe_every"] = min(int(probes.get("probe_every", 50)), 20)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None, help="cpu / cuda / auto (default: config or auto)")
    parser.add_argument("--compile", action="store_true", help="torch.compile the training forward")
    parser.add_argument("--output", default="results")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="dotted config override, e.g. --set controller.lr=3e-4 --set run_name=arm_B_lr3e-4",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    for item in args.overrides:
        key, _, raw = item.partition("=")
        node = cfg
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw)
    if args.smoke:
        cfg = apply_smoke_overrides(cfg)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.device is not None:
        cfg["device"] = args.device
    if args.compile:
        cfg["compile"] = True

    out_dir = Path(args.output) / cfg["run_name"] / f"seed{cfg['seed']}"
    print(f"run: {cfg['run_name']}  seed={cfg['seed']}  out={out_dir}", flush=True)
    summary = run_training(cfg, out_dir)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
