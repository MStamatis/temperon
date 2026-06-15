"""Training harness: controller-driven loop with bf16 autocast and probe hooks.

Precision policy (research-critical):
- training forward/backward runs in bf16 autocast on GPU, TF32 enabled;
- ALL curvature probes run outside this loop's autocast, in float32 with
  TF32 disabled, via probes.precision.probe_precision() (see ProbeScheduler).
- with --compile only the training forward is compiled; probes always call
  the original eager module (same parameters, no recompilation pressure).
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn.functional as F

from eos_switch.controllers import build_controller
from eos_switch.data import GPUCifar
from eos_switch.report.logging import MilestoneTracker, RunLogger
from eos_switch.train.models import build_model
from eos_switch.train.seed import make_generator, set_seed


def resolve_device(spec: str | None) -> torch.device:
    if spec in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


@torch.no_grad()
def evaluate(model, data: GPUCifar, autocast_enabled: bool, device: torch.device):
    model.eval()
    correct = 0
    loss_sum = 0.0
    n = 0
    for x, y in data.eval_batches():
        if device.type == "cuda":
            x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            logits = model(x)
        loss_sum += F.cross_entropy(logits.float(), y, reduction="sum").item()
        correct += (logits.argmax(dim=1) == y).sum().item()
        n += len(y)
    model.train()
    return correct / n, loss_sum / n


def run_training(cfg: dict, out_dir: str | Path) -> dict:
    device = resolve_device(cfg.get("device"))
    is_cuda = device.type == "cuda"
    seed = int(cfg["seed"])
    set_seed(seed, bool(cfg.get("deterministic", False)))
    if is_cuda:
        # TF32 for training compute; probes disable it inside probe_precision().
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    ds_cfg = cfg["dataset"]
    data = GPUCifar(ds_cfg["name"], device=device, smoke_subset=ds_cfg.get("smoke_subset"))
    model = build_model(cfg["model"], data.num_classes).to(device)
    if is_cuda:
        model = model.to(memory_format=torch.channels_last)

    batch_size = int(cfg["batch_size"])
    epochs = int(cfg["epochs"])
    steps_per_epoch = len(data) // batch_size

    # Controllers with internal randomness (roulette) must vary by run seed.
    cfg["controller"].setdefault("seed", seed)
    controller = build_controller(cfg["controller"])
    controller.setup(model, steps_per_epoch, epochs)

    autocast_enabled = is_cuda and cfg.get("mixed_precision", "bf16") == "bf16"
    train_model = torch.compile(model) if cfg.get("compile", False) else model

    logger = RunLogger(out_dir)
    logger.snapshot_nvidia_smi()
    milestones = MilestoneTracker()
    gen = make_generator(seed, device)
    loss_fn = F.cross_entropy
    # Controllers that probe internally (e.g. the EoS-switch controller) get
    # the data/loss context so they can measure margins during training.
    if hasattr(controller, "attach_probe_context"):
        controller.attach_probe_context(loss_fn, data, make_generator(seed + 2, device))
    aug_cfg = cfg.get("augment", {})
    augment = bool(aug_cfg.get("enabled", True))
    cutout = int(aug_cfg.get("cutout", 0))
    grad_clip = float(cfg.get("grad_clip", 0.0))

    probe_sched = None
    if cfg.get("probes", {}).get("enabled", False):
        from eos_switch.probes.scheduler import ProbeScheduler

        probe_sched = ProbeScheduler(
            cfg["probes"],
            model=model,
            loss_fn=loss_fn,
            data=data,
            controller=controller,
            logger=logger,
            generator=make_generator(seed + 1, device),
        )

    t0 = time.perf_counter()
    global_step = 0
    n_switches_seen = 0
    val_acc = float("nan")
    best_val_acc = 0.0

    for epoch in range(epochs):
        ep_start = time.perf_counter()
        ep_loss_sum = 0.0
        ep_steps = 0
        for x, y in data.train_batches(batch_size, gen, augment=augment, cutout=cutout):
            if is_cuda:
                x = x.contiguous(memory_format=torch.channels_last)
            opt = controller.begin_step(global_step)
            if len(controller.switch_events) > n_switches_seen:
                for ev in controller.switch_events[n_switches_seen:]:
                    if probe_sched is not None:
                        probe_sched.notify_switch(ev)
                n_switches_seen = len(controller.switch_events)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                logits = train_model(x)
                loss = loss_fn(logits, y)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            loss_val = loss.item()
            controller.end_step(global_step, loss_val)
            t_now = time.perf_counter() - t0
            logger.log_step(
                {
                    "step": global_step,
                    "epoch": round(epoch + ep_steps / max(steps_per_epoch, 1), 4),
                    "loss": round(loss_val, 6),
                    "lr": controller.active_lr,
                    "optimizer": controller.active_name,
                    "t": round(t_now, 3),
                }
            )
            if probe_sched is not None:
                probe_sched.maybe_probe(global_step, t_now)
            ep_loss_sum += loss_val
            ep_steps += 1
            global_step += 1

        val_acc, val_loss = evaluate(model, data, autocast_enabled, device)
        best_val_acc = max(best_val_acc, val_acc)
        wall = time.perf_counter() - t0
        milestones.update(epoch, wall, val_acc)
        controller.end_epoch(epoch, val_acc)
        logger.log_epoch(
            {
                "epoch": epoch,
                "train_loss": ep_loss_sum / max(ep_steps, 1),
                "val_acc": round(val_acc, 4),
                "val_loss": round(val_loss, 4),
                "optimizer": controller.active_name,
                "lr": controller.active_lr,
                "epoch_time_s": round(time.perf_counter() - ep_start, 2),
                "wall_clock_s": round(wall, 2),
            }
        )

    total_s = time.perf_counter() - t0
    summary = {
        "run_name": cfg.get("run_name", "run"),
        "seed": seed,
        "device": str(device),
        "model": cfg["model"],
        "dataset": ds_cfg["name"],
        "epochs": epochs,
        "final_val_acc": round(val_acc, 4),
        "best_val_acc": round(best_val_acc, 4),
        "total_time_s": round(total_s, 2),
        "n_switches": len(controller.switch_events),
        "probe_overhead_frac": round(_probe_overhead(probe_sched, controller, total_s), 4),
        **milestones.as_flat_dict(),
    }
    meta = {
        "config": cfg,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0) if is_cuda else "cpu",
        "steps_per_epoch": steps_per_epoch,
        "switch_events": [ev.as_dict() for ev in controller.switch_events],
    }
    logger.write_meta(meta)
    logger.finalize(summary)
    import json

    for attr, fname in (("rewards", "rewards.jsonl"), ("checks", "checks.jsonl")):
        records = getattr(controller, attr, None)
        if records:
            with open(Path(out_dir) / fname, "w", encoding="utf-8") as fh:
                for r in records:
                    fh.write(json.dumps(r) + "\n")
    return summary


def _probe_overhead(probe_sched, controller, total_s: float) -> float:
    if probe_sched is not None:
        return probe_sched.overhead_frac(total_s)
    if hasattr(controller, "overhead_frac"):
        return controller.overhead_frac(total_s)
    return 0.0
