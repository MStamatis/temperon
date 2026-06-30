"""Training harness: controller-driven loop with bf16 autocast and probe hooks.

Precision policy (research-critical):
- training forward/backward runs in bf16 autocast on GPU, TF32 enabled;
- ALL curvature probes run outside this loop's autocast, in float32 with
  TF32 disabled, via probes.precision.probe_precision() (see ProbeScheduler).
- with --compile only the training forward is compiled; probes always call
  the original eager module (same parameters, no recompilation pressure).
"""

from __future__ import annotations

import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F

from eos_switch.controllers import build_controller
from eos_switch.data import GPUCifar
from eos_switch.probes.hvp import preserve_bn_stats
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


@torch.no_grad()
def final_metrics(model, data: GPUCifar, autocast_enabled: bool, device: torch.device) -> dict:
    """Full classification metric suite on the eval set (accuracy, macro/weighted
    precision/recall/F1, macro one-vs-rest ROC-AUC). sklearn is optional -- without
    it only accuracy is reported."""
    model.eval()
    logits_all, y_all = [], []
    for x, y in data.eval_batches():
        if device.type == "cuda":
            x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            logits = model(x)
        logits_all.append(logits.float().cpu())
        y_all.append(y.cpu())
    model.train()
    logits = torch.cat(logits_all)
    y_true = torch.cat(y_all).numpy()
    probs = torch.softmax(logits, dim=1).numpy()
    pred = logits.argmax(1).numpy()
    out = {
        "accuracy": float((pred == y_true).mean()),
        "n_eval": int(len(y_true)),
        "n_classes": int(logits.shape[1]),
    }
    try:
        from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

        for avg in ("macro", "weighted"):
            p, r, f, _ = precision_recall_fscore_support(y_true, pred, average=avg, zero_division=0)
            out[f"precision_{avg}"], out[f"recall_{avg}"], out[f"f1_{avg}"] = float(p), float(r), float(f)
        try:
            out["roc_auc_macro_ovr"] = float(
                roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
            )
        except Exception:
            out["roc_auc_macro_ovr"] = None  # e.g. a class absent from the eval set
    except ImportError:
        out["sklearn"] = "missing (pip install scikit-learn for full metrics)"
    return out


def _real_opt(controller):
    """The underlying torch optimizer to checkpoint (unwrap the SAM shell)."""
    o = controller.active_optimizer
    return getattr(o, "base_optimizer", o)


def _save_checkpoint(path, *, next_epoch, global_step, best_val_acc, model, opt,
                     controller, milestones, epoch_rows, gen, is_cuda) -> None:
    """Atomically write the SINGLE resume checkpoint (tmp -> os.replace), so a
    crash mid-write cannot corrupt it. Overwrites the previous one (no growth)."""
    ckpt = {
        "next_epoch": next_epoch, "global_step": global_step, "best_val_acc": best_val_acc,
        "model": model.state_dict(), "optimizer": opt.state_dict(),
        "controller": controller.state_dict(), "milestones": milestones.state_dict(),
        "epoch_rows": epoch_rows,
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if is_cuda else None,
        "rng_gen": gen.get_state(),
    }
    tmp = Path(str(path) + ".tmp")
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def _load_checkpoint(path, *, model, opt, controller, milestones, logger, gen, is_cuda):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["optimizer"])
    controller.load_state_dict(ckpt["controller"])
    milestones.load_state_dict(ckpt["milestones"])
    logger._epoch_rows = list(ckpt["epoch_rows"])
    torch.set_rng_state(ckpt["rng_torch"])
    if is_cuda and ckpt.get("rng_cuda") is not None:
        torch.cuda.set_rng_state_all(ckpt["rng_cuda"])
    gen.set_state(ckpt["rng_gen"])
    return ckpt["next_epoch"], ckpt["global_step"], ckpt["best_val_acc"]


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
    data = GPUCifar(
        ds_cfg["name"],
        device=device,
        smoke_subset=ds_cfg.get("smoke_subset"),
        val_split=ds_cfg.get("validation_split"),
        val_seed=int(ds_cfg.get("val_seed", 42)),
        color_jitter=float(cfg.get("augment", {}).get("color_jitter", 0.0)),
    )
    model = build_model(
        cfg["model"],
        data.num_classes,
        initial_channels=cfg.get("initial_channels"),
        bn_momentum=float(cfg.get("bn_momentum", 0.1)),
        stem_stride=int(cfg.get("stem_stride", 1)),
    ).to(device)
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
    flip = bool(aug_cfg.get("flip", True))  # off for label-changing flips (SVHN)
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

    # --- checkpoint / resume (--continue) ---
    start_epoch = 0
    ckpt_path = Path(out_dir) / "checkpoint.pt"
    if bool(cfg.get("continue", False)):
        if (Path(out_dir) / "summary.csv").exists():
            import pandas as pd  # this run already finished -> skip it

            print(f"[skip] {out_dir} already complete", flush=True)
            return pd.read_csv(Path(out_dir) / "summary.csv").iloc[0].to_dict()
        if ckpt_path.exists():
            start_epoch, global_step, best_val_acc = _load_checkpoint(
                ckpt_path, model=model, opt=_real_opt(controller), controller=controller,
                milestones=milestones, logger=logger, gen=gen, is_cuda=is_cuda)
            print(f"[resume] {out_dir} from epoch {start_epoch} (step {global_step})", flush=True)

    for epoch in range(start_epoch, epochs):
        ep_start = time.perf_counter()
        ep_loss_sum = 0.0
        ep_steps = 0
        for x, y in data.train_batches(batch_size, gen, augment=augment, cutout=cutout, flip=flip):
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
            sam_on = getattr(controller, "sam", False)
            if sam_on:
                # Periodic SAM (speed hack): only do the full two-pass every
                # sam_period steps; other steps take a plain single-pass base
                # update with the clean gradient.
                period = getattr(controller, "sam_period", 1)
                sam_on = (period <= 1) or (global_step % period == 0)
            if not sam_on and getattr(controller, "sam", False):
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.base_optimizer.step()
            elif sam_on:
                # SAM (arm L): perturb to the local worst case, recompute the
                # gradient there, then step from the original weights with that
                # perturbed-point gradient. grad_clip applies to the UPDATE
                # gradient (2nd pass); loss_val below reports the clean loss.
                opt.first_step()
                # When freezing BN, the perturbed pass must not leave a trace in
                # the running stats. preserve_bn_stats restores them on __exit__,
                # so the backward must run INSIDE the context (restore after it)
                # to avoid an inplace-version conflict on the BN buffers.
                bn_ctx = (
                    preserve_bn_stats(model)
                    if getattr(controller, "sam_freeze_bn", False)
                    else nullcontext()
                )
                with bn_ctx:
                    with torch.autocast(
                        device.type, dtype=torch.bfloat16, enabled=autocast_enabled
                    ):
                        loss2 = loss_fn(train_model(x), y)
                    loss2.backward()
                # Record the free same-batch sharpness from the UNCLIPPED g'
                # (clipping below would bias it negative). arm N reads it to time
                # catapults; harmless (a scalar) for other SAM arms.
                if hasattr(opt, "record_sharpness"):
                    opt.record_sharpness()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.second_step()
            else:
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
        _save_checkpoint(
            ckpt_path, next_epoch=epoch + 1, global_step=global_step,
            best_val_acc=best_val_acc, model=model, opt=_real_opt(controller),
            controller=controller, milestones=milestones,
            epoch_rows=logger._epoch_rows, gen=gen, is_cuda=is_cuda,
        )

    total_s = time.perf_counter() - t0
    metrics = final_metrics(model, data, autocast_enabled, device)
    summary = {
        "run_name": cfg.get("run_name", "run"),
        "seed": seed,
        "device": str(device),
        "model": cfg["model"],
        "dataset": ds_cfg["name"],
        "eval_set": data.eval_name,
        "epochs": epochs,
        "final_val_acc": round(val_acc, 4),
        "best_val_acc": round(best_val_acc, 4),
        # full metric suite on the eval set (final model)
        "test_accuracy": round(metrics["accuracy"], 4),
        "f1_macro": round(metrics.get("f1_macro", float("nan")), 4),
        "precision_macro": round(metrics.get("precision_macro", float("nan")), 4),
        "recall_macro": round(metrics.get("recall_macro", float("nan")), 4),
        "f1_weighted": round(metrics.get("f1_weighted", float("nan")), 4),
        "roc_auc_macro_ovr": (round(metrics["roc_auc_macro_ovr"], 4)
                              if metrics.get("roc_auc_macro_ovr") is not None else None),
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

    for attr, fname in (
        ("rewards", "rewards.jsonl"),
        ("checks", "checks.jsonl"),
        ("selections", "selections.jsonl"),
    ):
        records = getattr(controller, attr, None)
        if records:
            with open(Path(out_dir) / fname, "w", encoding="utf-8") as fh:
                for r in records:
                    fh.write(json.dumps(r) + "\n")
    with open(Path(out_dir) / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    if ckpt_path.exists():
        ckpt_path.unlink()  # run finished -> drop the resume checkpoint (saves space)
    return summary


def _probe_overhead(probe_sched, controller, total_s: float) -> float:
    if probe_sched is not None:
        return probe_sched.overhead_frac(total_s)
    if hasattr(controller, "overhead_frac"):
        return controller.overhead_frac(total_s)
    return 0.0
