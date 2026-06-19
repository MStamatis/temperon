"""EoS-triggered warm restarts (arm K) -- novel super-convergence controller.

A cyclic-catapult (SGDR-style) schedule whose RESTART timing is driven by the
Edge of Stability instead of a fixed cosine period. We keep SGD-momentum and
cosine-decay the lr within a cycle, but a restart (catapult: lr -> peak) fires
when the optimizer has settled deep inside its stability region -- measured by

    edge_ratio = batch_sharpness * lr / (2 + 2*beta)

(the momentum-SGD edge is at edge_ratio = 1). When edge_ratio drops below
`restart_ratio` the model is in a flat/converged basin, so a catapult back to
the peak lr lets it escape and explore again -- the right moment to restart,
rather than an arbitrary epoch. Restarts are disabled in the last
`1 - switch_until_frac` of training so the final cosine anneals to settle
(final ~ best). batch_sharpness is 1 cheap HVP (fp32, in probe_precision).

Not aware of published edge-of-stability-triggered warm restarts -- this ties
our EoS probes directly to the catapult schedule.
"""

from __future__ import annotations

import math
import time

import torch

from eos_switch.controllers.base import Controller, SwitchEvent


class EosRestartController(Controller):
    def _build(self) -> None:
        from eos_switch.optimizers.pool import build_optimizer

        self.momentum = float(self.cfg.get("momentum", 0.9))
        self.peak_lr = float(self.cfg.get("lr", 0.1))
        self.min_lr = float(self.cfg.get("min_lr", 0.0))
        self._opt = build_optimizer(
            "sgd_momentum", self.model.parameters(), lr=self.peak_lr,
            momentum=self.momentum, weight_decay=float(self.cfg.get("weight_decay", 5e-4)),
            nesterov=bool(self.cfg.get("nesterov", True)),
        )
        self.warmup_steps = int(self.cfg.get("warmup_steps", 200))
        self.check_every = int(self.cfg.get("check_every", 50))
        self.micro_batch = int(self.cfg.get("micro_batch", 128))
        self.restart_ratio = float(self.cfg.get("restart_ratio", 0.1))
        self.cycle_epochs = float(self.cfg.get("cycle_epochs", 20.0))
        self.min_dwell_checks = int(self.cfg.get("min_dwell_checks", 2))
        self.switch_until_frac = float(self.cfg.get("switch_until_frac", 0.8))

        self._cycle_start = self.warmup_steps
        self._checks_since_restart = 0
        self._final_started = False
        self._lr = self.peak_lr
        self._probe_seconds = 0.0
        self.n_probes = 0
        self.checks: list[dict] = []
        self._loss_fn = self._data = self._gen = None

    def attach_probe_context(self, loss_fn, data, generator) -> None:
        self._loss_fn, self._data, self._gen = loss_fn, data, generator

    def _set_lr(self, lr: float) -> None:
        self._lr = lr
        for g in self._opt.param_groups:
            g["lr"] = lr

    def _cosine(self, t: float, length: float) -> float:
        frac = min(max(t / max(length, 1.0), 0.0), 1.0)
        return self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (1 + math.cos(math.pi * frac))

    def _probe_edge_ratio(self) -> float:
        from eos_switch.probes.sharpness import batch_sharpness

        batch = self._data.sample_probe_batch(self.micro_batch, self._gen)
        t0 = time.perf_counter()
        s = batch_sharpness(self.model, self._loss_fn, batch)["batch_sharpness"]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._probe_seconds += time.perf_counter() - t0
        self.n_probes += 1
        threshold = (2.0 + 2.0 * self.momentum) / max(self._lr, 1e-9)
        return s / threshold, s

    def begin_step(self, step: int) -> torch.optim.Optimizer:
        total = self.total_epochs * max(self.steps_per_epoch, 1)
        spe = max(self.steps_per_epoch, 1)
        cycle_steps = self.cycle_epochs * spe
        window_close = self.switch_until_frac * total

        if step < self.warmup_steps:
            self._set_lr(self.peak_lr * (step + 1) / self.warmup_steps)
            return self._opt

        # Final settle: one long cosine to min over the last (1-frac) of training.
        if step >= window_close:
            if not self._final_started:
                self._final_started = True
                self._cycle_start = step
            self._set_lr(self._cosine(step - self._cycle_start, total - window_close))
            return self._opt

        # In-window: EoS-triggered restart check.
        if self._data is not None and step % self.check_every == 0:
            edge_ratio, sharp = self._probe_edge_ratio()
            # Restart only when settled in a flat region: small POSITIVE edge
            # ratio (negative = non-convex noise, not a converged basin).
            restart = (
                0.0 <= edge_ratio < self.restart_ratio
                and self._checks_since_restart >= self.min_dwell_checks
            )
            if restart:
                self._cycle_start = step
                self._checks_since_restart = 0
                self._record_switch(SwitchEvent(
                    step=step, epoch=self.epoch_of(step), from_name="sgd_momentum",
                    to_name="sgd_momentum", from_lr=self._lr, to_lr=self.peak_lr,
                    reason=f"eos_restart(edge={edge_ratio:.3f})",
                ))
            else:
                self._checks_since_restart += 1
            self.checks.append({
                "step": step, "epoch": round(self.epoch_of(step), 3),
                "batch_sharpness": round(sharp, 4), "lr": round(self._lr, 5),
                "edge_ratio": round(edge_ratio, 4), "restart": restart,
            })

        self._set_lr(self._cosine(step - self._cycle_start, cycle_steps))
        return self._opt

    def overhead_frac(self, total_train_seconds: float) -> float:
        return self._probe_seconds / max(total_train_seconds, 1e-9)

    @property
    def active_name(self) -> str:
        return "sgd_eos_restart"

    @property
    def active_optimizer(self) -> torch.optim.Optimizer:
        return self._opt
