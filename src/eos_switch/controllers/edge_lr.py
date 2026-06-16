"""Edge-of-Stability adaptive learning rate (arm H, super-convergence).

Instead of switching optimizers, keep a single SGD-momentum optimizer and
drive its learning rate to the Edge of Stability: the momentum-SGD stability
condition is lambda_max < (2 + 2*beta)/lr, so the largest stable lr is
(2 + 2*beta)/lambda_max. We set

    lr = safety * (2 + 2*beta) / lambda_max_ema      (safety ~ 0.9, just below)

with lambda_max measured by power iteration every `check_every` steps (fp32).
This directly exploits the EoS mechanism (train at the edge -> catapult /
super-convergence) rather than changing optimizer identity. A short linear LR
warmup precedes edge control so the early curvature estimate can settle.

The training loop attaches the probe context (loss_fn, data, generator).
"""

from __future__ import annotations

import math
import time

import torch

from eos_switch.controllers.base import Controller


class EdgeLRController(Controller):
    def _build(self) -> None:
        from eos_switch.optimizers.pool import build_optimizer

        self.momentum = float(self.cfg.get("momentum", 0.9))
        self._opt = build_optimizer(
            "sgd_momentum",
            self.model.parameters(),
            lr=float(self.cfg.get("lr_init", 0.1)),
            momentum=self.momentum,
            weight_decay=float(self.cfg.get("weight_decay", 5e-4)),
            nesterov=bool(self.cfg.get("nesterov", True)),
        )
        self.lr_init = float(self.cfg.get("lr_init", 0.1))
        self.lr_min = float(self.cfg.get("lr_min", 1e-3))
        self.lr_max = float(self.cfg.get("lr_max", 0.5))
        self.safety = float(self.cfg.get("safety", 0.9))
        self.check_every = int(self.cfg.get("check_every", 50))
        self.power_iters = int(self.cfg.get("power_iters", 20))
        self.micro_batch = int(self.cfg.get("micro_batch", 128))
        self.lambda_beta = float(self.cfg.get("lambda_ema_beta", 0.6))
        self.warmup_steps = int(self.cfg.get("warmup_steps", 200))
        # anneal: none -> hold lr at the edge; cosine -> one-cycle-at-the-edge
        # (edge sets the PEAK, a cosine envelope decays it to 0). Pure
        # edge-holding does not converge; super-convergence needs the decay.
        self.anneal = str(self.cfg.get("anneal", "none")).lower()

        self._lr = self.lr_init
        self._base_edge_lr = self.lr_init
        self._lambda_ema: float | None = None
        self._probe_seconds = 0.0
        self.n_probes = 0
        self.checks: list[dict] = []
        self._loss_fn = self._data = self._gen = None

    def attach_probe_context(self, loss_fn, data, generator) -> None:
        self._loss_fn = loss_fn
        self._data = data
        self._gen = generator

    def _set_lr(self, lr: float) -> None:
        self._lr = lr
        for g in self._opt.param_groups:
            g["lr"] = lr

    def _probe_lambda_max(self) -> float:
        from eos_switch.probes.lambda_max import lambda_max

        batch = self._data.sample_probe_batch(self.micro_batch, self._gen)
        t0 = time.perf_counter()
        out = lambda_max(self.model, self._loss_fn, batch, iters=self.power_iters, generator=self._gen)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._probe_seconds += time.perf_counter() - t0
        self.n_probes += 1
        return max(out["lambda_max"], 1e-6)

    def _envelope(self, step: int) -> float:
        if self.anneal != "cosine":
            return 1.0
        total = self.total_epochs * self.steps_per_epoch
        p = (step - self.warmup_steps) / max(total - self.warmup_steps, 1)
        p = min(max(p, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * p))  # 1 -> 0 over the post-warmup run

    def begin_step(self, step: int) -> torch.optim.Optimizer:
        # Linear LR warmup before edge control engages (the ramp-up of one-cycle).
        if step < self.warmup_steps:
            self._set_lr(self.lr_init * (step + 1) / self.warmup_steps)
            return self._opt

        lam = None
        if self._data is not None and step % self.check_every == 0:
            lam = self._probe_lambda_max()
            self._lambda_ema = lam if self._lambda_ema is None else (
                self.lambda_beta * self._lambda_ema + (1 - self.lambda_beta) * lam
            )
            edge_lr = self.safety * (2.0 + 2.0 * self.momentum) / self._lambda_ema
            self._base_edge_lr = min(max(edge_lr, self.lr_min), self.lr_max)

        # The edge sets the peak; a cosine envelope anneals it toward 0.
        self._set_lr(self._base_edge_lr * self._envelope(step))

        if lam is not None:
            self.checks.append({
                "step": step,
                "epoch": round(self.epoch_of(step), 3),
                "lambda_max": round(lam, 4),
                "lambda_ema": round(self._lambda_ema, 4),
                "base_edge_lr": round(self._base_edge_lr, 6),
                "lr": round(self._lr, 6),
            })
        return self._opt

    def overhead_frac(self, total_train_seconds: float) -> float:
        return self._probe_seconds / max(total_train_seconds, 1e-9)

    @property
    def active_name(self) -> str:
        return "sgd_edge_lr"

    @property
    def active_optimizer(self) -> torch.optim.Optimizer:
        return self._opt
