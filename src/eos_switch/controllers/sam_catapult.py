"""SAM-coupled catapult controller (arm L).

Stage 1: Sharpness-Aware Minimization on top of the proven cyclic-catapult
schedule (arm J). The cosine warm-restart LR scheduling of
CyclicCatapultController is reused verbatim -- only the per-cycle base optimizer
is wrapped in a SAM two-pass shell. The training loop detects `sam == True` and
runs the second (perturbed) forward/backward pass; everything else (warm
restarts, optional pool-per-restart bandit, state transfer) is inherited.

Stage 2 (NOVEL, `eos_rho: true`): EoS-coupled radius -- calibrate the SAM radius
rho from the Edge of Stability instead of holding it fixed. Every `check_every`
steps we measure batch_sharpness (1 cheap HVP, fp32 inside probe_precision) and
form the momentum-SGD edge ratio

    edge_ratio = batch_sharpness * lr / (2 + 2*beta)      (edge at ratio = 1)

then set

    rho = base_rho * clamp(target_ratio / edge_ratio, rho_min/base_rho,
                                                       rho_max/base_rho)

-> LARGER rho far inside the stability region (small edge_ratio: flat/converged
basin, the worst-case ascent is gentle so we can probe wider) and SMALLER rho
near/over the edge (large edge_ratio: already sharp/unstable, do not over-
perturb). The radius `target_ratio` at which rho == base_rho is auto-calibrated
to the first valid (positive) edge_ratio, so rho starts at base_rho and moves
RELATIVE to this run's own baseline -- robust without knowing absolute scales.

`edge_lr_basis` selects which lr enters edge_ratio:
  - "peak"    : the fixed cycle peak lr -> edge_ratio tracks curvature only,
                decoupled from the deterministic cosine sweep we already control;
  - "current" : the live within-cycle lr (the true edge condition, as in
                eos_restart) -> rho also rides the cosine (big near a restart's
                tail where lr -> 0, small just after a catapult).

The training loop attaches the probe context (loss_fn, data, generator) and
drives the two SAM passes; we only retune `self._opt.rho` between steps.
"""

from __future__ import annotations

import time

import torch

from eos_switch.controllers.cyclic import CyclicCatapultController
from eos_switch.optimizers.sam import SAM


class SamCatapultController(CyclicCatapultController):
    sam = True  # loop signal: run the SAM two-pass for this controller

    def _build(self) -> None:
        # rho must exist before super()._build(), which calls self._make().
        self.rho = float(self.cfg.get("rho", 0.05))
        # BN running-stats policy for the perturbed (2nd) forward pass:
        #   False -> standard SAM (BN updates on both passes; official default)
        #   True  -> freeze BN stats on the perturbed pass (no pollution)
        self.sam_freeze_bn = bool(self.cfg.get("sam_freeze_bn", False))
        super()._build()

        # --- stage 2: EoS-coupled rho ----------------------------------------
        self.eos_rho = bool(self.cfg.get("eos_rho", False))
        self.base_rho = self.rho
        self.rho_min = float(self.cfg.get("rho_min", 0.01))
        self.rho_max = float(self.cfg.get("rho_max", 0.20))
        self.check_every = int(self.cfg.get("check_every", 50))
        self.micro_batch = int(self.cfg.get("micro_batch", 128))
        self.sharp_ema_beta = float(self.cfg.get("sharp_ema_beta", 0.6))
        self.edge_lr_basis = str(self.cfg.get("edge_lr_basis", "peak")).lower()
        # momentum of the (single) base optimizer, for the edge threshold.
        self.momentum = float(self.specs[self._name].get("momentum", 0.9))
        # target_ratio: fixed if a number is given, else auto-calibrated below.
        tr = self.cfg.get("target_ratio", None)
        self.target_ratio = None if tr in (None, "auto") else float(tr)

        self._cur_rho = self.base_rho
        self._sharp_ema: float | None = None
        self._probe_seconds = 0.0
        self.n_probes = 0
        self.checks: list[dict] = []
        self._loss_fn = self._data = self._gen = None

    def _make(self, name: str):
        # Wrap the cyclic schedule's base optimizer in a SAM two-pass shell.
        return SAM(super()._make(name), rho=self.rho)

    # --- stage 2 probe wiring (mirrors edge_lr / eos_restart) ----------------
    def attach_probe_context(self, loss_fn, data, generator) -> None:
        self._loss_fn, self._data, self._gen = loss_fn, data, generator

    def _probe_sharpness(self) -> float:
        from eos_switch.probes.sharpness import batch_sharpness

        batch = self._data.sample_probe_batch(self.micro_batch, self._gen)
        t0 = time.perf_counter()
        s = batch_sharpness(self.model, self._loss_fn, batch)["batch_sharpness"]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._probe_seconds += time.perf_counter() - t0
        self.n_probes += 1
        return s

    def _update_rho(self, step: int) -> None:
        sharp = self._probe_sharpness()
        self._sharp_ema = sharp if self._sharp_ema is None else (
            self.sharp_ema_beta * self._sharp_ema + (1 - self.sharp_ema_beta) * sharp
        )
        if self.edge_lr_basis == "current":
            lr_basis = max(self.active_lr, 1e-9)
        else:  # "peak"
            lr_basis = float(self.specs[self._name]["lr"])
        threshold = (2.0 + 2.0 * self.momentum) / max(lr_basis, 1e-9)
        edge_ratio = self._sharp_ema / threshold
        # Auto-calibrate the anchor on the first positive edge ratio: rho == base.
        if self.target_ratio is None and edge_ratio > 0:
            self.target_ratio = edge_ratio
        if self.target_ratio is not None and edge_ratio > 0:
            rho = self.base_rho * (self.target_ratio / edge_ratio)
        else:
            # negative curvature (non-convex noise, not a converged basin):
            # minimal perturbation.
            rho = self.rho_min
        self._cur_rho = min(max(rho, self.rho_min), self.rho_max)
        self.checks.append({
            "step": step, "epoch": round(self.epoch_of(step), 3),
            "batch_sharpness": round(sharp, 5),
            "sharp_ema": round(self._sharp_ema, 5),
            "lr_basis": round(lr_basis, 6),
            "edge_ratio": round(edge_ratio, 5),
            "target_ratio": (None if self.target_ratio is None
                             else round(self.target_ratio, 5)),
            "rho": round(self._cur_rho, 5),
        })

    def begin_step(self, step: int) -> torch.optim.Optimizer:
        opt = super().begin_step(step)  # cyclic schedule sets the lr
        if self.eos_rho:
            if step >= self.warmup_steps and self._data is not None \
                    and step % self.check_every == 0:
                self._update_rho(step)
            # Apply to whatever optimizer is active (robust to pool rebuilds).
            opt.rho = self._cur_rho
        return opt

    def overhead_frac(self, total_train_seconds: float) -> float:
        return self._probe_seconds / max(total_train_seconds, 1e-9)

    @property
    def active_name(self) -> str:
        return f"sam:{super().active_name}"
