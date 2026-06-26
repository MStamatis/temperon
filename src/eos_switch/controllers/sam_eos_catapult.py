"""Free-HVP Edge-of-Stability catapult (arm N -- the novelty).

Builds on the project's best, SAM+Muon-catapult. SAM computes two gradients on
the SAME batch, so g'-g is an exact (quadratic) noise-free Hessian-vector product
we already pay for. SAM.record_sharpness() stores, at zero extra cost,

    last_sharpness = g_hat^T H g_hat   (the batch_sharpness / EoS quantity)

and we use it to TIME the warm restarts (catapults) instead of a fixed period.

ROBUST RELATIVE TRIGGER (after the absolute-threshold version mis-fired). The
first version used edge_ratio = sharpness*lr/(2+2*beta) < restart_ratio, but the
(2+2*beta) edge is the SGD reference and is wrong for a Muon base: Muon's
sharpness is small (~O(1)), so edge_ratio stayed far below restart_ratio even at
peak lr -> a restart fired every check -> cycle_start kept resetting -> lr pinned
at peak, no annealing. Here the trigger is scale-free and lr-gated:

  restart when  sharp_ema < flat_frac * sharp_at_cycle_start   (basin flattened,
                RELATIVE to this cycle -- no absolute threshold)
           AND  lr <= lr_gate_frac * peak_lr                   (cycle has annealed
                -> structurally impossible to pin lr at peak)
           AND  dwell satisfied,
       OR  the cycle reached its max length cycle_epochs        (cap: falls back to
                fixed-cyclic behaviour, so it can't do worse than fixed).

sharp_ema smooths bf16 finite-difference noise. The optimizer is a SAM-wrapped
Muon driven two-pass by the loop (sam = True); the loop calls record_sharpness()
before grad clipping.
"""

from __future__ import annotations

from eos_switch.controllers.base import SwitchEvent
from eos_switch.controllers.eos_restart import EosRestartController
from eos_switch.optimizers.pool import build_optimizer
from eos_switch.optimizers.sam import SAM


class SamEosCatapultController(EosRestartController):
    sam = True  # loop signal: run the SAM two-pass (which fills last_sharpness)

    def _build(self) -> None:
        self.base_name = str(self.cfg.get("base_optimizer", "muon"))
        self.momentum = float(self.cfg.get("momentum", 0.9))
        self.peak_lr = float(self.cfg.get("lr", 0.01))
        self.min_lr = float(self.cfg.get("min_lr", 0.0))
        self.rho = float(self.cfg.get("rho", 0.05))
        self.sam_freeze_bn = bool(self.cfg.get("sam_freeze_bn", False))

        base = build_optimizer(
            self.base_name, self.model.parameters(), lr=self.peak_lr,
            momentum=self.momentum, weight_decay=float(self.cfg.get("weight_decay", 0.2)),
            nesterov=bool(self.cfg.get("nesterov", True)),
        )
        self._opt = SAM(base, rho=self.rho)

        # Schedule + robust-trigger knobs.
        self.warmup_steps = int(self.cfg.get("warmup_steps", 200))
        self.check_every = int(self.cfg.get("check_every", 50))
        self.cycle_epochs = float(self.cfg.get("cycle_epochs", 20.0))   # cosine length + cap
        self.min_dwell_checks = int(self.cfg.get("min_dwell_checks", 2))
        self.switch_until_frac = float(self.cfg.get("switch_until_frac", 0.8))
        self.flat_frac = float(self.cfg.get("flat_frac", 0.5))          # restart when sharp < this*baseline
        self.lr_gate_frac = float(self.cfg.get("lr_gate_frac", 0.3))    # only after lr <= this*peak
        self.sharp_ema_beta = float(self.cfg.get("sharp_ema_beta", 0.6))

        self._cycle_start = self.warmup_steps
        self._cycle_start_sharp: float | None = None
        self._sharp_ema: float | None = None
        self._checks_since_restart = 0
        self._final_started = False
        self._lr = self.peak_lr
        self._probe_seconds = 0.0   # curvature is free -> no probe overhead
        self.n_probes = 0
        self.checks: list[dict] = []

    def _probe_edge_ratio(self):
        # Free same-batch sharpness from the last SAM two-pass; edge_ratio kept for
        # logging only (the restart decision is the relative trigger below).
        s = getattr(self._opt, "last_sharpness", None)
        s = 0.0 if s is None else float(s)
        threshold = (2.0 + 2.0 * self.momentum) / max(self._lr, 1e-9)
        return s / threshold, s

    def begin_step(self, step: int):
        spe = max(self.steps_per_epoch, 1)
        total = self.total_epochs * spe
        cycle_steps = max(self.cycle_epochs * spe, 1.0)
        window_close = self.switch_until_frac * total

        if step < self.warmup_steps:
            self._set_lr(self.peak_lr * (step + 1) / self.warmup_steps)
            return self._opt

        # Final settle: one long cosine to min over the last (1 - frac) of training.
        if step >= window_close:
            if not self._final_started:
                self._final_started = True
                self._cycle_start = step
            self._set_lr(self._cosine(step - self._cycle_start, total - window_close))
            return self._opt

        if step % self.check_every == 0:
            s = getattr(self._opt, "last_sharpness", None)
            s = 0.0 if s is None else float(s)
            self._sharp_ema = s if self._sharp_ema is None else (
                self.sharp_ema_beta * self._sharp_ema + (1 - self.sharp_ema_beta) * s
            )
            if self._cycle_start_sharp is None:           # baseline at the cycle's start
                self._cycle_start_sharp = self._sharp_ema

            lr_ratio = self._lr / max(self.peak_lr, 1e-12)
            flattened = (
                self._cycle_start_sharp is not None
                and self._cycle_start_sharp > 0
                and self._sharp_ema < self.flat_frac * self._cycle_start_sharp
            )
            annealed = lr_ratio <= self.lr_gate_frac
            capped = (step - self._cycle_start) >= cycle_steps
            adaptive = flattened and annealed and self._checks_since_restart >= self.min_dwell_checks
            restart = bool(adaptive or capped)

            if restart:
                self._record_switch(SwitchEvent(
                    step=step, epoch=self.epoch_of(step), from_name=self.base_name,
                    to_name=self.base_name, from_lr=self._lr, to_lr=self.peak_lr,
                    reason=("flatten" if adaptive else "cap")
                    + f"(s={self._sharp_ema:.3f}/{self._cycle_start_sharp:.3f},lr_r={lr_ratio:.2f})",
                ))
                self._cycle_start = step
                self._cycle_start_sharp = None            # re-baseline next check (back at peak lr)
                self._checks_since_restart = 0
            else:
                self._checks_since_restart += 1

            edge_ratio, _ = self._probe_edge_ratio()
            self.checks.append({
                "step": step, "epoch": round(self.epoch_of(step), 3),
                "batch_sharpness": round(s, 4),
                "sharp_ema": round(self._sharp_ema, 4),
                "cycle_base": (round(self._cycle_start_sharp, 4)
                               if self._cycle_start_sharp is not None else None),
                "lr": round(self._lr, 6), "lr_ratio": round(lr_ratio, 3),
                "edge_ratio": round(edge_ratio, 5),
                "flattened": flattened, "annealed": annealed,
                "capped": capped, "restart": restart,
            })

        self._set_lr(self._cosine(step - self._cycle_start, cycle_steps))
        return self._opt

    @property
    def active_name(self) -> str:
        return f"sam_eos:{self.base_name}"
