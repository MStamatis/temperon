"""Static controllers: single optimizer (arms A/B) and fixed sequences (arm C)."""

from __future__ import annotations

import math

import torch

from eos_switch.controllers.base import Controller, SwitchEvent
from eos_switch.optimizers.pool import build_optimizer


class FixedController(Controller):
    """One optimizer for the whole run, with optional cosine LR schedule.

    cfg keys: optimizer, lr, weight_decay, momentum, schedule (none|cosine),
    min_lr (for cosine, default 0).
    """

    def _build(self) -> None:
        self._name = self.cfg["optimizer"]
        self._base_lr = float(self.cfg["lr"])
        self._opt = build_optimizer(
            self._name,
            self.model.parameters(),
            lr=self._base_lr,
            momentum=float(self.cfg.get("momentum", 0.9)),
            weight_decay=float(self.cfg.get("weight_decay", 0.0)),
        )
        self._schedule = str(self.cfg.get("schedule", "none")).lower()
        self._min_lr = float(self.cfg.get("min_lr", 0.0))

    def begin_step(self, step: int) -> torch.optim.Optimizer:
        if self._schedule == "cosine":
            total = self.total_epochs * self.steps_per_epoch
            frac = min(step / max(total, 1), 1.0)
            lr = self._min_lr + 0.5 * (self._base_lr - self._min_lr) * (1 + math.cos(math.pi * frac))
            for group in self._opt.param_groups:
                group["lr"] = lr
        return self._opt

    @property
    def active_name(self) -> str:
        return self._name

    @property
    def active_optimizer(self) -> torch.optim.Optimizer:
        return self._opt


class SequentialController(Controller):
    """Fixed sequence of optimizer phases, switching at epoch boundaries.

    cfg keys: phases: list of {optimizer, lr, epochs, momentum?, weight_decay?,
    schedule?}; the last phase may omit `epochs` (runs to the end). Each new
    phase starts with a FRESH optimizer state (this is the point of arm C:
    plain warmup-then-switch with no state tricks).
    """

    def _build(self) -> None:
        self._phases = list(self.cfg["phases"])
        self._phase_idx = 0
        self._switch_steps: list[int] = []
        boundary = 0
        for phase in self._phases[:-1]:
            boundary += int(phase["epochs"]) * self.steps_per_epoch
            self._switch_steps.append(boundary)
        self._opt = self._make(self._phases[0])

    def _make(self, phase: dict) -> torch.optim.Optimizer:
        return build_optimizer(
            phase["optimizer"],
            self.model.parameters(),
            lr=float(phase["lr"]),
            momentum=float(phase.get("momentum", 0.9)),
            weight_decay=float(phase.get("weight_decay", 0.0)),
            nesterov=bool(phase.get("nesterov", False)),
        )

    def begin_step(self, step: int) -> torch.optim.Optimizer:
        if self._phase_idx < len(self._switch_steps) and step >= self._switch_steps[self._phase_idx]:
            old = self._phases[self._phase_idx]
            self._phase_idx += 1
            new = self._phases[self._phase_idx]
            old_lr = self.active_lr
            self._opt = self._make(new)
            self._record_switch(
                SwitchEvent(
                    step=step,
                    epoch=self.epoch_of(step),
                    from_name=old["optimizer"],
                    to_name=new["optimizer"],
                    from_lr=old_lr,
                    to_lr=float(new["lr"]),
                    reason="scheduled_phase_boundary",
                )
            )
        return self._opt

    @property
    def active_name(self) -> str:
        return self._phases[self._phase_idx]["optimizer"]

    @property
    def active_optimizer(self) -> torch.optim.Optimizer:
        return self._opt
