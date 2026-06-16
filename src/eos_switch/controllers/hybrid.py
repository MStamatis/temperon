"""Adam->SGD one-cycle hybrid (arm I) -- targets super-convergence.

Motivated directly by our measurements: AdamW reaches moderate accuracy ~3x
faster early (0.65 @ ep16 vs strong-SGD's ep45) but caps low (0.732), while
SGD is slow early but generalizes best (0.774). So: ride AdamW early for fast
descent, then hand off to SGD-momentum (with cosine decay) for the final
generalization push. This is the SWATS recipe (Keskar & Socher) and the
*opposite order* of arm C (which was SGD->AdamW).

Handoff fires on a train-loss-EMA plateau, or by a fraction-of-training cap,
whichever comes first. The Adam first moment is carried into SGD's momentum
buffer (geometry-consistent state transfer).
"""

from __future__ import annotations

import math

import torch

from eos_switch.controllers.base import Controller, SwitchEvent
from eos_switch.optimizers.pool import build_optimizer
from eos_switch.optimizers.state_transfer import transfer_state


class AdamSgdHybridController(Controller):
    def _build(self) -> None:
        a = self.cfg.get("adam", {})
        self._adam_name = a.get("optimizer", "adamw")
        self._adam_lr = float(a.get("lr", 1e-3))
        self._adam = build_optimizer(
            self._adam_name,
            self.model.parameters(),
            lr=self._adam_lr,
            betas=tuple(a.get("betas", (0.9, 0.999))),
            weight_decay=float(a.get("weight_decay", 0.01)),
        )
        s = self.cfg.get("sgd", {})
        self._sgd_lr = float(s.get("lr", 0.1))
        self._sgd_momentum = float(s.get("momentum", 0.9))
        self._sgd_wd = float(s.get("weight_decay", 5e-4))
        self._sgd_nesterov = bool(s.get("nesterov", True))
        self._sgd_min_lr = float(s.get("min_lr", 0.0))

        self.handoff_frac = float(self.cfg.get("handoff_frac", 0.3))
        self.plateau_window = int(self.cfg.get("plateau_window", 3))
        self.plateau_delta = float(self.cfg.get("plateau_delta", 0.01))
        self.loss_ema_beta = float(self.cfg.get("loss_ema_beta", 0.9))
        self.transfer_mode = str(self.cfg.get("state_transfer", "geometry"))

        self.phase = "adam"
        self._sgd: torch.optim.Optimizer | None = None
        self._handoff_step: int | None = None
        self._loss_ema: float | None = None
        self._epoch_loss: list[float] = []
        self._seen_epoch = 0

    def _plateau(self) -> bool:
        h = self._epoch_loss
        if len(h) <= self.plateau_window:
            return False
        old, new = h[-1 - self.plateau_window], h[-1]
        if old is None or new is None or old == 0:
            return False
        return (old - new) / abs(old) < self.plateau_delta

    def _handoff(self, step: int, reason: str) -> None:
        self._sgd = build_optimizer(
            "sgd_momentum",
            self.model.parameters(),
            lr=self._sgd_lr,
            momentum=self._sgd_momentum,
            weight_decay=self._sgd_wd,
            nesterov=self._sgd_nesterov,
        )
        info = transfer_state(self._adam, self._sgd, "sgd_momentum", mode=self.transfer_mode)
        self.phase = "sgd"
        self._handoff_step = step
        self._record_switch(SwitchEvent(
            step=step, epoch=self.epoch_of(step), from_name=self._adam_name,
            to_name="sgd_momentum", from_lr=self._adam_lr, to_lr=self._sgd_lr,
            reason=f"adam_sgd_handoff:{reason}|transfer:{info['mode']}",
        ))

    def begin_step(self, step: int) -> torch.optim.Optimizer:
        spe = max(self.steps_per_epoch, 1)
        epoch = step // spe
        if step % spe == 0 and epoch != self._seen_epoch:
            self._seen_epoch = epoch
            self._epoch_loss.append(self._loss_ema if self._loss_ema is not None else float("nan"))

        if self.phase == "adam":
            cap = self.handoff_frac * self.total_epochs
            if epoch >= cap or self._plateau():
                self._handoff(step, reason="plateau" if self._plateau() else "cap")

        if self.phase == "sgd":
            total = self.total_epochs * spe
            p = (step - self._handoff_step) / max(total - self._handoff_step, 1)
            p = min(max(p, 0.0), 1.0)
            lr = self._sgd_min_lr + 0.5 * (self._sgd_lr - self._sgd_min_lr) * (1 + math.cos(math.pi * p))
            for g in self._sgd.param_groups:
                g["lr"] = lr

        return self.active_optimizer

    def end_step(self, step: int, loss: float) -> None:
        be = self.loss_ema_beta
        self._loss_ema = loss if self._loss_ema is None else be * self._loss_ema + (1 - be) * loss

    @property
    def active_name(self) -> str:
        return self._adam_name if self.phase == "adam" else "sgd_momentum"

    @property
    def active_optimizer(self) -> torch.optim.Optimizer:
        return self._adam if self.phase == "adam" else self._sgd
