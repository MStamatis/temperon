"""Cyclic-catapult schedule (arm J): warm restarts (SGDR), evolving the one
working ingredient of OptiRoulette -- the high-LR catapult phase -- while
dropping the destructive random switching.

Equal-length cosine cycles: each cycle starts at the peak lr (catapult back to
the Edge of Stability) and cosine-decays to min_lr, then restarts. Anytime
high accuracy at each restart is exactly the super-convergence signal.

Base-optimizer-agnostic: `base_optimizer` can be sgd_momentum / adam / adamw,
so the same schedule can be tried on any optimizer. Optionally a `pool` of
optimizers is given, and a UCB bandit picks the optimizer for each new cycle
(rewarded by that cycle's loss-EMA drop), with geometry-consistent state
transfer at the restart -- the only constructive use of OptiRoulette's pool.
"""

from __future__ import annotations

import math

import torch

from eos_switch.controllers.base import Controller, SwitchEvent
from eos_switch.optimizers.pool import build_optimizer
from eos_switch.optimizers.state_transfer import transfer_state


class _UCB:
    def __init__(self, arms, c=2.0):
        self.arms = list(arms)
        self.n = {a: 0 for a in arms}
        self.v = {a: 0.0 for a in arms}
        self.t = 0
        self.c = c

    def select(self):
        for a in self.arms:
            if self.n[a] == 0:
                return a
        return max(self.arms, key=lambda a: self.v[a] + self.c * math.sqrt(math.log(self.t + 1) / self.n[a]))

    def update(self, a, r):
        self.n[a] += 1
        self.t += 1
        self.v[a] += (r - self.v[a]) / self.n[a]


class CyclicCatapultController(Controller):
    def _build(self) -> None:
        self.n_cycles = int(self.cfg.get("n_cycles", 4))
        self.min_lr = float(self.cfg.get("min_lr", 0.0))
        self.transfer_mode = str(self.cfg.get("state_transfer", "geometry"))
        self.loss_ema_beta = float(self.cfg.get("loss_ema_beta", 0.9))

        # Either a single base optimizer, or a pool selected per restart.
        pool = self.cfg.get("pool")
        if pool:
            self.specs = {s["optimizer"]: s for s in pool}
        else:
            b = self.cfg.get("base_optimizer", "sgd_momentum")
            self.specs = {b: {
                "optimizer": b,
                "lr": float(self.cfg.get("lr", 0.1)),
                "momentum": float(self.cfg.get("momentum", 0.9)),
                "weight_decay": float(self.cfg.get("weight_decay", 5e-4)),
                "nesterov": bool(self.cfg.get("nesterov", True)),
                "betas": tuple(self.cfg.get("betas", (0.9, 0.999))),
            }}
        self.names = list(self.specs)
        self._name = self.names[0]
        self._opt = self._make(self._name)
        self._bandit = _UCB(self.names, c=float(self.cfg.get("ucb_c", 2.0))) if len(self.names) > 1 else None

        self._last_cycle = 0
        self._loss_ema: float | None = None
        self._cycle_start_loss: float | None = None
        self.restarts: list[dict] = []

    def _make(self, name: str) -> torch.optim.Optimizer:
        s = self.specs[name]
        return build_optimizer(
            name, self.model.parameters(), lr=float(s["lr"]),
            momentum=float(s.get("momentum", 0.9)),
            betas=tuple(s.get("betas", (0.9, 0.999))),
            weight_decay=float(s.get("weight_decay", 0.0)),
            nesterov=bool(s.get("nesterov", False)),
        )

    def _cycle_len_steps(self) -> float:
        total = self.total_epochs * max(self.steps_per_epoch, 1)
        return max(total / self.n_cycles, 1.0)

    def _position(self, step: int):
        cl = self._cycle_len_steps()
        idx = min(int(step / cl), self.n_cycles - 1)
        frac = (step - idx * cl) / cl
        return idx, min(max(frac, 0.0), 1.0)

    def begin_step(self, step: int) -> torch.optim.Optimizer:
        idx, frac = self._position(step)
        if idx != self._last_cycle:  # a restart (catapult)
            self._on_restart(step)
            self._last_cycle = idx
        peak = float(self.specs[self._name]["lr"])
        lr = self.min_lr + 0.5 * (peak - self.min_lr) * (1 + math.cos(math.pi * frac))
        for g in self._opt.param_groups:
            g["lr"] = lr
        return self._opt

    def _on_restart(self, step: int) -> None:
        if self._bandit is not None:
            reward = 0.0
            if self._cycle_start_loss is not None and self._loss_ema is not None:
                reward = self._cycle_start_loss - self._loss_ema
            self._bandit.update(self._name, reward)
            nxt = self._bandit.select()
            if nxt != self._name:
                old = self._opt
                self._opt = self._make(nxt)
                transfer_state(old, self._opt, nxt, mode=self.transfer_mode)
                self._record_switch(SwitchEvent(
                    step=step, epoch=self.epoch_of(step), from_name=self._name,
                    to_name=nxt, from_lr=float(self.specs[self._name]["lr"]),
                    to_lr=float(self.specs[nxt]["lr"]),
                    reason=f"cycle_restart|transfer:{self.transfer_mode}",
                ))
                self._name = nxt
        self.restarts.append({"step": step, "epoch": round(self.epoch_of(step), 3),
                              "optimizer": self._name, "loss_ema": self._loss_ema})
        self._cycle_start_loss = self._loss_ema

    def end_step(self, step: int, loss: float) -> None:
        be = self.loss_ema_beta
        self._loss_ema = loss if self._loss_ema is None else be * self._loss_ema + (1 - be) * loss
        if self._cycle_start_loss is None:
            self._cycle_start_loss = self._loss_ema

    @property
    def active_name(self) -> str:
        return self._name

    @property
    def active_optimizer(self) -> torch.optim.Optimizer:
        return self._opt
