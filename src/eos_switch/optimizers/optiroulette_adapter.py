"""Adapter around the upstream OptiRoulette package (pip: ``optiroulette``).

Acquisition route (a) succeeded: the package is installed from PyPI and its
source is vendored under ``third_party/optiroulette`` for study. Everything
else in this repo talks to OptiRoulette exclusively through this adapter so
the upstream API can change without touching controllers or the train loop.

Upstream behavior summary (v0.1.0, "as-is" for arm D):
- default profile: 17-epoch SGD warmup (lr 0.1, momentum 0.9, nesterov),
  warmup optimizer dropped from the pool afterwards;
- epoch-granularity random switching with avoid_repeat over an active pool
  of 7 (sgd, nadam, adam, adamw, ranger, adan, lion), LR-scaling rules
  applied when crossing optimizer families;
- each underlying optimizer keeps its own (possibly stale) state between
  activations; no cross-optimizer state transfer happens on a switch.
"""

from __future__ import annotations

from typing import Any

import torch

from optiroulette import OptiRoulette


class OptiRouletteAdapter:
    """Thin lifecycle wrapper exposing a stable interface to our harness."""

    def __init__(
        self,
        params,
        *,
        warmup: bool = True,
        warmup_epochs: int | None = None,
        seed: int | None = None,
        optimizer_specs: Any = None,
    ) -> None:
        roulette_overrides: dict[str, Any] = {}
        if not warmup:
            # Fully disable both fixed and plateau warmup modes.
            roulette_overrides = {
                "warmup_epochs": 0,
                "warmup_config": {},
                "warmup_optimizer": None,
                "drop_after_warmup": False,
            }
        elif warmup_epochs is not None:
            roulette_overrides = {"warmup_epochs": int(warmup_epochs)}

        kwargs: dict[str, Any] = {"seed": seed}
        if roulette_overrides:
            kwargs["roulette"] = roulette_overrides
        if optimizer_specs is not None:
            kwargs["optimizer_specs"] = optimizer_specs
        self.opt = OptiRoulette(params, **kwargs)
        self._consumed_switches = 0

    # -- lifecycle ---------------------------------------------------------
    def on_epoch_start(self, epoch: int) -> None:
        self.opt.on_epoch_start(epoch)

    def on_batch_start(self, batch_idx: int) -> None:
        self.opt.on_batch_start(batch_idx)

    def on_epoch_end(self, val_acc: float) -> None:
        self.opt.on_epoch_end(val_acc=val_acc)

    # -- introspection -----------------------------------------------------
    @property
    def active_name(self) -> str:
        return self.opt.active_optimizer_name

    @property
    def active_underlying(self) -> torch.optim.Optimizer:
        """The real optimizer behind the wrapper (probes need its state)."""
        return self.opt.active_optimizer

    @property
    def active_lr(self) -> float:
        return float(self.opt.param_groups[0]["lr"])

    @property
    def phase(self) -> str:
        return self.opt.phase

    def consume_new_switches(self) -> list[tuple[str | None, str, str]]:
        """Return (old, new, reason) entries appended since the last call."""
        history = self.opt.switch_history
        new = list(history[self._consumed_switches :])
        self._consumed_switches = len(history)
        return new
