"""Controller driving the upstream OptiRoulette wrapper (arms D and E).

cfg keys:
- warmup: bool (default true) -- false gives arm E (roulette from epoch 0);
- warmup_epochs: optional int override of the upstream default (17);
- optimizer_specs: optional explicit spec passthrough (default: upstream
  bundled profile, i.e. "as-is");
- seed: injected by the harness so switching sequences vary across seeds.
"""

from __future__ import annotations

import torch

from eos_switch.controllers.base import Controller, SwitchEvent
from eos_switch.optimizers.optiroulette_adapter import OptiRouletteAdapter


class OptiRouletteController(Controller):
    def _build(self) -> None:
        self.adapter = OptiRouletteAdapter(
            self.model.parameters(),
            warmup=bool(self.cfg.get("warmup", True)),
            warmup_epochs=self.cfg.get("warmup_epochs"),
            seed=self.cfg.get("seed"),
            optimizer_specs=self.cfg.get("optimizer_specs"),
        )
        self._last_epoch_started = -1

    def begin_step(self, step: int) -> torch.optim.Optimizer:
        epoch = step // max(self.steps_per_epoch, 1)
        batch_idx = step % max(self.steps_per_epoch, 1)
        if epoch != self._last_epoch_started:
            self.adapter.on_epoch_start(epoch)
            self._last_epoch_started = epoch
        self.adapter.on_batch_start(batch_idx)
        for old, new, reason in self.adapter.consume_new_switches():
            self._record_switch(
                SwitchEvent(
                    step=step,
                    epoch=self.epoch_of(step),
                    from_name=old,
                    to_name=new,
                    from_lr=None,  # upstream rescales lr internally on switch
                    to_lr=self.adapter.active_lr,
                    reason=f"optiroulette:{reason}",
                )
            )
        return self.adapter.opt

    def end_epoch(self, epoch: int, val_acc: float) -> None:
        self.adapter.on_epoch_end(val_acc)

    @property
    def active_name(self) -> str:
        return self.adapter.active_name

    @property
    def active_optimizer(self) -> torch.optim.Optimizer:
        # The underlying optimizer, not the wrapper: probes must see the
        # real optimizer class and its state (exp_avg_sq etc.).
        return self.adapter.active_underlying
