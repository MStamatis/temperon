"""Scheduled hand-off: cheap explorer -> sharpness-aware refiner (arm P).

Phase 6 "Sharpness-Budget Allocation", variant B. Stage 1 runs the proven
cyclic-catapult SGD schedule (cyclic-J) -- the cheapest per-epoch arm
(~18 s/epoch on wide-R110) and the fastest wall-clock route to mid-range
targets. At ONE fixed, scheduled epoch boundary (`handoff_frac` of training)
the model is handed to SAM+Muon on a single cosine anneal for the tail -- the
generalization refiner that lifts the ceiling by ~+3pp (Phase-5 2x2).

Rationale: the 2x2 attribution shows SAM's value is confined to final basin
selection, and late-phase SAM is sufficient in theory and practice
(arXiv:2410.10373: escape, then re-converge flatter within the same valley).
Paying SGD prices early and SAM+Muon prices (~66 s/epoch) only in the tail
targets super-convergence in BOTH epochs and wall-clock.

Unlike Phase-3's reactive switching (many trigger-driven switches, Adam
v-explosion), this is ONE pre-scheduled switch with momentum-only state
transfer: SGD's `momentum_buffer` and Muon's `momentum_buffer` accumulate the
same quantity (a discounted sum of gradients), so a raw copy is semantically
exact and there is no second moment to mis-seed.

Config shape::

    controller:
      type: handoff
      handoff_frac: 0.43        # or handoff_epoch: 43 (epoch boundary)
      handoff_transfer: momentum  # momentum | geometry | none
      stage1:                   # defaults to cyclic_catapult (cyclic-J)
        base_optimizer: sgd_momentum
        lr: 0.1
        ...
      stage2:                   # defaults to sam_catapult muon, single cosine
        lr: 0.01
        weight_decay: 0.2
        rho: 0.05
        warmup_steps: 200       # gentle lr re-entry after the hand-off
        sam_rho_ramp_steps: 400
        ...

Each stage sees only its own sub-horizon: stage 1's cycle structure completes
exactly at the hand-off; stage 2's anneal spans the tail.
"""

from __future__ import annotations

import torch

from eos_switch.controllers.base import Controller, SwitchEvent


class HandoffController(Controller):
    def _build(self) -> None:
        he = self.cfg.get("handoff_epoch")
        if he is None:
            he = round(float(self.cfg.get("handoff_frac", 0.43)) * self.total_epochs)
        # Clamp to [1, total-1]: both stages must own at least one epoch.
        self.handoff_epoch = max(1, min(int(he), self.total_epochs - 1))
        self._handoff_step = self.handoff_epoch * self.steps_per_epoch
        self.transfer = str(self.cfg.get("handoff_transfer", "momentum")).lower()
        if self.transfer not in ("momentum", "geometry", "none"):
            raise ValueError(f"unknown handoff_transfer {self.transfer!r}")

        from eos_switch.controllers import build_controller

        s1_cfg = dict(self.cfg.get("stage1") or {})
        s1_cfg.setdefault("type", "cyclic_catapult")
        s1_cfg.setdefault("base_optimizer", "sgd_momentum")
        s2_cfg = dict(self.cfg.get("stage2") or {})
        s2_cfg.setdefault("type", "sam_catapult")
        s2_cfg.setdefault("base_optimizer", "muon")
        s2_cfg.setdefault("n_cycles", 1)  # single cosine anneal over the tail
        s2_cfg.setdefault("t_mult", 1.0)
        self.s1 = build_controller(s1_cfg)
        self.s2 = build_controller(s2_cfg)
        if not getattr(self.s2, "sam", False):
            raise ValueError("handoff stage2 must be a SAM controller")
        self.s1.setup(self.model, self.steps_per_epoch, self.handoff_epoch)
        self.s2.setup(self.model, self.steps_per_epoch,
                      self.total_epochs - self.handoff_epoch)
        self._stage = 1
        self._done_handoff = False

    # --- loop protocol -------------------------------------------------------
    def begin_step(self, step: int) -> torch.optim.Optimizer:
        if step < self._handoff_step:
            return self.s1.begin_step(step)
        if not self._done_handoff:
            self._do_handoff(step)
        return self.s2.begin_step(step - self._handoff_step)

    def end_step(self, step: int, loss: float) -> None:
        if self._stage == 1:
            self.s1.end_step(step, loss)
        else:
            self.s2.end_step(step - self._handoff_step, loss)

    def end_epoch(self, epoch: int, val_acc: float) -> None:
        if epoch < self.handoff_epoch:
            self.s1.end_epoch(epoch, val_acc)
        else:
            self.s2.end_epoch(epoch - self.handoff_epoch, val_acc)

    def attach_probe_context(self, loss_fn, data, generator) -> None:
        for s in (self.s1, self.s2):
            if hasattr(s, "attach_probe_context"):
                s.attach_probe_context(loss_fn, data, generator)

    # --- SAM signals proxied to the loop (stage-aware) -----------------------
    @property
    def sam(self) -> bool:
        return self._stage == 2

    def sam_active(self, step: int) -> bool:
        return self._stage == 2 and self.s2.sam_active(step - self._handoff_step)

    @property
    def sam_freeze_bn(self) -> bool:
        return bool(getattr(self.s2, "sam_freeze_bn", False))

    @property
    def sam_period(self) -> int:
        return int(getattr(self.s2, "sam_period", 1))

    # --- the single scheduled switch -----------------------------------------
    def _do_handoff(self, step: int) -> None:
        old = self.s1.active_optimizer
        new = self.s2.active_optimizer
        base = getattr(new, "base_optimizer", new)  # unwrap the SAM shell
        if self.transfer == "momentum":
            n = self._copy_momentum(old, base)
        elif self.transfer == "geometry":
            from eos_switch.optimizers.state_transfer import transfer_state

            n = transfer_state(old, base, "muon", mode="geometry")["transferred"]
        else:
            n = 0
        from_name, from_lr = self.s1.active_name, self.s1.active_lr
        self._stage = 2
        self._done_handoff = True
        self._record_switch(SwitchEvent(
            step=step, epoch=self.epoch_of(step), from_name=from_name,
            to_name=self.s2.active_name, from_lr=from_lr,
            to_lr=self.s2.active_lr,
            reason=f"handoff|transfer:{self.transfer}|params:{n}",
        ))

    @staticmethod
    @torch.no_grad()
    def _copy_momentum(old_opt, new_opt) -> int:
        """Raw momentum_buffer copy old -> new for shared params. SGD and Muon
        buffers accumulate the same discounted gradient sum, so no reprojection
        is needed (Muon orthogonalizes at update time, not in the buffer)."""
        n = 0
        old_params = {p for g in old_opt.param_groups for p in g["params"]}
        for group in new_opt.param_groups:
            for p in group["params"]:
                if p not in old_params:
                    continue
                buf = old_opt.state.get(p, {}).get("momentum_buffer")
                if buf is not None:
                    new_opt.state[p]["momentum_buffer"] = buf.detach().clone()
                    n += 1
        return n

    # --- checkpoint/resume ---------------------------------------------------
    def state_dict(self) -> dict:
        return {"stage": self._stage, "done_handoff": self._done_handoff,
                "s1": self.s1.state_dict(), "s2": self.s2.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        # Restoring stage 2 marks the hand-off done WITHOUT re-transferring:
        # the tail optimizer's state arrives from the checkpoint right after
        # this call (loop loads controller state first, then the optimizer).
        self._stage = int(state.get("stage", self._stage))
        self._done_handoff = bool(state.get("done_handoff", self._done_handoff))
        self.s1.load_state_dict(state.get("s1", {}))
        self.s2.load_state_dict(state.get("s2", {}))

    # --- introspection -------------------------------------------------------
    @property
    def active_name(self) -> str:
        return (self.s1 if self._stage == 1 else self.s2).active_name

    @property
    def active_optimizer(self) -> torch.optim.Optimizer:
        return (self.s1 if self._stage == 1 else self.s2).active_optimizer
