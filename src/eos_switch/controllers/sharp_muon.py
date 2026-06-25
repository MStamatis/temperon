"""Sharpness-aware Muon on the cyclic-catapult schedule (arm M -- the novelty).

Single-pass: unlike sam_catapult (which signals the loop to run a second forward-
backward), this controller has NO `sam` flag, so train/loop.py runs one pass and
calls step(). The sharpness-awareness lives inside SharpMuon (temporal / spectral;
see optimizers/sharp_muon.py). The cyclic warm-restart LR schedule is inherited
verbatim from CyclicCatapultController -- only the per-cycle optimizer changes.

Goal: recover SAM's +3pp Muon-final benefit at ~half the SAM compute (1 pass).
Bar = SAM+Muon-catapult 0.8226; baseline to beat = plain Muon 0.795.
"""

from __future__ import annotations

from eos_switch.controllers.cyclic import CyclicCatapultController
from eos_switch.optimizers.sharp_muon import SharpMuon


class SharpMuonCatapultController(CyclicCatapultController):
    def _build(self) -> None:
        self.sharp_mode = str(self.cfg.get("sharp_mode", "temporal"))
        self.rho = float(self.cfg.get("rho", 0.05))
        self.spec_iters = int(self.cfg.get("spec_iters", 2))
        self.ns_steps = int(self.cfg.get("ns_steps", 5))
        super()._build()

    def _make(self, name: str):
        # Always build SharpMuon from the (single) base spec; ignore the pool path.
        s = self.specs[name]
        return SharpMuon(
            self.model.parameters(),
            lr=float(s["lr"]),
            momentum=float(s.get("momentum", 0.9)),
            nesterov=bool(s.get("nesterov", True)),
            weight_decay=float(s.get("weight_decay", 0.0)),
            ns_steps=self.ns_steps,
            mode=self.sharp_mode,
            rho=self.rho,
            spec_iters=self.spec_iters,
        )

    @property
    def active_name(self) -> str:
        return f"sharpmuon:{self.sharp_mode}"
