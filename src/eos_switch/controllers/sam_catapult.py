"""SAM-coupled catapult controller (arm L).

Stage 1 (this file): Sharpness-Aware Minimization on top of the proven cyclic-
catapult schedule (arm J). The cosine warm-restart LR scheduling of
CyclicCatapultController is reused verbatim -- only the per-cycle base optimizer
is wrapped in a SAM two-pass shell. The training loop detects `sam == True` and
runs the second (perturbed) forward/backward pass; everything else (warm
restarts, optional pool-per-restart bandit, state transfer) is inherited.

Rationale: EoS catapults implicitly seek flat minima. SAM seeks them
explicitly. Putting SAM on the winning cyclic-catapult tests whether the
explicit sharpness penalty improves the proven super-convergence schedule.

Stage 2 (planned): EoS-coupled rho -- calibrate the SAM radius from the
batch-sharpness probe (attach_probe_context is wired by the loop the same way
as edge_lr / eos_restart). Stage 1 keeps rho fixed.
"""

from __future__ import annotations

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

    def _make(self, name: str):
        # Wrap the cyclic schedule's base optimizer in a SAM two-pass shell.
        return SAM(super()._make(name), rho=self.rho)

    @property
    def active_name(self) -> str:
        return f"sam:{super().active_name}"
