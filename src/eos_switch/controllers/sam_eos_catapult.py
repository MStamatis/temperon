"""Free-HVP Edge-of-Stability catapult (arm N -- the novelty).

Builds on the project's best, SAM+Muon-catapult, and revives arm K's idea
(EoS-triggered warm restarts) after removing its fatal flaw. arm K timed restarts
from batch_sharpness measured by a SEPARATE, expensive, noisy probe batch -- and
lost to the fixed cosine schedule. Here the curvature is FREE and noise-free: SAM
already computes two gradients on the SAME batch, whose difference is an exact
(quadratic) Hessian-vector product. SAM.second_step() stores

    last_sharpness = g_hat^T H g_hat   (the batch_sharpness / EoS quantity)

at zero extra cost. We read it to time the catapults:

    edge_ratio = last_sharpness * lr / (2 + 2*beta)        (edge at ratio = 1)

and restart (lr -> peak) when edge_ratio drops below `restart_ratio` -- i.e. the
moment the model has settled into a flat basin -- escaping to explore again.
Restarts stop in the final `1 - switch_until_frac` so the last cosine settles
(final ~ best). The (2+2*beta) edge is the SGD-momentum reference; for a Muon
base it is heuristic (no closed-form spectral EoS threshold), so restart_ratio is
the real knob. This reuses EosRestartController's schedule wholesale; only the
sharpness SOURCE changes (free from SAM, not a probe) and the optimizer is a
SAM-wrapped Muon driven two-pass by the loop (sam = True).
"""

from __future__ import annotations

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

        # EoS-restart schedule knobs (same as arm K).
        self.warmup_steps = int(self.cfg.get("warmup_steps", 200))
        self.check_every = int(self.cfg.get("check_every", 50))
        self.restart_ratio = float(self.cfg.get("restart_ratio", 0.1))
        self.cycle_epochs = float(self.cfg.get("cycle_epochs", 20.0))
        self.min_dwell_checks = int(self.cfg.get("min_dwell_checks", 2))
        self.switch_until_frac = float(self.cfg.get("switch_until_frac", 0.8))

        self._cycle_start = self.warmup_steps
        self._checks_since_restart = 0
        self._final_started = False
        self._lr = self.peak_lr
        self._probe_seconds = 0.0   # the curvature is free -> no probe overhead
        self.n_probes = 0
        self.checks: list[dict] = []
        # Sharpness comes from SAM, not a probe batch; mark _data non-None so the
        # inherited begin_step runs its restart check.
        self._data = object()
        self._loss_fn = self._gen = None

    def _probe_edge_ratio(self):
        # Free same-batch sharpness from the last SAM two-pass (no HVP probe).
        s = getattr(self._opt, "last_sharpness", None)
        s = 0.0 if s is None else float(s)
        threshold = (2.0 + 2.0 * self.momentum) / max(self._lr, 1e-9)
        return s / threshold, s

    @property
    def active_name(self) -> str:
        return f"sam_eos:{self.base_name}"
