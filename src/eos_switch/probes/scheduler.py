"""Amortized probe scheduling with switch-burst densification.

Regular cadence: one full probe every `probe_every` steps on a dedicated
micro-batch. Around each optimizer switch the scheduler (a) re-emits the
last `burst_len` regular records tagged as pre-switch context, and (b) runs
`burst_len` consecutive probes right after the switch — giving the
"5 before / 5 after" sharpness picture for catapult analysis.

Wall-clock spent inside probes is accumulated so the report can state probe
overhead as a fraction of total training time (target < 10%).
"""

from __future__ import annotations

import time
from collections import deque

import torch

from eos_switch.probes.sharpness import full_probe


class ProbeScheduler:
    def __init__(self, cfg, model, loss_fn, data, controller, logger, generator) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.data = data
        self.controller = controller
        self.logger = logger
        self.generator = generator
        self.probe_every = int(cfg.get("probe_every", 50))
        self.micro_batch = int(cfg.get("micro_batch", 128))
        self.power_iters = int(cfg.get("power_iters", 15))
        self.burst_len = int(cfg.get("burst_len", 5))
        # Preconditioned lambda_max (the expensive part: `power_iters` HVPs)
        # runs on every Nth regular probe; batch_sharpness runs on all.
        self.precond_every = max(1, int(cfg.get("precond_every", 1)))
        self._recent: deque[dict] = deque(maxlen=self.burst_len)
        self._pending_burst = 0
        self._switch_step: int | None = None
        self._probe_seconds = 0.0
        self.n_probes = 0

    def notify_switch(self, event) -> None:
        self._pending_burst = self.burst_len
        self._switch_step = event.step
        for rec in self._recent:
            self.logger.log_probe(
                {**rec, "tag": "pre_switch", "switch_step": event.step, "switch_to": event.to_name}
            )
        self.logger.log_probe(
            {
                "type": "switch",
                "step": event.step,
                "from": event.from_name,
                "to": event.to_name,
                "reason": event.reason,
            }
        )

    def maybe_probe(self, step: int, t_now: float) -> dict | None:
        burst = self._pending_burst > 0
        if not burst and step % self.probe_every != 0:
            return None
        t0 = time.perf_counter()
        batch = self.data.sample_probe_batch(self.micro_batch, self.generator)
        include_precond = burst or (self.n_probes % self.precond_every == 0)
        rec = full_probe(
            self.model,
            self.loss_fn,
            batch,
            self.controller.active_optimizer,
            self.controller.active_name,
            lr=self.controller.active_lr,
            iters=self.power_iters,
            generator=self.generator,
            include_precond=include_precond,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        self._probe_seconds += dt
        self.n_probes += 1
        rec.update(
            {
                "type": "probe",
                "step": step,
                "t": round(t_now, 3),
                "optimizer": self.controller.active_name,
                "probe_time_s": round(dt, 4),
            }
        )
        if burst:
            rec["tag"] = "post_switch"
            rec["switch_step"] = self._switch_step
            self._pending_burst -= 1
        else:
            self._recent.append(rec)
        self.logger.log_probe(rec)
        return rec

    def overhead_frac(self, total_train_seconds: float) -> float:
        return self._probe_seconds / max(total_train_seconds, 1e-9)
