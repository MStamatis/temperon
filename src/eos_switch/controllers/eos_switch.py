"""EoS-aware switching controller (arm F) -- replaces the random roulette.

Switch rule (Phase 3):
1. Every `check_every` steps, measure the active optimizer's stability_margin
   (preconditioned sharpness vs its EoS threshold). If it stays below
   `margin_floor` for `k` consecutive checks (the optimizer is stuck at its
   own edge), trigger a switch.
2. Pick the next optimizer as argmax over the pool of `lr_i * margin_i`,
   where each candidate's margin is a cheap 1-HVP directional estimate in its
   geometry. Tie-break: least-recently-used.
3. Transfer state geometry-consistently (see optimizers/state_transfer.py).

A dense reward (Delta loss / second + lambda * margin) is logged per check
for analysis / a future bandit. An optional fixed warmup (e.g. 17 epochs of
SGD, matching arm D) precedes the EoS phase for a fair comparison.

The training loop attaches the probe context (loss_fn, data, generator) via
`attach_probe_context`; all probing runs in fp32 inside probe_precision().
"""

from __future__ import annotations

import time

import torch

from eos_switch.controllers.base import Controller, SwitchEvent
from eos_switch.optimizers.pool import build_optimizer, stability_threshold
from eos_switch.optimizers.state_transfer import transfer_state


class EosSwitchController(Controller):
    def _build(self) -> None:
        self.check_every = int(self.cfg.get("check_every", 100))
        self.margin_floor = float(self.cfg.get("margin_floor", 0.1))
        self.k = int(self.cfg.get("k", 3))
        # Loss-plateau trigger (primary, since adaptive optimizers rarely reach
        # their loose EoS edge): switch if the loss-EMA improved by less than
        # `plateau_delta` (relative) over the last `plateau_window` checks.
        self.plateau_window = int(self.cfg.get("plateau_window", 5))
        self.plateau_delta = float(self.cfg.get("plateau_delta", 0.01))
        self.loss_ema_beta = float(self.cfg.get("loss_ema_beta", 0.9))
        # Cooldown: no second switch for this many checks (anti-thrash).
        self.cooldown_checks = int(self.cfg.get("cooldown_checks", 3))
        # Annealed switching window: allow switches only in the first
        # `switch_until_frac` of training, then settle (final ~ best).
        self.switch_until_frac = float(self.cfg.get("switch_until_frac", 0.6))
        self.transfer_mode = str(self.cfg.get("state_transfer", "geometry"))
        self.reward_lambda = float(self.cfg.get("reward_lambda", 1.0))
        self.adam_warm_steps = int(self.cfg.get("adam_warm_steps", 1000))
        self.grad_sq_beta = float(self.cfg.get("grad_sq_beta", 0.99))
        self.micro_batch = int(self.cfg.get("micro_batch", 128))
        self.power_iters = int(self.cfg.get("power_iters", 15))

        self.warmup_epochs = int(self.cfg.get("warmup_epochs", 0))
        self.warmup_name = str(self.cfg.get("warmup_optimizer", "sgd_momentum"))
        self.warmup_lr = float(self.cfg.get("warmup_lr", 0.1))
        self.warmup_momentum = float(self.cfg.get("warmup_momentum", 0.9))
        self.warmup_wd = float(self.cfg.get("warmup_weight_decay", 5e-4))

        # Pool: list of {optimizer, lr, momentum?, weight_decay?, betas?}
        self.specs = {s["optimizer"]: s for s in self.cfg["pool"]}
        self.opts: dict[str, torch.optim.Optimizer] = {}
        for name, spec in self.specs.items():
            self.opts[name] = self._make(spec)
        self.lrs = {n: float(s["lr"]) for n, s in self.specs.items()}
        self.momentums = {n: float(s.get("momentum", 0.0)) for n, s in self.specs.items()}

        if self.warmup_epochs > 0:
            self._warmup_opt = build_optimizer(
                self.warmup_name,
                self.model.parameters(),
                lr=self.warmup_lr,
                momentum=self.warmup_momentum,
                weight_decay=self.warmup_wd,
                nesterov=bool(self.cfg.get("warmup_nesterov", True)),
            )
            self._active = "__warmup__"
            self.phase = "warmup"
        else:
            self._active = next(iter(self.opts))
            self._warmup_opt = None
            self.phase = "eos"

        self._low_margin_count = 0
        self._last_used: dict[str, int] = {}
        self._grad_sq_ema: dict[torch.Tensor, torch.Tensor] = {}
        self._probe_seconds = 0.0
        self.n_probes = 0
        self._transitioned = False
        # plateau / cooldown / per-check diagnostics
        self._loss_ema: float | None = None
        self._loss_hist: list[float] = []
        self._checks_since_switch = 10**9  # first post-warmup switch ungated
        self.checks: list[dict] = []
        # dense-reward bookkeeping
        self._last_reward_step = 0
        self._last_reward_loss: float | None = None
        self._last_reward_time = time.perf_counter()
        self.rewards: list[dict] = []
        # probe context (attached by the loop)
        self._loss_fn = None
        self._data = None
        self._gen = None

    def _make(self, spec: dict) -> torch.optim.Optimizer:
        return build_optimizer(
            spec["optimizer"],
            self.model.parameters(),
            lr=float(spec["lr"]),
            momentum=float(spec.get("momentum", 0.9)),
            betas=tuple(spec.get("betas", (0.9, 0.999))),
            weight_decay=float(spec.get("weight_decay", 0.0)),
        )

    # ---- probe context -------------------------------------------------
    def attach_probe_context(self, loss_fn, data, generator) -> None:
        self._loss_fn = loss_fn
        self._data = data
        self._gen = generator

    @property
    def _has_probe_ctx(self) -> bool:
        return self._loss_fn is not None and self._data is not None

    # ---- pure decision helpers (unit-tested without probes) ------------
    def _register_active_margin(self, margin: float) -> bool:
        if margin < self.margin_floor:
            self._low_margin_count += 1
        else:
            self._low_margin_count = 0
        return self._low_margin_count >= self.k

    def _window_open(self, step: int) -> bool:
        return step < self.switch_until_frac * self.total_epochs * self.steps_per_epoch

    def _select_next(self, margins: dict[str, float], exclude: str | None) -> str:
        cands = [n for n in self.opts if n != exclude]
        ranked = sorted(
            cands,
            key=lambda n: (-(self.lrs[n] * margins[n]), self._last_used.get(n, -1)),
        )
        return ranked[0]

    # ---- probing -------------------------------------------------------
    def _probe_all_margins(self) -> dict[str, float]:
        """1-HVP directional stability margin for every pool optimizer, sharing
        one gradient graph (active + candidates in a single cheap probe)."""
        from eos_switch.probes.hvp import HvpOperator, preserve_bn_stats
        from eos_switch.probes.precision import probe_precision
        from eos_switch.probes.sharpness import candidate_margin

        batch = self._data.sample_probe_batch(self.micro_batch, self._gen)
        t0 = time.perf_counter()
        margins: dict[str, float] = {}
        with probe_precision(), preserve_bn_stats(self.model):
            op = HvpOperator(self.model, self._loss_fn, batch)
            for name in self.opts:
                out = candidate_margin(
                    op, name, self.lrs[name], self.momentums[name], self._grad_sq_ema
                )
                margins[name] = out["margin"]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._probe_seconds += time.perf_counter() - t0
        self.n_probes += 1
        return margins

    def _plateau(self) -> bool:
        """True if loss-EMA improved < plateau_delta (relative) over the last
        plateau_window checks."""
        h = self._loss_hist
        if len(h) <= self.plateau_window:
            return False
        old, new = h[-1 - self.plateau_window], h[-1]
        if old is None or not (old == old) or old == 0.0:  # NaN/zero guard
            return False
        return (old - new) / abs(old) < self.plateau_delta

    # ---- switching -----------------------------------------------------
    def _switch_to(self, step: int, new_name: str, reason: str) -> None:
        old_name = self.active_name
        old_opt = self.active_optimizer
        new_opt = self.opts[new_name]
        info = transfer_state(
            old_opt,
            new_opt,
            new_name,
            mode=self.transfer_mode,
            grad_sq_ema=self._grad_sq_ema,
            adam_warm_steps=self.adam_warm_steps,
        )
        self._record_switch(
            SwitchEvent(
                step=step,
                epoch=self.epoch_of(step),
                from_name=old_name,
                to_name=new_name,
                from_lr=self.active_lr if old_name != "__warmup__" else self.warmup_lr,
                to_lr=self.lrs[new_name],
                reason=f"{reason}|transfer:{info['mode']}",
            )
        )
        self._active = new_name
        self._last_used[new_name] = step
        self._low_margin_count = 0
        self._checks_since_switch = 0

    def begin_step(self, step: int) -> torch.optim.Optimizer:
        epoch = step // max(self.steps_per_epoch, 1)

        # warmup -> eos transition (once)
        if self.phase == "warmup" and epoch >= self.warmup_epochs:
            self.phase = "eos"
            margins = self._probe_all_margins() if self._has_probe_ctx else {n: 1.0 for n in self.opts}
            nxt = self._select_next(margins, exclude=None)
            self._switch_to(step, nxt, reason="warmup_exit")
            self._transitioned = True
            return self.active_optimizer

        # EoS switching: plateau (primary) or margin-floor (secondary), gated
        # by a cooldown to prevent thrashing.
        if (
            self.phase == "eos"
            and self._has_probe_ctx
            and step > 0
            and step % self.check_every == 0
        ):
            margins = self._probe_all_margins()
            active_margin = margins[self.active_name]
            self._loss_hist.append(self._loss_ema if self._loss_ema is not None else float("nan"))
            floor_hit = self._register_active_margin(active_margin)
            plateau_hit = self._plateau()
            trigger = "margin_floor" if floor_hit else ("plateau" if plateau_hit else None)
            window_open = self._window_open(step)
            rec = {
                "step": step,
                "epoch": round(self.epoch_of(step), 3),
                "optimizer": self.active_name,
                "margin": round(active_margin, 4),
                "loss_ema": self._loss_ema,
                "low_margin_count": self._low_margin_count,
                "trigger": trigger,
                "window_open": window_open,
            }
            if trigger and window_open and self._checks_since_switch >= self.cooldown_checks:
                nxt = self._select_next(margins, exclude=self.active_name)
                self._switch_to(step, nxt, reason=trigger)
                rec["switched_to"] = nxt
            else:
                self._checks_since_switch += 1
            self.checks.append(rec)

        return self.active_optimizer

    def end_step(self, step: int, loss: float) -> None:
        # Loss EMA drives the plateau trigger.
        be = self.loss_ema_beta
        self._loss_ema = loss if self._loss_ema is None else be * self._loss_ema + (1 - be) * loss

        # Update the running grad^2 EMA (for fresh second-moment estimates).
        b = self.grad_sq_beta
        with torch.no_grad():
            for p in self.model.parameters():
                if p.grad is None:
                    continue
                g2 = p.grad.detach() ** 2
                if p not in self._grad_sq_ema:
                    self._grad_sq_ema[p] = g2.clone()
                else:
                    self._grad_sq_ema[p].mul_(b).add_(g2, alpha=1 - b)

        # Dense reward: Delta loss / second + lambda * (1 - low_margin_count/k)
        if step - self._last_reward_step >= self.check_every:
            now = time.perf_counter()
            dt = max(now - self._last_reward_time, 1e-6)
            dloss = (self._last_reward_loss - loss) if self._last_reward_loss is not None else 0.0
            margin_term = 1.0 - min(self._low_margin_count / max(self.k, 1), 1.0)
            self.rewards.append(
                {
                    "step": step,
                    "optimizer": self.active_name,
                    "reward": dloss / dt + self.reward_lambda * margin_term,
                    "dloss_per_s": dloss / dt,
                }
            )
            self._last_reward_step = step
            self._last_reward_loss = loss
            self._last_reward_time = now

    def overhead_frac(self, total_train_seconds: float) -> float:
        return self._probe_seconds / max(total_train_seconds, 1e-9)

    @property
    def active_name(self) -> str:
        return self.warmup_name if self._active == "__warmup__" else self._active

    @property
    def active_optimizer(self) -> torch.optim.Optimizer:
        return self._warmup_opt if self._active == "__warmup__" else self.opts[self._active]
