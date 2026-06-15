"""Per-layer geometry pool with a UCB bandit (arm G, Phase 4).

Each layer-group's >=2D weights are assigned a geometry from
{muon, adamw, lion, sgd_momentum} by an independent UCB bandit; all 1D params
share a fixed AdamW. Geometries are re-selected every `select_every` steps;
the shared reward is global Delta-loss/second. Each bandit is seeded with a
context prior from the group's gradient (stable rank + noise scale). On a
geometry change the group's optimizer state is transferred geometry-
consistently (see optimizers/state_transfer.py). An optional fixed SGD warmup
(matching the other arms) precedes the per-layer phase.

Context features are computed from gradients only (no HVPs), so probe overhead
stays well under 10%.
"""

from __future__ import annotations

import math
import time

import torch

from eos_switch.controllers.base import Controller, SwitchEvent
from eos_switch.optimizers.layer_groups import MultiOptimizer, group_parameters
from eos_switch.optimizers.pool import build_optimizer
from eos_switch.optimizers.state_transfer import transfer_state

GEOMS = ("muon", "adamw", "lion", "sgd_momentum")


class UCBBandit:
    """UCB1 over a fixed arm set, with optional prior value bias."""

    def __init__(self, arms, c: float = 2.0, prior: dict | None = None) -> None:
        self.arms = list(arms)
        self.c = c
        self.n = {a: 0 for a in self.arms}
        self.value = {a: (prior.get(a, 0.0) if prior else 0.0) for a in self.arms}
        self.total = 0

    def select(self) -> str:
        for a in self.arms:  # try each arm once first
            if self.n[a] == 0:
                return a
        return max(
            self.arms,
            key=lambda a: self.value[a] + self.c * math.sqrt(math.log(self.total + 1) / self.n[a]),
        )

    def update(self, arm: str, reward: float) -> None:
        self.n[arm] += 1
        self.total += 1
        self.value[arm] += (reward - self.value[arm]) / self.n[arm]


def _context_prior(stable_rank: float, noise: float) -> dict:
    """Light context bias: high stable-rank favours Muon (orthogonalization),
    high gradient noise favours AdamW (adaptive). Small biases only."""
    prior = {g: 0.0 for g in GEOMS}
    prior["muon"] += 0.05 * stable_rank        # stable_rank in ~[0,1] (normalized)
    prior["adamw"] += 0.05 * noise             # noise in ~[0,1]
    return prior


class PerLayerController(Controller):
    def _build(self) -> None:
        self.select_every = int(self.cfg.get("select_every", 200))
        # Freeze geometry re-selection after this fraction of training so the
        # model settles (UCB has converged by then; avoids late-training churn).
        self.select_until_frac = float(self.cfg.get("select_until_frac", 0.8))
        self.ucb_c = float(self.cfg.get("ucb_c", 2.0))
        self.transfer_mode = str(self.cfg.get("state_transfer", "geometry"))
        self.grad_sq_beta = float(self.cfg.get("grad_sq_beta", 0.99))
        self.onedim_lr = float(self.cfg.get("onedim_lr", 1.0e-3))
        self.lrs = {g: float(lr) for g, lr in self.cfg.get("geom_lrs", {}).items()}
        for g in GEOMS:  # defaults
            self.lrs.setdefault(g, {"muon": 0.02, "adamw": 1e-3, "lion": 3e-4, "sgd_momentum": 0.1}[g])

        self.warmup_epochs = int(self.cfg.get("warmup_epochs", 0))
        self.warmup_lr = float(self.cfg.get("warmup_lr", 0.1))

        self.geom_groups, self.onedim = group_parameters(self.model)
        self.regions = list(self.geom_groups)
        self._region_geom = {r: "adamw" for r in self.regions}  # cold start
        self._region_opt: dict[str, torch.optim.Optimizer] = {}
        self._onedim_opt = build_optimizer("adamw", self.onedim, lr=self.onedim_lr, weight_decay=0.0)
        self._bandits = {r: UCBBandit(GEOMS, c=self.ucb_c) for r in self.regions}

        if self.warmup_epochs > 0:
            self._warmup_opt = build_optimizer(
                "sgd_momentum", self.model.parameters(), lr=self.warmup_lr,
                momentum=0.9, weight_decay=float(self.cfg.get("warmup_weight_decay", 5e-4)),
                nesterov=True,
            )
            self.phase = "warmup"
        else:
            self._warmup_opt = None
            self.phase = "perlayer"
            self._init_region_opts()

        self._multi: MultiOptimizer | None = None if self.phase == "warmup" else self._make_multi()
        self._grad_ema: dict[torch.Tensor, torch.Tensor] = {}
        self._grad_sq_ema: dict[torch.Tensor, torch.Tensor] = {}
        self._loss_ema: float | None = None
        self._last_reward_time = time.perf_counter()
        # per-region gradient-norm EMA -> per-region bandit credit
        self._norm_beta = float(self.cfg.get("norm_ema_beta", 0.9))
        self._region_norm_ema: dict[str, float] = {}
        self._last_region_norm: dict[str, float] = {}
        self.selections: list[dict] = []

    # ---- construction helpers -----------------------------------------
    def _make_opt_for(self, region: str, geom: str) -> torch.optim.Optimizer:
        return build_optimizer(geom, self.geom_groups[region], lr=self.lrs[geom],
                               momentum=0.9, weight_decay=0.0)

    def _init_region_opts(self) -> None:
        for r in self.regions:
            self._region_opt[r] = self._make_opt_for(r, self._region_geom[r])

    def _make_multi(self) -> MultiOptimizer:
        opts = {f"region:{r}": self._region_opt[r] for r in self.regions}
        opts["onedim"] = self._onedim_opt
        return MultiOptimizer(opts)

    # ---- context features (gradient-only, no HVP) ---------------------
    def _stable_rank(self, region: str) -> float:
        p = max(self.geom_groups[region], key=lambda t: t.numel())
        g = p.grad
        if g is None:
            return 0.0
        G = g.detach().reshape(g.shape[0], -1).float()
        fro2 = float((G * G).sum())
        # spectral norm^2 via a few power iterations on G^T G
        v = torch.randn(G.shape[1], device=G.device)
        v /= v.norm() + 1e-12
        for _ in range(4):
            v = G.T @ (G @ v)
            v /= v.norm() + 1e-12
        spec2 = float(((G @ v) ** 2).sum())
        sr = fro2 / max(spec2, 1e-12)  # in [1, min(rows,cols)]
        return min(sr / min(G.shape), 1.0)  # normalize to ~[0,1]

    def _region_noise(self, region: str) -> float:
        vals = []
        for p in self.geom_groups[region]:
            ge, gse = self._grad_ema.get(p), self._grad_sq_ema.get(p)
            if ge is None or gse is None:
                continue
            vals.append(float((1 - (ge * ge) / (gse + 1e-12)).clamp(0, 1).mean()))
        return sum(vals) / len(vals) if vals else 0.5

    # ---- selection ----------------------------------------------------
    def _reselect(self, step: int, rewards: dict[str, float | None]) -> None:
        changed = []
        for r in self.regions:
            b = self._bandits[r]
            rw = rewards.get(r)
            if rw is not None:
                b.update(self._region_geom[r], rw)
            new_geom = b.select()
            if new_geom != self._region_geom[r]:
                old = self._region_opt[r]
                new = self._make_opt_for(r, new_geom)
                transfer_state(old, new, new_geom, mode=self.transfer_mode,
                               grad_sq_ema=self._grad_sq_ema)
                self._region_opt[r] = new
                self._region_geom[r] = new_geom
                changed.append((r, new_geom))
        if changed:
            self._multi = self._make_multi()
            for r, g in changed:
                self._record_switch(SwitchEvent(
                    step=step, epoch=self.epoch_of(step), from_name=f"{r}",
                    to_name=f"{r}:{g}", from_lr=None, to_lr=self.lrs[g],
                    reason=f"bandit_select|transfer:{self.transfer_mode}",
                ))
        self.selections.append({
            "step": step, "epoch": round(self.epoch_of(step), 3),
            "geoms": dict(self._region_geom),
            "rewards": {r: (round(v, 5) if v is not None else None) for r, v in rewards.items()},
        })

    def _seed_priors(self) -> None:
        for r in self.regions:
            prior = _context_prior(self._stable_rank(r), self._region_noise(r))
            b = self._bandits[r]
            for a, v in prior.items():
                b.value[a] += v

    # ---- lifecycle ----------------------------------------------------
    def begin_step(self, step: int) -> "MultiOptimizer | torch.optim.Optimizer":
        epoch = step // max(self.steps_per_epoch, 1)
        if self.phase == "warmup" and epoch >= self.warmup_epochs:
            self.phase = "perlayer"
            self._init_region_opts()
            # transfer warmup momentum into each region + the 1D AdamW
            for r in self.regions:
                transfer_state(self._warmup_opt, self._region_opt[r], self._region_geom[r],
                               mode=self.transfer_mode, grad_sq_ema=self._grad_sq_ema)
            transfer_state(self._warmup_opt, self._onedim_opt, "adamw",
                           mode=self.transfer_mode, grad_sq_ema=self._grad_sq_ema)
            self._seed_priors()
            self._multi = self._make_multi()
            self._record_switch(SwitchEvent(
                step=step, epoch=self.epoch_of(step), from_name="sgd_momentum(warmup)",
                to_name="perlayer", from_lr=self.warmup_lr, to_lr=0.0, reason="warmup_exit",
            ))
            return self._multi

        total = self.total_epochs * self.steps_per_epoch
        window_open = step < self.select_until_frac * total
        if self.phase == "perlayer" and window_open and step > 0 and step % self.select_every == 0:
            now = time.perf_counter()
            dt = max(now - self._last_reward_time, 1e-6)
            rewards: dict[str, float | None] = {}
            for r in self.regions:
                cur = self._region_norm_ema.get(r)
                prev = self._last_region_norm.get(r)
                # reward = how fast this region's gradient norm dropped (per s)
                rewards[r] = ((prev - cur) / dt) if (cur is not None and prev is not None) else None
                if cur is not None:
                    self._last_region_norm[r] = cur
            self._reselect(step, rewards)
            self._last_reward_time = now

        return self.active_optimizer

    def end_step(self, step: int, loss: float) -> None:
        be = float(self.cfg.get("loss_ema_beta", 0.9))
        self._loss_ema = loss if self._loss_ema is None else be * self._loss_ema + (1 - be) * loss
        b = self.grad_sq_beta
        with torch.no_grad():
            for p in self.model.parameters():
                if p.grad is None:
                    continue
                g = p.grad.detach()
                if p not in self._grad_sq_ema:
                    self._grad_sq_ema[p] = (g * g).clone()
                    self._grad_ema[p] = g.clone()
                else:
                    self._grad_sq_ema[p].mul_(b).add_(g * g, alpha=1 - b)
                    self._grad_ema[p].mul_(b).add_(g, alpha=1 - b)
            # per-region gradient-norm EMA (per-region bandit credit signal)
            nb = self._norm_beta
            for r in self.regions:
                sq = sum(float(p.grad.detach().pow(2).sum()) for p in self.geom_groups[r] if p.grad is not None)
                norm = sq**0.5
                if r not in self._region_norm_ema:
                    self._region_norm_ema[r] = norm
                else:
                    self._region_norm_ema[r] = nb * self._region_norm_ema[r] + (1 - nb) * norm

    @property
    def active_name(self) -> str:
        if self.phase == "warmup":
            return "sgd_momentum"
        return "|".join(f"{r}:{self._region_geom[r]}" for r in self.regions)

    @property
    def active_optimizer(self):
        if self.phase == "warmup":
            return self._warmup_opt
        return self._multi
