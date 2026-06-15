"""Geometry-consistent optimizer state transfer on a switch.

When the EoS controller switches optimizer i -> j, we move the *first moment*
(an EMA of the gradient, comparable across families) into j's geometry and
re-estimate j's *second moment* fresh from recent grad^2 (never carrying a
stale ``v`` from epochs ago). Modes:

- ``geometry`` (default): reproject the first moment to j's geometry
    * adam-family  : exp_avg <- m,  exp_avg_sq <- recent grad^2 EMA (fresh)
    * sgd_momentum : momentum_buffer <- m
    * lion (sign)  : exp_avg <- m
    * muon (spectral): momentum_buffer <- Newton-Schulz(m)   (2D params)
- ``naive``: the paper's baseline -- keep the first moment scaled by 0.5,
  still re-estimate the second moment.
- ``none``: fresh state (cold restart).

The Adam warm-start sets ``step`` to ``adam_warm_steps`` so the bias
corrections do not blow up the seeded moments on the first step (research
heuristic; compared against ``naive`` as an ablation).
"""

from __future__ import annotations

import torch

from eos_switch.optimizers.muon import newton_schulz_orthogonalize
from eos_switch.optimizers.pool import geometry_of


def _first_moment(opt: torch.optim.Optimizer, p: torch.Tensor) -> torch.Tensor | None:
    """The first-moment buffer of `p` in `opt` (exp_avg or momentum_buffer)."""
    st = opt.state.get(p, {})
    for key in ("exp_avg", "momentum_buffer"):
        buf = st.get(key)
        if buf is not None:
            return buf
    return None


def _params(opt: torch.optim.Optimizer) -> list[torch.Tensor]:
    return [p for group in opt.param_groups for p in group["params"]]


def _seed_state(
    new_opt: torch.optim.Optimizer,
    p: torch.Tensor,
    geom: str,
    m: torch.Tensor,
    grad_sq_ema: dict | None,
    adam_warm_steps: int,
) -> None:
    st = new_opt.state[p]
    if geom == "adam":
        st["step"] = torch.tensor(float(adam_warm_steps))
        st["exp_avg"] = m.clone()
        v = grad_sq_ema.get(p) if grad_sq_ema else None
        st["exp_avg_sq"] = v.detach().clone() if v is not None else torch.zeros_like(p)
        # NAdam keeps an extra running product of the momentum schedule; seed it
        # so NAdam.step() does not KeyError. Harmless for Adam/AdamW (ignored).
        st["mu_product"] = torch.tensor(1.0)
    elif geom == "sign":  # Lion
        st["exp_avg"] = m.clone()
    else:  # sgd / spectral both use momentum_buffer
        st["momentum_buffer"] = m.clone()


def transfer_state(
    old_opt: torch.optim.Optimizer,
    new_opt: torch.optim.Optimizer,
    new_name: str,
    mode: str = "geometry",
    grad_sq_ema: dict | None = None,
    adam_warm_steps: int = 1000,
) -> dict:
    """Move state from `old_opt` into `new_opt`. Returns a small info dict.

    `new_opt`'s per-parameter state is overwritten for params that had a
    first moment in `old_opt`; others are left fresh.
    """
    if mode not in ("geometry", "naive", "none"):
        raise ValueError(f"unknown state-transfer mode {mode!r}")
    info = {"mode": mode, "transferred": 0, "orthogonalized": 0}
    if mode == "none":
        for p in _params(new_opt):
            new_opt.state.pop(p, None)
        return info

    geom = geometry_of(new_name)
    moments = {p: _first_moment(old_opt, p) for p in _params(old_opt)}
    for p in _params(new_opt):
        m = moments.get(p)
        new_opt.state.pop(p, None)  # drop any stale state first
        if m is None:
            continue
        m = m.detach().clone()
        if mode == "naive":
            m = m * 0.5
        elif geom == "spectral" and m.ndim == 2:
            m = newton_schulz_orthogonalize(m)
            info["orthogonalized"] += 1
        _seed_state(new_opt, p, geom, m, grad_sq_ema, adam_warm_steps)
        info["transferred"] += 1
    return info
