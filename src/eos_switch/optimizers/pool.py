"""Optimizer construction plus per-geometry Edge-of-Stability thresholds.

Stability thresholds (on the *preconditioned* sharpness, see probes/):
- plain SGD:          2 / lr
- SGD + momentum b:   (2 + 2b) / lr
- Adam family (b1=0.9): ~38 / lr  -- empirical reference value from the
  adaptive-EoS literature (Cohen et al.), NOT a closed-form bound.
- Muon / spectral:    2 / lr used as a *reference only*; no closed-form EoS
  threshold is known for orthogonalized updates (research question here).
"""

from __future__ import annotations

import torch

# Maps optimizer name -> curvature geometry used by the probes. Includes the
# names appearing in the upstream OptiRoulette default pool.
_GEOMETRY = {
    "sgd": "sgd",
    "sgd_momentum": "sgd",
    "adam": "adam",
    "adamw": "adam",
    "nadam": "adam",
    "radam": "adam",
    "amsgrad": "adam",
    "adan": "adam",
    "ranger": "adam",
    "ranger21": "adam",
    "adabelief": "adam",
    "yogi": "adam",
    "adamp": "adam",
    "qhadam": "adam",
    "lion": "sign",
    "muon": "spectral",
}


def geometry_of(name: str) -> str:
    # Unknown adaptive methods default to adam geometry; the probe falls back
    # to raw curvature anyway when no exp_avg_sq state is found.
    return _GEOMETRY.get(name.lower(), "adam")


def build_optimizer(
    name: str,
    params,
    lr: float,
    momentum: float = 0.9,
    betas: tuple[float, float] = (0.9, 0.999),
    weight_decay: float = 0.0,
    nesterov: bool = False,
) -> torch.optim.Optimizer:
    name = name.lower()
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd_momentum":
        return torch.optim.SGD(
            params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov
        )
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=betas, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)
    if name == "lion":
        from pytorch_optimizer import Lion

        return Lion(params, lr=lr, weight_decay=weight_decay)
    if name == "muon":
        from eos_switch.optimizers.muon import Muon  # Phase 4

        return Muon(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer {name!r}")


def stability_threshold(name: str, lr: float, momentum: float = 0.0) -> float:
    """Stability limit on the preconditioned sharpness for the given optimizer.

    For sgd the momentum actually carried by the optimizer must be passed in:
    plain SGD -> 2/lr, momentum b -> (2+2b)/lr.
    """
    name = name.lower()
    geometry = geometry_of(name)
    if geometry == "sgd":
        return (2.0 + 2.0 * momentum) / lr
    if geometry == "adam":
        return 38.0 / lr  # empirical reference (beta1=0.9), not closed-form
    if geometry in ("spectral", "sign"):
        # Reference value only; see module docstring.
        return 2.0 / lr
    raise ValueError(f"unknown geometry {geometry!r} for optimizer {name!r}")
