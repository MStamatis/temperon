"""Layer-group partitioning and a multi-optimizer wrapper (Phase 4).

Parameters are split into layer-groups by their top-level module name; within
a group the >=2D weights get a bandit-chosen geometry, while ALL 1D params
(biases, norm scales) always go to a single shared AdamW (per the brief).
``MultiOptimizer`` presents the per-group optimizers to the training loop as
one optimizer (zero_grad/step delegate to all sub-optimizers).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def group_parameters(model: nn.Module) -> tuple[dict[str, list[torch.Tensor]], list[torch.Tensor]]:
    """Return ({region_name: [>=2D params]}, [all 1D params]).

    Region = first dotted component of the parameter name (e.g. conv1, layers,
    head). Regions with no >=2D params are omitted from the first dict.
    """
    geom_groups: dict[str, list[torch.Tensor]] = {}
    onedim: list[torch.Tensor] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2:
            region = name.split(".")[0]
            geom_groups.setdefault(region, []).append(p)
        else:
            onedim.append(p)
    return geom_groups, onedim


class MultiOptimizer:
    """Hold several optimizers (one per layer-group) and drive them together."""

    def __init__(self, opts: dict[str, torch.optim.Optimizer]) -> None:
        self.opts = opts

    def zero_grad(self, set_to_none: bool = True) -> None:
        for o in self.opts.values():
            o.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        for o in self.opts.values():
            o.step()

    @property
    def param_groups(self) -> list:
        return [g for o in self.opts.values() for g in o.param_groups]

    @property
    def state(self) -> dict:
        merged: dict = {}
        for o in self.opts.values():
            merged.update(o.state)
        return merged
