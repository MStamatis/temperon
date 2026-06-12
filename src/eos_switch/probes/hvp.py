"""Hessian-vector products via double backward, plus probe-safety helpers."""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm


def trainable_params(model: nn.Module) -> list[torch.Tensor]:
    return [p for p in model.parameters() if p.requires_grad]


def dot(a: list[torch.Tensor], b: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack([(x * y).sum() for x, y in zip(a, b)]).sum()


def norm(a: list[torch.Tensor]) -> torch.Tensor:
    return dot(a, a).sqrt()


def flatten(tensors: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.reshape(-1) for t in tensors])


def unflatten(flat: torch.Tensor, like: list[torch.Tensor]) -> list[torch.Tensor]:
    out = []
    offset = 0
    for t in like:
        out.append(flat[offset : offset + t.numel()].view_as(t))
        offset += t.numel()
    return out


@contextmanager
def preserve_bn_stats(model: nn.Module):
    """Snapshot/restore BatchNorm running stats around a probe forward.

    Probes evaluate the model in train mode (the same landscape the optimizer
    sees) but must not pollute the running statistics used by evaluation.
    """
    saved = []
    for m in model.modules():
        if isinstance(m, _BatchNorm) and m.track_running_stats and m.running_mean is not None:
            saved.append(
                (m, m.running_mean.clone(), m.running_var.clone(), m.num_batches_tracked.clone())
            )
    try:
        yield
    finally:
        with torch.no_grad():
            for m, mean, var, count in saved:
                m.running_mean.copy_(mean)
                m.running_var.copy_(var)
                m.num_batches_tracked.copy_(count)


class HvpOperator:
    """Reusable HVP for a fixed (model, loss_fn, batch).

    The gradient graph is built once with create_graph=True; each apply()
    then costs a single extra backward. Always construct INSIDE
    probe_precision() (the constructor runs the forward pass).
    """

    def __init__(self, model: nn.Module, loss_fn, batch: tuple[torch.Tensor, torch.Tensor]) -> None:
        self.params = trainable_params(model)
        x, y = batch
        loss = loss_fn(model(x), y)
        self.grads = list(torch.autograd.grad(loss, self.params, create_graph=True))
        self.loss = float(loss.detach())

    def apply(self, vecs: list[torch.Tensor]) -> list[torch.Tensor]:
        s = dot(self.grads, vecs)
        return [
            h.detach()
            for h in torch.autograd.grad(s, self.params, retain_graph=True)
        ]

    def grad_vector(self) -> list[torch.Tensor]:
        return [g.detach() for g in self.grads]

    def random_like_params(self, generator: torch.Generator | None = None) -> list[torch.Tensor]:
        return [
            torch.randn(p.shape, device=p.device, dtype=p.dtype, generator=generator)
            for p in self.params
        ]
