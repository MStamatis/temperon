"""Muon optimizer + Newton-Schulz orthogonalization.

`newton_schulz_orthogonalize` is used both by the spectral-geometry sharpness
probe (Phase 1) and by the Muon optimizer (Phase 4). Muon orthogonalizes the
momentum of 2D parameters (conv/linear weights) via Newton-Schulz; non-2D
parameters (biases, norms) fall back to a plain momentum-SGD update, but the
per-layer controller routes 1D params to AdamW anyway.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def newton_schulz_orthogonalize(
    mat: torch.Tensor, steps: int = 5, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Approximate UV^T of the SVD of `mat` via the quintic Newton-Schulz
    iteration used by Muon (Jordan et al.). Input must be 2D. `dtype` selects the
    iteration precision: fp32 (default) is safest; bf16 matches the original Muon
    implementation and is ~free on tensor cores (speed hack, near-zero risk)."""
    if mat.ndim != 2:
        raise ValueError("newton_schulz_orthogonalize expects a 2D matrix")
    a, b, c = 3.4445, -4.7750, 2.0315
    x = mat.to(dtype)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        xxt = x @ x.T
        x = a * x + (b * xxt + c * (xxt @ xxt)) @ x
    if transposed:
        x = x.T
    return x


class Muon(torch.optim.Optimizer):
    """Muon: SGD-momentum whose update is orthogonalized (Newton-Schulz).

    For each parameter with >= 2 dims the (Nesterov) momentum is reshaped to a
    matrix (out, -1), orthogonalized, scaled by sqrt(max(1, rows/cols)), and
    applied. 1D parameters get a plain momentum-SGD update. Decoupled weight
    decay. Uses the ``momentum_buffer`` state key (compatible with the
    geometry-consistent state-transfer's spectral branch).
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.9,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
        ns_dtype: str = "fp32",
        min_numel: int = 0,
    ) -> None:
        # ns_dtype: Newton-Schulz iteration precision ("fp32" safe / "bf16" fast).
        # min_numel: 2D params with fewer elements than this get a plain momentum-
        # SGD update instead of Newton-Schulz (skip NS where it barely helps).
        nsd = {"fp32": torch.float32, "bf16": torch.bfloat16}.get(ns_dtype.lower())
        if nsd is None:
            raise ValueError(f"ns_dtype must be fp32/bf16, got {ns_dtype!r}")
        defaults = dict(
            lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay,
            ns_steps=ns_steps, ns_dtype=nsd, min_numel=int(min_numel),
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, mom = group["lr"], group["momentum"]
            nesterov, wd, ns = group["nesterov"], group["weight_decay"], group["ns_steps"]
            ns_dtype, min_numel = group["ns_dtype"], group["min_numel"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                buf = state.get("momentum_buffer")
                if buf is None:
                    buf = state["momentum_buffer"] = torch.zeros_like(p)
                buf.mul_(mom).add_(g)
                d = g.add(buf, alpha=mom) if nesterov else buf
                if wd != 0:
                    p.mul_(1 - lr * wd)  # decoupled weight decay
                if p.ndim >= 2 and p.numel() >= min_numel:
                    mat = d.reshape(p.shape[0], -1)
                    u = newton_schulz_orthogonalize(mat, ns, dtype=ns_dtype)
                    scale = max(1.0, mat.shape[0] / mat.shape[1]) ** 0.5
                    p.add_(u.reshape(p.shape).to(p.dtype), alpha=-lr * scale)
                else:
                    p.add_(d, alpha=-lr)  # 1D, or small 2D when min_numel skips NS
        return loss
