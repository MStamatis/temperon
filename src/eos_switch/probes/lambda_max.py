"""Largest Hessian eigenvalue via power iteration; optional Lanczos top-k."""

from __future__ import annotations

import torch
import torch.nn as nn

from eos_switch.probes.hvp import (
    HvpOperator,
    dot,
    flatten,
    preserve_bn_stats,
    unflatten,
)
from eos_switch.probes.precision import probe_precision


def _power_iteration(
    op: HvpOperator,
    iters: int,
    tol: float,
    generator: torch.Generator | None,
    apply_fn=None,
) -> tuple[float, list[torch.Tensor], int]:
    """Power iteration on H (or a wrapped operator). Assumes the dominant
    eigenvalue is the algebraically largest, which holds for the positive
    spectral edge of NN Hessians during training."""
    apply_fn = apply_fn or op.apply
    v = op.random_like_params(generator)
    n = dot(v, v).sqrt()
    v = [t / n for t in v]
    lam = 0.0
    used = 0
    for i in range(iters):
        hv = apply_fn(v)
        lam_new = float(dot(hv, v))
        hv_norm = float(dot(hv, hv).sqrt())
        used = i + 1
        if hv_norm == 0.0:
            lam = 0.0
            break
        v = [t / hv_norm for t in hv]
        if abs(lam_new - lam) <= tol * max(abs(lam_new), 1e-12):
            lam = lam_new
            break
        lam = lam_new
    return lam, v, used


def lambda_max(
    model: nn.Module,
    loss_fn,
    batch: tuple[torch.Tensor, torch.Tensor],
    iters: int = 20,
    tol: float = 1e-3,
    generator: torch.Generator | None = None,
    return_eigvec: bool = False,
) -> dict:
    """Top Hessian eigenvalue of the batch loss, fp32, autocast/TF32 off."""
    with probe_precision(), preserve_bn_stats(model):
        op = HvpOperator(model, loss_fn, batch)
        lam, v, used = _power_iteration(op, iters, tol, generator)
    out = {"lambda_max": lam, "power_iters": used, "probe_loss": op.loss}
    if return_eigvec:
        out["eigvec"] = v
    return out


def lanczos_topk(
    model: nn.Module,
    loss_fn,
    batch: tuple[torch.Tensor, torch.Tensor],
    k: int = 3,
    iters: int = 20,
    generator: torch.Generator | None = None,
) -> dict:
    """Top-k Hessian eigenvalues via Lanczos with full reorthogonalization.

    Memory: stores `iters` flat vectors of parameter size; intended for the
    small models used here (k <= 3, iters ~ 20).
    """
    k = min(k, 3)
    with probe_precision(), preserve_bn_stats(model):
        op = HvpOperator(model, loss_fn, batch)
        q = flatten(op.random_like_params(generator))
        q = q / q.norm()
        basis = [q]
        alphas: list[float] = []
        betas: list[float] = []
        for j in range(iters):
            hv = flatten(op.apply(unflatten(basis[j], op.params)))
            alpha = float(torch.dot(hv, basis[j]))
            alphas.append(alpha)
            w = hv - alpha * basis[j]
            if j > 0:
                w = w - betas[-1] * basis[j - 1]
            # Full reorthogonalization for numerical stability.
            for b in basis:
                w = w - torch.dot(w, b) * b
            beta = float(w.norm())
            if beta < 1e-10 or j == iters - 1:
                break
            betas.append(beta)
            basis.append(w / beta)
        t_mat = torch.diag(torch.tensor(alphas, dtype=torch.float64))
        for i, b in enumerate(betas[: len(alphas) - 1]):
            t_mat[i, i + 1] = b
            t_mat[i + 1, i] = b
        eigs = torch.linalg.eigvalsh(t_mat)
    top = sorted(eigs.tolist(), reverse=True)[:k]
    return {"eigenvalues": top, "lanczos_iters": len(alphas), "probe_loss": op.loss}
