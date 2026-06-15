"""Sharpness measures per optimizer geometry, and the stability margin.

Quantities (all computed in fp32 inside probe_precision()):

- batch_sharpness: g^T H g / ||g||^2 on one mini-batch — the Edge of
  Stochastic Stability quantity that equilibrates at 2/eta for SGD.
- preconditioned_sharpness:
    * sgd geometry: raw lambda_max; threshold 2/eta (or (2+2b)/eta with
      momentum b);
    * adam geometry: lambda_max of P^{-1}H with P = diag(sqrt(v_hat)+eps)
      taken from optimizer state (power iteration on the similar symmetric
      operator P^{-1/2} H P^{-1/2}); empirical reference threshold ~38/eta
      for beta1=0.9 — NOT a closed-form bound;
    * spectral/sign geometry (Muon/Lion): directional curvature along the
      (orthogonalized) update direction. No closed-form EoS threshold exists
      in the literature for these updates; 2/eta is used as a reference
      value only and this approximation is itself part of the research.
- stability_margin: (threshold - sharpness)/threshold in (-inf, 1].
"""

from __future__ import annotations

import torch
import torch.nn as nn

from eos_switch.optimizers.pool import geometry_of, stability_threshold
from eos_switch.probes.hvp import HvpOperator, dot, preserve_bn_stats
from eos_switch.probes.lambda_max import _power_iteration
from eos_switch.probes.precision import probe_precision


def stability_margin(threshold: float, sharpness: float) -> float:
    return (threshold - sharpness) / threshold


def batch_sharpness(model: nn.Module, loss_fn, batch) -> dict:
    with probe_precision(), preserve_bn_stats(model):
        op = HvpOperator(model, loss_fn, batch)
        return _batch_sharpness_from_op(op)


def _batch_sharpness_from_op(op: HvpOperator) -> dict:
    g = op.grad_vector()
    hg = op.apply(g)
    gg = float(dot(g, g))
    ghg = float(dot(hg, g))
    return {
        "batch_sharpness": ghg / max(gg, 1e-30),
        "grad_norm": gg**0.5,
        "probe_loss": op.loss,
    }


def _adam_preconditioner(optimizer: torch.optim.Optimizer) -> list[torch.Tensor] | None:
    """diag(sqrt(v_hat)+eps) per param from Adam-family state, bias-corrected.

    Returns None when the state does not expose exp_avg_sq for every param
    (fresh optimizer, exotic implementation) — caller falls back to raw.
    """
    inv_sqrt: list[torch.Tensor] = []
    for group in optimizer.param_groups:
        betas = group.get("betas", (0.9, 0.999))
        beta2 = betas[1] if isinstance(betas, (tuple, list)) and len(betas) > 1 else 0.999
        eps = group.get("eps", 1e-8)
        for p in group["params"]:
            state = optimizer.state.get(p, {})
            v = state.get("exp_avg_sq")
            if v is None:
                return None
            step = state.get("step")
            if step is not None:
                step_val = float(step.item() if torch.is_tensor(step) else step)
                bias_corr = 1.0 - beta2 ** max(step_val, 1.0)
            else:
                bias_corr = 1.0
            v_hat = v.float() / bias_corr
            inv_sqrt.append(v_hat.sqrt().add(eps))
    return inv_sqrt


def _spectral_direction(
    optimizer: torch.optim.Optimizer, op: HvpOperator, opt_name: str
) -> list[torch.Tensor]:
    """Approximate update direction for spectral/sign geometries.

    Muon: Newton-Schulz orthogonalization of the momentum buffer (gradient
    if no state yet) for 2D params; raw vector for others.
    Lion: sign of the interpolated momentum (gradient fallback).
    """
    from eos_switch.optimizers.muon import newton_schulz_orthogonalize

    state_dirs: list[torch.Tensor] = []
    grads = op.grad_vector()
    by_param = {}
    for group in optimizer.param_groups:
        for p in group["params"]:
            st = optimizer.state.get(p, {})
            buf = st.get("momentum_buffer", st.get("exp_avg"))
            by_param[p] = buf
    for p, g in zip(op.params, grads):
        buf = by_param.get(p)
        d = buf.float() if buf is not None else g
        if opt_name == "lion":
            d = d.sign()
        elif d.ndim == 2:
            d = newton_schulz_orthogonalize(d)
        state_dirs.append(d)
    return state_dirs


def preconditioned_sharpness(
    optimizer: torch.optim.Optimizer,
    opt_name: str,
    model: nn.Module,
    loss_fn,
    batch,
    lr: float | None = None,
    iters: int = 15,
    tol: float = 1e-3,
    generator: torch.Generator | None = None,
) -> dict:
    with probe_precision(), preserve_bn_stats(model):
        op = HvpOperator(model, loss_fn, batch)
        return _preconditioned_sharpness_from_op(
            op, optimizer, opt_name, lr=lr, iters=iters, tol=tol, generator=generator
        )


def _preconditioned_sharpness_from_op(
    op: HvpOperator,
    optimizer: torch.optim.Optimizer,
    opt_name: str,
    lr: float | None,
    iters: int,
    tol: float,
    generator: torch.Generator | None,
) -> dict:
    if lr is None:
        lr = float(optimizer.param_groups[0]["lr"])
    momentum = float(optimizer.param_groups[0].get("momentum", 0.0) or 0.0)
    geometry = geometry_of(opt_name)
    fallback_raw = False

    if geometry == "adam":
        pre = _adam_preconditioner(optimizer)
        if pre is None:
            geometry = "sgd"  # fresh/exotic state: report raw curvature
            fallback_raw = True

    if geometry == "adam":
        inv_sqrt_p = [t.rsqrt() for t in pre]  # P^{-1/2}

        def apply_precond(vecs: list[torch.Tensor]) -> list[torch.Tensor]:
            half = [v * s for v, s in zip(vecs, inv_sqrt_p)]
            hv = op.apply(half)
            return [h * s for h, s in zip(hv, inv_sqrt_p)]

        sharp, _, used = _power_iteration(op, iters, tol, generator, apply_fn=apply_precond)
    elif geometry in ("spectral", "sign"):
        d = _spectral_direction(optimizer, op, opt_name)
        dd = float(dot(d, d))
        if dd == 0.0:
            sharp, used = 0.0, 0
        else:
            hd = op.apply(d)
            sharp = float(dot(hd, d)) / dd
            used = 1
    else:  # sgd geometry: raw lambda_max
        sharp, _, used = _power_iteration(op, iters, tol, generator)

    threshold = stability_threshold(opt_name, lr, momentum=momentum)
    return {
        "precond_sharpness": sharp,
        "threshold": threshold,
        "stability_margin": stability_margin(threshold, sharp),
        "geometry": geometry,
        "fallback_raw": fallback_raw,
        "hvp_iters": used,
        "lr": lr,
    }


def candidate_margin(
    op: HvpOperator,
    name: str,
    lr: float,
    momentum: float = 0.0,
    grad_sq_ema: dict | None = None,
    eps: float = 1e-8,
) -> dict:
    """Cheap (1 HVP) stability-margin estimate for a *candidate* optimizer.

    Curvature along the candidate's would-be update direction in its own
    geometry: adam -> preconditioned gradient g/(sqrt(v_est)+eps); sgd -> g;
    lion -> sign(g); muon -> Newton-Schulz(g). Used by the EoS controller to
    pick the next optimizer; must be called inside probe_precision().
    """
    from eos_switch.optimizers.muon import newton_schulz_orthogonalize

    geom = geometry_of(name)
    g = op.grad_vector()
    d: list[torch.Tensor] = []
    for p, gi in zip(op.params, g):
        if geom == "adam":
            v = grad_sq_ema.get(p) if grad_sq_ema else None
            denom = (v.sqrt() + eps) if v is not None else torch.ones_like(gi)
            d.append(gi / denom)
        elif geom == "sign":
            d.append(gi.sign())
        elif geom == "spectral":
            d.append(newton_schulz_orthogonalize(gi) if gi.ndim == 2 else gi)
        else:  # sgd
            d.append(gi)
    dd = float(dot(d, d))
    if dd == 0.0:
        sharp = 0.0
    else:
        hd = op.apply(d)
        sharp = float(dot(hd, d)) / dd
    threshold = stability_threshold(name, lr, momentum=momentum)
    return {
        "name": name,
        "lr": lr,
        "directional_sharpness": sharp,
        "threshold": threshold,
        "margin": stability_margin(threshold, sharp),
    }


def full_probe(
    model: nn.Module,
    loss_fn,
    batch,
    optimizer: torch.optim.Optimizer,
    opt_name: str,
    lr: float | None = None,
    iters: int = 15,
    generator: torch.Generator | None = None,
    include_precond: bool = True,
) -> dict:
    """batch_sharpness + (optionally) preconditioned_sharpness on ONE graph.

    batch_sharpness costs a single HVP; the preconditioned lambda_max costs
    `iters` HVPs. The ProbeScheduler therefore measures batch_sharpness at
    every probe and amortizes the preconditioned estimate via its
    `precond_every` knob (always included in switch bursts).
    """
    with probe_precision(), preserve_bn_stats(model):
        op = HvpOperator(model, loss_fn, batch)
        out = _batch_sharpness_from_op(op)
        if include_precond:
            out.update(
                _preconditioned_sharpness_from_op(
                    op, optimizer, opt_name, lr=lr, iters=iters, tol=1e-3, generator=generator
                )
            )
    return out
