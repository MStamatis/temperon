"""SharpMuon: single-pass sharpness-aware Muon (arm M -- the novelty).

SAM lifts the Muon-catapult final by ~+3pp (0.795 -> 0.826 in the 2x2 ablation)
but costs a SECOND forward-backward (the perturbed-point gradient). SharpMuon
recovers a sharpness-aware signal WITHOUT that second pass, two ways:

  mode="temporal":  finite-difference HVP along the optimizer's own trajectory.
      g - g_prev  ~=  H * dw   (dw = the last weight step), a FREE Hessian-vector
      product. We blend it into the gradient,  g_tilde = g + rho*(g - g_prev),
      before momentum + Newton-Schulz. One extra buffer (g_prev); single pass.

  mode="spectral":  Muon-native. Newton-Schulz already works on the gradient
      matrix; two warm-started power iterations extract its dominant singular
      direction sigma*u v^T, and we perturb  mat -> mat + rho*sigma*u v^T  before
      orthogonalizing. rho's SIGN is part of the experiment (emphasize vs avoid
      the dominant gradient mode). Theoretically loose -- the gradient spectrum
      is not the loss Hessian -- but at the Edge of Stability they align; this is
      the project's "EoS in spectral geometry" idea made into an update rule.

mode="off" reduces to plain Muon. Non-2D params (biases/norms) get a plain
(temporal-corrected) momentum-SGD update, as in Muon. Decoupled weight decay.
The controller drives this single-pass (no SAM two-pass in the loop).
"""

from __future__ import annotations

import torch

from eos_switch.optimizers.muon import newton_schulz_orthogonalize


class SharpMuon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 0.01,
        momentum: float = 0.9,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
        mode: str = "temporal",
        rho: float = 0.05,
        spec_iters: int = 2,
        eps: float = 1e-12,
    ) -> None:
        if mode not in ("temporal", "spectral", "off"):
            raise ValueError(f"mode must be temporal/spectral/off, got {mode!r}")
        defaults = dict(
            lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay,
            ns_steps=ns_steps, mode=mode, rho=rho, spec_iters=spec_iters, eps=eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def _spectral_perturb(self, mat, state, iters, rho, eps):
        # Dominant singular direction of `mat` via warm-started power iteration on
        # mat^T mat; reuse the right vector across steps so 1-2 iters suffice.
        cols = mat.shape[1]
        v = state.get("spec_v")
        if v is None or v.shape[0] != cols:
            v = torch.randn(cols, device=mat.device, dtype=mat.dtype)
        v = v / (v.norm() + eps)
        u = sigma = None
        for _ in range(max(1, iters)):
            u = mat @ v
            u = u / (u.norm() + eps)
            v = mat.t() @ u
            sigma = v.norm()
            v = v / (sigma + eps)
        state["spec_v"] = v.detach()
        top = sigma * torch.outer(u, v)        # rank-1 dominant mode (sigma * u v^T)
        return mat + rho * top

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, mom, nesterov = group["lr"], group["momentum"], group["nesterov"]
            wd, ns, eps = group["weight_decay"], group["ns_steps"], group["eps"]
            mode, rho, si = group["mode"], group["rho"], group["spec_iters"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]

                # --- temporal sharpness correction (single-pass HVP proxy) ---
                if mode == "temporal" and rho != 0:
                    gp = state.get("g_prev")
                    if gp is not None:
                        g = g + rho * (g - gp)
                    state["g_prev"] = p.grad.detach().clone()

                buf = state.get("momentum_buffer")
                if buf is None:
                    buf = state["momentum_buffer"] = torch.zeros_like(p)
                buf.mul_(mom).add_(g)
                d = g.add(buf, alpha=mom) if nesterov else buf

                if wd != 0:
                    p.mul_(1 - lr * wd)  # decoupled weight decay

                if p.ndim >= 2:
                    mat = d.reshape(p.shape[0], -1).float()
                    if mode == "spectral" and rho != 0:
                        mat = self._spectral_perturb(mat, state, si, rho, eps)
                    u = newton_schulz_orthogonalize(mat, ns)
                    scale = max(1.0, mat.shape[0] / mat.shape[1]) ** 0.5
                    p.add_(u.reshape(p.shape).to(p.dtype), alpha=-lr * scale)
                else:
                    p.add_(d, alpha=-lr)
        return loss
