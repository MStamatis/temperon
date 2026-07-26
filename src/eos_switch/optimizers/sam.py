"""Sharpness-Aware Minimization wrapper (arm L).

Two-pass update (Foret et al. 2021, "Sharpness-Aware Minimization"):

  1. compute the gradient g at the current weights w;
  2. perturb to the local worst case  w + e,  e = rho * g / ||g||  (ascent
     along the gradient, fixed radius rho), and recompute the gradient there;
  3. restore w, then let the BASE optimizer step from the ORIGINAL weights
     using that perturbed-point gradient.

This biases training toward flat minima -- the implicit goal of EoS catapults
made explicit. The wrapper is base-optimizer-agnostic; `param_groups` and
`state` alias the base optimizer so an external LR scheduler (the cyclic-
catapult schedule) writes straight through to the real parameter groups.

The training loop drives the two passes: it calls first_step() between the two
forward/backward passes and second_step() after the second backward (see
train/loop.py). Calling step() directly is a usage error.
"""

from __future__ import annotations

import torch


class SAM:
    def __init__(self, base_optimizer: torch.optim.Optimizer, rho: float = 0.05, eps: float = 1e-12):
        if rho < 0:
            raise ValueError(f"rho must be >= 0, got {rho}")
        self.base_optimizer = base_optimizer
        self.rho = float(rho)
        self.eps = float(eps)
        # Free same-batch curvature byproduct (arm N): the two SAM gradients give
        # a noise-free Hessian-vector product. second_step() fills last_sharpness
        # with g_hat^T H g_hat = g^T H g / ||g||^2 (the batch_sharpness / EoS
        # quantity) at zero extra cost. None until the first second_step().
        self.last_sharpness: float | None = None
        self._gnorm = 0.0
        # Alias the base optimizer's groups/state: LR scheduling and zero_grad
        # operate on the real parameters.
        self.param_groups = base_optimizer.param_groups
        self.state = base_optimizer.state
        # The perturbation is kept in SAM's OWN dict, never in the base
        # optimizer's state. torch.optim.Adam/AdamW decide whether to lazily
        # initialize exp_avg/exp_avg_sq with `if len(state) == 0`, so writing
        # e_w there before the base optimizer's first step makes it skip its
        # own initialization and blow up with KeyError: 'exp_avg' (SGD and Muon
        # read their buffers with .get() and were unaffected, which is why this
        # only ever surfaced with an Adam-family base).
        self._e_w: dict[torch.Tensor, torch.Tensor] = {}

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def _grad_norm(self) -> torch.Tensor:
        # Global L2 norm over all parameter gradients (classic SAM geometry).
        device = self.param_groups[0]["params"][0].device
        return torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(device)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2,
        )

    @torch.no_grad()
    def first_step(self, zero_grad: bool = True) -> None:
        """Climb to the local worst case w + rho * g / ||g|| and stash e_w."""
        gnorm = self._grad_norm()
        self._gnorm = float(gnorm)  # ||g||, kept for the free-sharpness estimate
        scale = self.rho / (gnorm + self.eps)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)
                self._e_w[p] = e_w
        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def record_sharpness(self) -> None:
        """Record the FREE same-batch sharpness from the two gradients. Call AFTER
        the second backward and BEFORE any grad clipping -- clipping g' biases the
        estimate negative (the clipped g' shrinks while ||g|| does not). Reads the
        current grad g' and the stashed e_w; leaves weights/grads/e_w untouched.

        With e_w = rho * g/||g|| and g' the perturbed-point gradient,
            <e_w, g'> = rho * g_hat^T g'
            g_hat^T H g_hat ~= g_hat^T (g' - g) / rho = (<e_w,g'>/rho - ||g||)/rho
        so last_sharpness = (<e_w, g'> - rho*||g||) / rho^2 (exact for quadratics).
        """
        dot = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                e_w = self._e_w.get(p)
                if e_w is not None and p.grad is not None:
                    dot += float(torch.sum(e_w * p.grad))
        rho = self.rho
        self.last_sharpness = (dot - rho * self._gnorm) / (rho * rho + self.eps) if rho > 0 else None

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        """Restore the original weights, then base-step with the perturbed grad."""
        for group in self.param_groups:
            for p in group["params"]:
                e_w = self._e_w.pop(p, None)
                if e_w is not None:
                    p.sub_(e_w)
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad(set_to_none=True)

    def step(self, closure=None):  # pragma: no cover - guarded usage error
        raise RuntimeError(
            "SAM requires the two-pass first_step()/second_step(); the training "
            "loop drives both passes when controller.sam is True."
        )
