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
        # Alias the base optimizer's groups/state: LR scheduling and zero_grad
        # operate on the real parameters; e_w perturbations live alongside the
        # base optimizer's own per-parameter state.
        self.param_groups = base_optimizer.param_groups
        self.state = base_optimizer.state

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
        scale = self.rho / (self._grad_norm() + self.eps)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        """Restore the original weights, then base-step with the perturbed grad."""
        for group in self.param_groups:
            for p in group["params"]:
                e_w = self.state[p].get("e_w") if p in self.state else None
                if e_w is not None:
                    p.sub_(e_w)
                    self.state[p]["e_w"] = None
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad(set_to_none=True)

    def step(self, closure=None):  # pragma: no cover - guarded usage error
        raise RuntimeError(
            "SAM requires the two-pass first_step()/second_step(); the training "
            "loop drives both passes when controller.sam is True."
        )
