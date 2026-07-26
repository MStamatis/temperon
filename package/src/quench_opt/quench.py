"""Quench: allocate the sharpness-aware budget to the final anneal.

Sharpness-Aware Minimization doubles the cost of every step. The measured
finding behind this package is that it only earns that cost near the end of
training: running SAM for a contiguous tail aligned to the learning-rate decay
reaches full-time-SAM quality for roughly a third less wall-clock, while
spreading the same budget uniformly over training does *worse* than either.

`Quench` wraps an ordinary optimizer and drives SAM's two passes itself, so the
allocation is a scheduling decision rather than a change to your model code.
"""

from __future__ import annotations

from typing import Callable, Iterable, Literal

import torch

Transfer = Literal["momentum", "none"]


class Quench:
    """Run `optimizer` normally, then hand the final `tail_frac` of training to
    SAM (optionally switching to a second optimizer at the same boundary).

    Args:
        optimizer: the optimizer for the cheap phase; also the tail optimizer
            unless `tail_optimizer` is given.
        total_steps: total number of `step()` calls planned for the run. The
            tail boundary is derived from it, so it must be the real total.
        tail_frac: fraction of training the SAM tail owns. **Align this with
            your learning-rate decay** -- the tail should own a full anneal.
            With a warmup-stable-decay schedule, set it equal to the decay
            fraction (see `quench_opt.schedules.wsd`).
        rho: SAM neighbourhood radius.
        rho_ramp_steps: linearly ramp rho from 0 over this many steps after the
            tail opens, to soften the switch. 0 disables the ramp.
        tail_optimizer: optional second optimizer to switch to when the tail
            opens (the cheap-explorer / expensive-refiner hand-off).
        transfer: what to carry across that switch. "momentum" copies
            `momentum_buffer` for parameters that have one; "none" starts the
            tail optimizer cold.

    The closure follows the `torch.optim.LBFGS` convention: it must zero the
    gradients, compute the loss, call `backward()`, and return the loss. Quench
    calls it once per step in the cheap phase and twice in the tail.

        opt = Quench(torch.optim.AdamW(model.parameters(), lr=2e-5),
                     total_steps=len(loader) * epochs, tail_frac=0.3)

        for batch in loader:
            def closure():
                opt.zero_grad()
                loss = loss_fn(model(batch.x), batch.y)
                loss.backward()
                return loss
            loss = opt.step(closure)

    Any gradient clipping belongs inside the closure, before `return`.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        tail_frac: float = 0.3,
        rho: float = 0.05,
        rho_ramp_steps: int = 0,
        tail_optimizer: torch.optim.Optimizer | None = None,
        transfer: Transfer = "momentum",
        eps: float = 1e-12,
    ) -> None:
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")
        if not 0.0 <= tail_frac <= 1.0:
            raise ValueError(f"tail_frac must be in [0, 1], got {tail_frac}")
        if rho < 0:
            raise ValueError(f"rho must be >= 0, got {rho}")
        if transfer not in ("momentum", "none"):
            raise ValueError(f"transfer must be momentum/none, got {transfer!r}")

        self._opt = optimizer
        self._tail_opt = tail_optimizer
        self.total_steps = int(total_steps)
        self.tail_frac = float(tail_frac)
        self.rho = float(rho)
        self.rho_ramp_steps = int(rho_ramp_steps)
        self.transfer: Transfer = transfer
        self.eps = float(eps)

        # First step index owned by the SAM tail. tail_frac=0 means SAM from
        # the start; tail_frac=1 means never (a no-SAM baseline arm).
        self.tail_start = int(round((1.0 - self.tail_frac) * self.total_steps))
        self.step_count = 0
        self._switched = False
        # The perturbation lives here, never in the wrapped optimizer's state:
        # torch.optim.Adam decides whether to initialize its own slots with
        # `if len(state) == 0`, so polluting that dict before its first step
        # makes it skip initialization and raise KeyError: 'exp_avg'.
        self._e_w: dict[torch.Tensor, torch.Tensor] = {}

    # --- introspection --------------------------------------------------------

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        """The optimizer currently in use (the tail one after the switch)."""
        return self._tail_opt if self._switched else self._opt

    @property
    def param_groups(self) -> list[dict]:
        return self.optimizer.param_groups

    def sam_active(self, step: int | None = None) -> bool:
        step = self.step_count if step is None else step
        return self.tail_frac > 0.0 and step >= self.tail_start

    def current_rho(self, step: int | None = None) -> float:
        step = self.step_count if step is None else step
        if not self.sam_active(step):
            return 0.0
        if self.rho_ramp_steps <= 0:
            return self.rho
        done = step - self.tail_start + 1
        return self.rho * min(1.0, done / self.rho_ramp_steps)

    # --- the step -------------------------------------------------------------

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure: Callable[[], torch.Tensor]) -> torch.Tensor:
        """One optimization step. Returns the loss at the *unperturbed* weights."""
        if closure is None:
            raise ValueError("Quench requires a closure; see the class docstring.")
        if self.sam_active() and not self._switched and self._tail_opt is not None:
            self._hand_off()

        loss = closure()
        if not self.sam_active():
            self.optimizer.step()
        else:
            self._ascend(self.current_rho())
            closure()               # gradients at the perturbed point
            self._restore()         # back to the original weights
            self.optimizer.step()   # ... and step with the perturbed gradient
        self.step_count += 1
        return loss

    # --- internals ------------------------------------------------------------

    def _params(self) -> Iterable[torch.Tensor]:
        for group in self.param_groups:
            for p in group["params"]:
                yield p

    @torch.no_grad()
    def _grad_norm(self) -> torch.Tensor:
        grads = [p.grad.norm(p=2) for p in self._params() if p.grad is not None]
        if not grads:
            return torch.tensor(0.0)
        return torch.norm(torch.stack(grads), p=2)

    @torch.no_grad()
    def _ascend(self, rho: float) -> None:
        scale = rho / (self._grad_norm() + self.eps)
        for p in self._params():
            if p.grad is None:
                continue
            e_w = p.grad * scale.to(p)
            p.add_(e_w)
            self._e_w[p] = e_w

    @torch.no_grad()
    def _restore(self) -> None:
        for p in self._params():
            e_w = self._e_w.pop(p, None)
            if e_w is not None:
                p.sub_(e_w)

    @torch.no_grad()
    def _hand_off(self) -> int:
        """Switch to the tail optimizer, optionally carrying momentum over.

        Only parameters the tail optimizer actually owns receive a buffer, and
        only if the cheap optimizer had one for them. A raw copy is exact here:
        SGD's and Muon's `momentum_buffer` both accumulate a discounted sum of
        gradients, and there is no second moment to mis-seed.
        """
        n = 0
        if self.transfer == "momentum":
            old_params = {p for g in self._opt.param_groups for p in g["params"]}
            for group in self._tail_opt.param_groups:
                for p in group["params"]:
                    if p not in old_params:
                        continue
                    buf = self._opt.state.get(p, {}).get("momentum_buffer")
                    if buf is not None:
                        self._tail_opt.state[p]["momentum_buffer"] = buf.detach().clone()
                        n += 1
        self._switched = True
        return n

    # --- checkpointing --------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "step_count": self.step_count,
            "switched": self._switched,
            "optimizer": self._opt.state_dict(),
            "tail_optimizer": (self._tail_opt.state_dict()
                               if self._tail_opt is not None else None),
        }

    def load_state_dict(self, state: dict) -> None:
        self.step_count = int(state["step_count"])
        self._switched = bool(state["switched"])
        self._opt.load_state_dict(state["optimizer"])
        if self._tail_opt is not None and state.get("tail_optimizer") is not None:
            self._tail_opt.load_state_dict(state["tail_optimizer"])

    def __repr__(self) -> str:
        return (f"Quench(tail_frac={self.tail_frac}, rho={self.rho}, "
                f"tail_start={self.tail_start}/{self.total_steps}, "
                f"step={self.step_count}, sam={'on' if self.sam_active() else 'off'})")
