"""Learning-rate schedules that give the SAM tail a full anneal to own.

Temperon's efficacy depends on *what the tail covers*, not just how long it is:
in our ablations a tail bolted onto the middle of an existing cycle gained
nothing, while a tail owning a complete anneal matched full-time SAM. These
helpers make the alignment explicit -- pass the same `decay_frac` to `wsd()`
and as `tail_frac` to `Temperon`, and the tail starts exactly where decay does.
"""

from __future__ import annotations

from typing import Callable


def wsd(total_steps: int, warmup_steps: int, decay_frac: float,
        min_frac: float = 0.0) -> Callable[[int], float]:
    """Warmup-Stable-Decay multiplier, for `torch.optim.lr_scheduler.LambdaLR`.

        sched = LambdaLR(optimizer, wsd(total, warmup=250, decay_frac=0.3))
        opt = Temperon(optimizer, total_steps=total, tail_frac=0.3)

    Returns a callable mapping step -> multiplier in [min_frac, 1].
    """
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    if not 0.0 <= decay_frac <= 1.0:
        raise ValueError(f"decay_frac must be in [0, 1], got {decay_frac}")
    decay_start = int(round((1.0 - decay_frac) * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        if step < decay_start:
            return 1.0
        span = max(1, total_steps - decay_start)
        frac = min(1.0, (step - decay_start) / span)
        return 1.0 - (1.0 - min_frac) * frac

    return lr_lambda


def cosine_tail(total_steps: int, tail_frac: float, warmup_steps: int = 0,
                min_frac: float = 0.0) -> Callable[[int], float]:
    """Constant during the cheap phase, then a fresh cosine anneal over the
    tail -- the shape used by the vision experiments, where the tail optimizer
    re-enters at full learning rate and anneals to `min_frac`."""
    if not 0.0 < tail_frac <= 1.0:
        raise ValueError(f"tail_frac must be in (0, 1], got {tail_frac}")
    import math

    tail_start = int(round((1.0 - tail_frac) * total_steps))

    def lr_lambda(step: int) -> float:
        if step < tail_start:
            return 1.0
        local = step - tail_start
        if local < warmup_steps:
            return (local + 1) / max(1, warmup_steps)
        span = max(1, total_steps - tail_start - warmup_steps)
        frac = min(1.0, (local - warmup_steps) / span)
        return min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * frac))

    return lr_lambda
