"""Numerical-precision guarantee for curvature probes.

Research-critical invariant: every Hessian-vector product, power iteration
and sharpness estimate runs in full float32 (or wider), with autocast AND
TF32 disabled, regardless of what the surrounding training loop does.
Accuracy of sharpness estimates takes precedence over speed.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch


@contextmanager
def probe_precision():
    """Context manager: fp32 math only — autocast off, TF32 off.

    Saves and restores the global TF32 flags, so it is safe to nest inside
    a bf16-autocast training loop that keeps TF32 enabled.
    """
    prev_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        with (
            torch.autocast("cuda", enabled=False),
            torch.autocast("cpu", enabled=False),
        ):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul_tf32
        torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32
