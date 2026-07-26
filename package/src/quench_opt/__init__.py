"""Quench -- pay for sharpness-aware minimization only where it pays you back."""

from quench_opt.quench import Quench
from quench_opt.schedules import cosine_tail, wsd

__all__ = ["Quench", "wsd", "cosine_tail"]
__version__ = "0.1.0.dev0"
