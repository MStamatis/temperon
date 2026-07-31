"""Temperon -- pay for sharpness-aware minimization only where it pays you back."""

from temperon.temperon import Temperon
from temperon.schedules import cosine_tail, wsd

__all__ = ["Temperon", "wsd", "cosine_tail"]
__version__ = "0.1.0"
