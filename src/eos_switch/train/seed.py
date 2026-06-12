"""Seeding utilities: full control over python/numpy/torch RNG state."""

from __future__ import annotations

import random

import numpy as np
import torch

# Default experiment seeds (project convention; do not change between arms).
DEFAULT_SEEDS = [42, 1181241943, 958682846]


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def make_generator(seed: int, device: torch.device | str = "cpu") -> torch.Generator:
    gen = torch.Generator(device=torch.device(device).type)
    gen.manual_seed(seed)
    return gen
