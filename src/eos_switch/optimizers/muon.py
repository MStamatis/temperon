"""Muon-style Newton-Schulz orthogonalization (full optimizer lands in Phase 4).

Phase 1 only needs `newton_schulz_orthogonalize` for the spectral-geometry
sharpness probe direction.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def newton_schulz_orthogonalize(mat: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate UV^T of the SVD of `mat` via the quintic Newton-Schulz
    iteration used by Muon (Jordan et al.). Input must be 2D."""
    if mat.ndim != 2:
        raise ValueError("newton_schulz_orthogonalize expects a 2D matrix")
    a, b, c = 3.4445, -4.7750, 2.0315
    x = mat.float()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        xxt = x @ x.T
        x = a * x + (b * xxt + c * (xxt @ xxt)) @ x
    if transposed:
        x = x.T
    return x
