"""Probe correctness on quadratic losses with analytically known curvature.

All tests run on CPU in float64 (tolerance: <5% as per project spec; most
results are far tighter because the quadratic Hessian is exact).
"""

import pytest
import torch
import torch.nn as nn

from eos_switch.probes import (
    batch_sharpness,
    lambda_max,
    lanczos_topk,
    preconditioned_sharpness,
    stability_margin,
)
from eos_switch.probes.sharpness import full_probe

DIM = 12


class QuadraticModel(nn.Module):
    """L(theta) = 0.5 theta^T A theta -- Hessian is exactly A everywhere."""

    def __init__(self, theta0: torch.Tensor) -> None:
        super().__init__()
        self.theta = nn.Parameter(theta0.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # input ignored
        return self.theta


def make_spd(dim: int = DIM, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(dim, dim, dtype=torch.float64, generator=gen))
    eigs = torch.linspace(1.0, 10.0, dim, dtype=torch.float64)
    a = q @ torch.diag(eigs) @ q.T
    return a, eigs


def quad_loss(a: torch.Tensor):
    def loss_fn(out, y):
        return 0.5 * out @ (a @ out)

    return loss_fn


def setup_quadratic(seed: int = 0):
    a, eigs = make_spd(seed=seed)
    gen = torch.Generator().manual_seed(seed + 1)
    theta0 = torch.randn(DIM, dtype=torch.float64, generator=gen)
    model = QuadraticModel(theta0)
    batch = (torch.zeros(1), torch.zeros(1))
    return model, quad_loss(a), batch, a, eigs


def test_lambda_max_within_tolerance():
    model, loss_fn, batch, _, eigs = setup_quadratic()
    gen = torch.Generator().manual_seed(3)
    out = lambda_max(model, loss_fn, batch, iters=100, tol=1e-10, generator=gen)
    rel_err = abs(out["lambda_max"] - eigs[-1].item()) / eigs[-1].item()
    assert rel_err < 0.05, (out["lambda_max"], eigs[-1].item())


def test_lanczos_topk_matches_spectrum():
    model, loss_fn, batch, _, eigs = setup_quadratic()
    gen = torch.Generator().manual_seed(4)
    out = lanczos_topk(model, loss_fn, batch, k=3, iters=DIM, generator=gen)
    expected = sorted(eigs.tolist(), reverse=True)[:3]
    for got, exp in zip(out["eigenvalues"], expected):
        assert abs(got - exp) / exp < 0.05, (out["eigenvalues"], expected)


def test_batch_sharpness_analytic():
    model, loss_fn, batch, a, _ = setup_quadratic()
    theta = model.theta.detach()
    g = a @ theta
    expected = float(g @ (a @ g) / (g @ g))
    out = batch_sharpness(model, loss_fn, batch)
    assert abs(out["batch_sharpness"] - expected) / abs(expected) < 0.05
    assert abs(out["grad_norm"] - float(g.norm())) / float(g.norm()) < 1e-6


def test_preconditioned_sharpness_sgd_is_raw_lambda_max():
    model, loss_fn, batch, _, eigs = setup_quadratic()
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    gen = torch.Generator().manual_seed(5)
    out = preconditioned_sharpness(
        opt, "sgd_momentum", model, loss_fn, batch, iters=100, tol=1e-10, generator=gen
    )
    assert abs(out["precond_sharpness"] - eigs[-1].item()) / eigs[-1].item() < 0.05
    # (2 + 2*0.9) / 0.1 = 38
    assert out["threshold"] == pytest.approx(38.0)
    assert out["stability_margin"] == pytest.approx(
        (38.0 - out["precond_sharpness"]) / 38.0
    )


def test_preconditioned_sharpness_adam_matches_dense_eig():
    model, loss_fn, batch, a, _ = setup_quadratic(seed=7)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    # Populate real Adam state with a few steps on the quadratic.
    for _ in range(5):
        opt.zero_grad()
        loss = loss_fn(model(batch[0]), batch[1])
        loss.backward()
        opt.step()

    # Rebuild the preconditioner exactly as the probe defines it.
    p = model.theta
    state = opt.state[p]
    beta2 = opt.param_groups[0]["betas"][1]
    eps = opt.param_groups[0]["eps"]
    step = float(state["step"].item() if torch.is_tensor(state["step"]) else state["step"])
    v_hat = state["exp_avg_sq"].double() / (1.0 - beta2**step)
    pre = v_hat.sqrt() + eps
    inv_half = torch.diag(pre.rsqrt())
    expected = torch.linalg.eigvalsh(inv_half @ a @ inv_half).max().item()

    gen = torch.Generator().manual_seed(8)
    out = preconditioned_sharpness(
        opt, "adam", model, loss_fn, batch, iters=200, tol=1e-12, generator=gen
    )
    assert not out["fallback_raw"]
    assert abs(out["precond_sharpness"] - expected) / abs(expected) < 0.05
    assert out["threshold"] == pytest.approx(38.0 / 1e-3)


def test_preconditioned_sharpness_fresh_adam_falls_back_to_raw():
    model, loss_fn, batch, _, eigs = setup_quadratic()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)  # no steps -> no state
    gen = torch.Generator().manual_seed(9)
    out = preconditioned_sharpness(
        opt, "adam", model, loss_fn, batch, iters=100, tol=1e-10, generator=gen
    )
    assert out["fallback_raw"] is True
    assert abs(out["precond_sharpness"] - eigs[-1].item()) / eigs[-1].item() < 0.05


def test_full_probe_consistent_with_individual_probes():
    model, loss_fn, batch, _, _ = setup_quadratic()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    gen = torch.Generator().manual_seed(10)
    combined = full_probe(model, loss_fn, batch, opt, "sgd", iters=100, generator=gen)
    separate = batch_sharpness(model, loss_fn, batch)
    assert combined["batch_sharpness"] == pytest.approx(
        separate["batch_sharpness"], rel=1e-9
    )
    assert combined["threshold"] == pytest.approx(2.0 / 0.1)


def test_stability_margin_formula():
    assert stability_margin(20.0, 0.0) == pytest.approx(1.0)
    assert stability_margin(20.0, 10.0) == pytest.approx(0.5)
    assert stability_margin(20.0, 40.0) == pytest.approx(-1.0)
