"""State-transfer correctness across optimizer geometries (CPU)."""

import torch
import torch.nn as nn

from eos_switch.optimizers.pool import build_optimizer
from eos_switch.optimizers.state_transfer import transfer_state


def _model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(6, 6), nn.ReLU(), nn.Linear(6, 3))


def _warm(opt, model, n=3):
    for _ in range(n):
        opt.zero_grad()
        model(torch.randn(8, 6)).square().mean().backward()
        opt.step()


def _grad_sq_ema(model):
    model(torch.randn(8, 6)).square().mean().backward()
    ema = {p: (p.grad.detach() ** 2).clone() for p in model.parameters()}
    model.zero_grad()
    return ema


def test_geometry_sgd_to_adam_transfers_m_and_fresh_v():
    model = _model()
    sgd = build_optimizer("sgd_momentum", model.parameters(), lr=0.1)
    _warm(sgd, model)
    adamw = build_optimizer("adamw", model.parameters(), lr=1e-3)
    ema = _grad_sq_ema(model)

    info = transfer_state(sgd, adamw, "adamw", mode="geometry", grad_sq_ema=ema, adam_warm_steps=500)
    assert info["transferred"] > 0
    p = next(iter(model.parameters()))
    st = adamw.state[p]
    m = sgd.state[p]["momentum_buffer"]
    # first moment carried from SGD momentum_buffer
    assert torch.allclose(st["exp_avg"], m)
    # second moment is the fresh grad^2 estimate, floored at m^2
    assert torch.allclose(st["exp_avg_sq"], torch.maximum(ema[p], m * m))
    assert float(st["step"]) == 500.0


def test_naive_mode_halves_first_moment():
    model = _model()
    sgd = build_optimizer("sgd_momentum", model.parameters(), lr=0.1)
    _warm(sgd, model)
    adamw = build_optimizer("adamw", model.parameters(), lr=1e-3)

    transfer_state(sgd, adamw, "adamw", mode="naive", grad_sq_ema=_grad_sq_ema(model))
    p = next(iter(model.parameters()))
    assert torch.allclose(adamw.state[p]["exp_avg"], 0.5 * sgd.state[p]["momentum_buffer"])


def test_adam_v_floored_at_m_squared_when_grads_tiny():
    # Late-training case: tiny grad^2 EMA. v must be floored at m^2 so the
    # first post-switch Adam step stays bounded (no 1/sqrt(tiny) explosion).
    model = _model()
    sgd = build_optimizer("sgd_momentum", model.parameters(), lr=0.1)
    _warm(sgd, model)
    adamw = build_optimizer("adamw", model.parameters(), lr=1e-3)
    tiny = {p: torch.full_like(p, 1e-12) for p in model.parameters()}
    transfer_state(sgd, adamw, "adamw", mode="geometry", grad_sq_ema=tiny)
    p = next(iter(model.parameters()))
    m = sgd.state[p]["momentum_buffer"]
    assert (adamw.state[p]["exp_avg_sq"] >= m * m - 1e-9).all()


def test_none_mode_is_cold():
    model = _model()
    sgd = build_optimizer("sgd_momentum", model.parameters(), lr=0.1)
    _warm(sgd, model)
    adamw = build_optimizer("adamw", model.parameters(), lr=1e-3)
    transfer_state(sgd, adamw, "adamw", mode="none")
    assert all(len(adamw.state.get(p, {})) == 0 for p in model.parameters())


def test_adam_to_sgd_transfers_first_moment():
    model = _model()
    adamw = build_optimizer("adamw", model.parameters(), lr=1e-3)
    _warm(adamw, model)
    sgd = build_optimizer("sgd_momentum", model.parameters(), lr=0.1)
    transfer_state(adamw, sgd, "sgd_momentum", mode="geometry")
    p = next(iter(model.parameters()))
    assert torch.allclose(sgd.state[p]["momentum_buffer"], adamw.state[p]["exp_avg"])


def test_transferred_state_allows_step():
    # A switched optimizer must be able to take a step without error.
    model = _model()
    sgd = build_optimizer("sgd_momentum", model.parameters(), lr=0.1)
    _warm(sgd, model)
    adamw = build_optimizer("adamw", model.parameters(), lr=1e-3)
    transfer_state(adamw if False else sgd, adamw, "adamw", grad_sq_ema=_grad_sq_ema(model))
    adamw.zero_grad()
    model(torch.randn(8, 6)).square().mean().backward()
    adamw.step()  # must not raise
    assert torch.isfinite(next(model.parameters())).all()
