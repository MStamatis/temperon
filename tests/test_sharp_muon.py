"""SharpMuon (arm M): single-pass sharpness-aware Muon, temporal + spectral."""

import torch
import torch.nn as nn

from eos_switch.controllers import build_controller
from eos_switch.optimizers.muon import Muon
from eos_switch.optimizers.sharp_muon import SharpMuon


def _model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(16, 12), nn.ReLU(), nn.Linear(12, 4))  # 2D weights + 1D biases


def _ctrl(mode, **over):
    cfg = {"type": "sharp_muon_catapult", "base_optimizer": "muon", "lr": 0.01,
           "min_lr": 0.0, "n_cycles": 4, "warmup_steps": 0, "sharp_mode": mode,
           "rho": 0.1}
    cfg.update(over)
    ctrl = build_controller(cfg)
    ctrl.setup(_model(), steps_per_epoch=10, total_epochs=8)
    return ctrl


def _one_step(opt, model):
    x, y = torch.randn(32, 16), torch.randint(0, 4, (32,))
    opt.zero_grad()
    nn.functional.cross_entropy(model(x), y).backward()
    opt.step()


def test_controller_builds_sharpmuon_single_pass():
    for mode in ("temporal", "spectral"):
        ctrl = _ctrl(mode)
        assert getattr(ctrl, "sam", False) is False          # loop runs single-pass
        assert ctrl.active_name == f"sharpmuon:{mode}"
        opt = ctrl.begin_step(5)
        assert isinstance(opt, SharpMuon)


def test_both_modes_step_finite_and_update():
    for mode in ("temporal", "spectral"):
        model = _model()
        ctrl = build_controller({"type": "sharp_muon_catapult", "lr": 0.02,
                                 "n_cycles": 4, "warmup_steps": 0, "sharp_mode": mode, "rho": 0.2})
        ctrl.setup(model, steps_per_epoch=10, total_epochs=8)
        w0 = [p.detach().clone() for p in model.parameters()]
        opt = ctrl.begin_step(5)
        _one_step(opt, model)               # first step (temporal: no g_prev yet)
        opt = ctrl.begin_step(6)
        _one_step(opt, model)               # second step exercises the correction
        assert all(torch.isfinite(p).all() for p in model.parameters()), mode
        assert any((p - q).norm() > 1e-6 for p, q in zip(model.parameters(), w0)), mode


def test_temporal_stores_prev_grad_and_corrects():
    torch.manual_seed(1)
    lin = nn.Linear(6, 3, bias=False)
    opt = SharpMuon(lin.parameters(), lr=0.0, momentum=0.0, mode="temporal", rho=0.5)
    x = torch.randn(8, 6)
    # step 1: no g_prev -> just stores it
    opt.zero_grad(); lin(x).pow(2).mean().backward(); opt.step()
    assert "g_prev" in opt.state[lin.weight]
    # lr=0 so weights unchanged; g_prev must equal the raw grad just seen
    assert torch.allclose(opt.state[lin.weight]["g_prev"], lin.weight.grad, atol=1e-6)


def test_spectral_rho_zero_equals_plain_muon():
    # With rho=0 the spectral perturbation is skipped -> identical to Muon.
    torch.manual_seed(2)
    x = torch.randn(16, 10)
    ma = nn.Linear(10, 8, bias=False)
    mb = nn.Linear(10, 8, bias=False)
    mb.load_state_dict(ma.state_dict())
    sm = SharpMuon(ma.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-3,
                   mode="spectral", rho=0.0)
    mu = Muon(mb.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-3)
    for _ in range(3):
        for m, o in ((ma, sm), (mb, mu)):
            o.zero_grad(); m(x).pow(2).mean().backward(); o.step()
    assert torch.allclose(ma.weight, mb.weight, atol=1e-5)


def test_invalid_mode_raises():
    try:
        SharpMuon(nn.Linear(4, 4).parameters(), mode="bogus")
    except ValueError:
        return
    raise AssertionError("expected ValueError for bad mode")
