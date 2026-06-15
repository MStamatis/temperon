"""Phase 4: Muon optimizer, layer grouping, multi-optimizer, UCB, controller."""

import torch
import torch.nn as nn

from eos_switch.controllers import build_controller
from eos_switch.controllers.per_layer import GEOMS, UCBBandit
from eos_switch.optimizers.layer_groups import MultiOptimizer, group_parameters
from eos_switch.optimizers.muon import Muon, newton_schulz_orthogonalize
from eos_switch.train.models import build_model


def test_newton_schulz_near_orthogonal():
    torch.manual_seed(0)
    G = torch.randn(16, 32)
    U = newton_schulz_orthogonalize(G, steps=6)
    # the quintic Newton-Schulz drives all singular values into a band around
    # 1 (~[0.68, 1.13]) -- semi-orthogonal, as Muon intends.
    sv = torch.linalg.svdvals(U)
    assert (sv > 0.5).all() and (sv < 1.5).all(), sv


def test_muon_step_runs_and_finite():
    torch.manual_seed(0)
    w = nn.Parameter(torch.randn(8, 12))
    opt = Muon([w], lr=0.02)
    for _ in range(3):
        opt.zero_grad()
        (w.square().sum()).backward()
        opt.step()
    assert torch.isfinite(w).all()
    assert "momentum_buffer" in opt.state[w]


def test_muon_handles_conv4d_and_1d():
    conv = nn.Parameter(torch.randn(6, 3, 3, 3))  # 4D
    bias = nn.Parameter(torch.randn(6))           # 1D
    opt = Muon([conv, bias], lr=0.02)
    opt.zero_grad()
    (conv.square().sum() + bias.square().sum()).backward()
    opt.step()
    assert torch.isfinite(conv).all() and torch.isfinite(bias).all()


def test_group_parameters_splits_2d_and_1d():
    model = build_model("smallcnn", 10)
    geom, onedim = group_parameters(model)
    assert "features" in geom and "head" in geom
    assert all(p.ndim >= 2 for ps in geom.values() for p in ps)
    assert all(p.ndim < 2 for p in onedim)


def test_multi_optimizer_delegates():
    a = nn.Parameter(torch.randn(4, 4))
    b = nn.Parameter(torch.randn(4, 4))
    oa = torch.optim.SGD([a], lr=0.1)
    ob = torch.optim.AdamW([b], lr=0.1)
    multi = MultiOptimizer({"a": oa, "b": ob})
    (a.sum() + b.sum()).backward()
    multi.step()
    multi.zero_grad()
    assert a.grad is None and b.grad is None
    assert len(multi.param_groups) == 2


def test_ucb_tries_each_arm_once_then_exploits():
    b = UCBBandit(["x", "y", "z"], c=0.0)  # c=0 -> pure exploitation after init
    seen = {b.select() for _ in range(1)}
    b.update("x", 1.0)
    b.update("y", 0.0)
    b.update("z", 0.0)
    # all tried once now; with c=0, best mean (x) is chosen
    assert b.select() == "x"


def _train(ctrl, model, start, end):
    for s in range(start, end):
        opt = ctrl.begin_step(s)
        opt.zero_grad(set_to_none=True)
        model(torch.randn(8, 3, 32, 32)).square().mean().backward()
        opt.step()
        ctrl.end_step(s, loss=1.0 / (s + 1))


def test_perlayer_controller_transitions_and_selects():
    cfg = {
        "type": "perlayer", "warmup_epochs": 1, "select_every": 5,
        "geom_lrs": {"muon": 0.02, "adamw": 1e-3, "lion": 3e-4, "sgd_momentum": 0.1},
    }
    ctrl = build_controller(cfg)
    model = build_model("smallcnn", 10)
    ctrl.setup(model, steps_per_epoch=10, total_epochs=3)
    assert ctrl.phase == "warmup"
    _train(ctrl, model, 0, 11)   # steps 0-10: step 10 (epoch 1) triggers transition
    assert ctrl.phase == "perlayer"
    _train(ctrl, model, 11, 31)  # several selection rounds
    assert len(ctrl.selections) >= 1
    assert all(g in GEOMS for g in ctrl._region_geom.values())
    assert torch.isfinite(next(model.parameters())).all()
