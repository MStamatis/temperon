"""Edge-of-Stability adaptive LR controller (arm H) — lr-setting logic."""

import torch
import torch.nn as nn

from eos_switch.controllers import build_controller


def _ctrl(**over):
    cfg = {
        "type": "edge_lr", "momentum": 0.9, "lr_init": 0.1, "lr_min": 1e-3,
        "lr_max": 0.5, "safety": 0.9, "check_every": 5, "warmup_steps": 10,
    }
    cfg.update(over)
    ctrl = build_controller(cfg)
    ctrl.setup(nn.Linear(4, 2), steps_per_epoch=10, total_epochs=3)
    return ctrl


def test_linear_lr_warmup():
    ctrl = _ctrl(warmup_steps=10)
    ctrl.begin_step(0)
    assert abs(ctrl.active_lr - 0.1 * 1 / 10) < 1e-9
    ctrl.begin_step(4)
    assert abs(ctrl.active_lr - 0.1 * 5 / 10) < 1e-9


def test_edge_lr_formula_and_clamp():
    ctrl = _ctrl(warmup_steps=0, check_every=5, safety=0.9, momentum=0.9)
    ctrl._data = object()  # truthy so the probe path runs
    ctrl._probe_lambda_max = lambda: 40.0  # constant curvature
    ctrl.begin_step(5)  # a check step
    # lr = 0.9 * (2 + 2*0.9) / 40 = 0.9 * 3.8 / 40 = 0.0855
    assert abs(ctrl.active_lr - 0.0855) < 1e-6


def test_edge_lr_clamps_to_max_when_curvature_tiny():
    ctrl = _ctrl(warmup_steps=0, lr_max=0.5)
    ctrl._data = object()
    ctrl._probe_lambda_max = lambda: 1e-4  # tiny curvature -> huge edge lr
    ctrl.begin_step(5)
    assert ctrl.active_lr == 0.5  # clamped to lr_max


def test_edge_lr_records_checks():
    ctrl = _ctrl(warmup_steps=0, check_every=5)
    ctrl._data = object()
    ctrl._probe_lambda_max = lambda: 38.0
    ctrl.begin_step(5)
    ctrl.begin_step(10)
    assert len(ctrl.checks) == 2
    assert "lambda_max" in ctrl.checks[0] and "lr" in ctrl.checks[0]


def test_annealed_edge_lr_decays_to_zero():
    # one-cycle-at-edge: peak set by edge, cosine envelope -> ~0 at the end
    ctrl = _ctrl(anneal="cosine", warmup_steps=0, check_every=5, lr_max=0.5)
    ctrl._data = object()
    ctrl._probe_lambda_max = lambda: 40.0  # edge peak = 0.0855
    ctrl.begin_step(5)               # early -> envelope ~1 -> near peak
    early = ctrl.active_lr
    total = ctrl.total_epochs * ctrl.steps_per_epoch  # 30
    ctrl.begin_step(total - 1)       # end -> envelope ~0
    late = ctrl.active_lr
    assert early > 0.07              # near the 0.0855 edge peak
    assert late < 0.02              # annealed toward 0
    assert late < early


def test_strong_sgd_fixed_controller_nesterov():
    cfg = {"type": "fixed", "optimizer": "sgd_momentum", "lr": 0.1, "momentum": 0.9,
           "nesterov": True, "weight_decay": 5e-4, "schedule": "cosine"}
    ctrl = build_controller(cfg)
    ctrl.setup(nn.Linear(4, 2), steps_per_epoch=10, total_epochs=2)
    opt = ctrl.begin_step(0)
    assert isinstance(opt, torch.optim.SGD)
    assert opt.param_groups[0]["nesterov"] is True
