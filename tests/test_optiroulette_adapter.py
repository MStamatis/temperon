"""Adapter/controller tests against the real upstream optiroulette package.

These run on CPU with a tiny model. They intentionally exercise the upstream
default profile (arm D semantics) and the no-warmup variant (arm E).
"""

import torch
import torch.nn as nn

from eos_switch.controllers import build_controller


def _model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Linear(8, 4)


def _train_steps(ctrl, model, steps):
    for step in range(steps):
        opt = ctrl.begin_step(step)
        opt.zero_grad(set_to_none=True)
        loss = model(torch.randn(4, 8)).square().mean()
        loss.backward()
        opt.step()


def test_warmup_locks_to_sgd_then_roulette():
    ctrl = build_controller({"type": "optiroulette", "warmup": True, "warmup_epochs": 2, "seed": 0})
    model = _model()
    ctrl.setup(model, steps_per_epoch=3, total_epochs=6)

    _train_steps(ctrl, model, 6)  # epochs 0-1: warmup
    assert ctrl.adapter.phase == "warmup"
    assert ctrl.adapter.active_name == "sgd"

    _train_steps(ctrl, model, 12)  # epochs 2+: roulette
    assert ctrl.adapter.phase == "roulette"
    # drop_after_warmup=true in the default profile: sgd must be gone.
    assert ctrl.adapter.active_name != "sgd"
    assert len(ctrl.switch_events) >= 1


def test_no_warmup_switches_from_epoch_zero():
    ctrl = build_controller({"type": "optiroulette", "warmup": False, "seed": 0})
    model = _model()
    ctrl.setup(model, steps_per_epoch=2, total_epochs=4)
    assert ctrl.adapter.phase == "roulette"
    _train_steps(ctrl, model, 8)
    # Epoch-granularity switching with avoid_repeat: 4 epochs -> >=1 switch.
    assert len(ctrl.switch_events) >= 1


def test_switch_events_have_real_optimizer_names():
    ctrl = build_controller({"type": "optiroulette", "warmup": False, "seed": 1})
    model = _model()
    ctrl.setup(model, steps_per_epoch=2, total_epochs=5)
    _train_steps(ctrl, model, 10)
    for ev in ctrl.switch_events:
        assert ev.to_name in ctrl.adapter.opt.optimizers
        assert ev.reason.startswith("optiroulette:")


def test_active_optimizer_is_underlying_not_wrapper():
    from optiroulette import OptiRoulette

    ctrl = build_controller({"type": "optiroulette", "warmup": False, "seed": 0})
    model = _model()
    ctrl.setup(model, steps_per_epoch=2, total_epochs=2)
    ctrl.begin_step(0)
    assert not isinstance(ctrl.active_optimizer, OptiRoulette)
    assert isinstance(ctrl.active_optimizer, torch.optim.Optimizer)
