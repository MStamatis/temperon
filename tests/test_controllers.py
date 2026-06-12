import math

import torch
import torch.nn as nn

from eos_switch.controllers import build_controller


def _model() -> nn.Module:
    return nn.Linear(4, 2)


def test_fixed_controller_constant_lr():
    ctrl = build_controller({"type": "fixed", "optimizer": "adamw", "lr": 1e-3})
    ctrl.setup(_model(), steps_per_epoch=10, total_epochs=2)
    opt = ctrl.begin_step(0)
    assert isinstance(opt, torch.optim.AdamW)
    ctrl.begin_step(19)
    assert math.isclose(ctrl.active_lr, 1e-3)
    assert ctrl.switch_events == []


def test_fixed_controller_cosine_schedule():
    ctrl = build_controller(
        {"type": "fixed", "optimizer": "sgd_momentum", "lr": 0.1, "schedule": "cosine"}
    )
    ctrl.setup(_model(), steps_per_epoch=100, total_epochs=1)
    ctrl.begin_step(0)
    assert math.isclose(ctrl.active_lr, 0.1, rel_tol=1e-6)
    ctrl.begin_step(50)
    assert math.isclose(ctrl.active_lr, 0.05, rel_tol=1e-6)
    ctrl.begin_step(100)
    assert ctrl.active_lr < 1e-6


def test_sequential_controller_switches_with_fresh_state():
    cfg = {
        "type": "sequential",
        "phases": [
            {"optimizer": "sgd_momentum", "lr": 0.1, "epochs": 1},
            {"optimizer": "adamw", "lr": 1e-3},
        ],
    }
    ctrl = build_controller(cfg)
    model = _model()
    ctrl.setup(model, steps_per_epoch=5, total_epochs=3)

    opt0 = ctrl.begin_step(0)
    assert isinstance(opt0, torch.optim.SGD)
    # Give the SGD optimizer some momentum state.
    out = model(torch.randn(3, 4)).sum()
    out.backward()
    opt0.step()

    opt1 = ctrl.begin_step(5)  # phase boundary: 1 epoch x 5 steps
    assert isinstance(opt1, torch.optim.AdamW)
    assert ctrl.active_name == "adamw"
    assert len(ctrl.switch_events) == 1
    ev = ctrl.switch_events[0]
    assert ev.step == 5 and ev.from_name == "sgd_momentum" and ev.to_name == "adamw"
    # Fresh state: AdamW has no step history yet.
    assert all(len(opt1.state[p]) == 0 for group in opt1.param_groups for p in group["params"])
