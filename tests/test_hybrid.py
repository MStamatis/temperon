"""Adam->SGD hybrid controller (arm I)."""

import torch
import torch.nn as nn

from eos_switch.controllers import build_controller


def _ctrl(**over):
    cfg = {
        "type": "adam_sgd_hybrid",
        "adam": {"optimizer": "adamw", "lr": 1e-3, "weight_decay": 0.01},
        "sgd": {"lr": 0.1, "momentum": 0.9, "nesterov": True, "weight_decay": 5e-4},
        "handoff_frac": 0.3, "plateau_window": 2, "plateau_delta": 0.01,
    }
    cfg.update(over)
    ctrl = build_controller(cfg)
    ctrl.setup(nn.Linear(8, 4), steps_per_epoch=10, total_epochs=10)
    return ctrl


def _train(ctrl, model, start, end, loss_fn):
    for s in range(start, end):
        opt = ctrl.begin_step(s)
        opt.zero_grad(set_to_none=True)
        model(torch.randn(8, 8)).square().mean().backward()
        opt.step()
        ctrl.end_step(s, loss=loss_fn(s))


def test_starts_on_adam():
    ctrl = _ctrl()
    opt = ctrl.begin_step(0)
    assert isinstance(opt, torch.optim.AdamW)
    assert ctrl.phase == "adam"


def test_handoff_by_fraction_cap():
    # constant loss -> no plateau before cap; cap = 0.3*10 = epoch 3 (step 30)
    ctrl = _ctrl(handoff_frac=0.3, plateau_delta=-1.0)  # plateau_delta<0 disables plateau
    model = ctrl.model
    _train(ctrl, model, 0, 30, lambda s: 1.0)
    assert ctrl.phase == "adam"
    ctrl.begin_step(30)  # epoch 3 >= cap
    assert ctrl.phase == "sgd"
    assert isinstance(ctrl.active_optimizer, torch.optim.SGD)
    assert len(ctrl.switch_events) == 1
    assert "handoff" in ctrl.switch_events[0].reason


def test_handoff_transfers_adam_moment_to_sgd():
    ctrl = _ctrl(handoff_frac=0.2)
    model = ctrl.model
    _train(ctrl, model, 0, 20, lambda s: 1.0)
    ctrl.begin_step(20)  # epoch 2 >= 0.2*10
    assert ctrl.phase == "sgd"
    p = next(iter(model.parameters()))
    # SGD momentum buffer seeded from Adam's exp_avg
    assert "momentum_buffer" in ctrl.active_optimizer.state[p]


def test_sgd_phase_cosine_decays():
    ctrl = _ctrl(handoff_frac=0.1)
    model = ctrl.model
    _train(ctrl, model, 0, 10, lambda s: 1.0)
    ctrl.begin_step(10)  # handoff at step 10
    lr_start = ctrl.active_lr
    ctrl.begin_step(99)  # near end (total=100)
    lr_end = ctrl.active_lr
    assert lr_start > 0.05 and lr_end < lr_start
