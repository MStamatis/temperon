"""EoS-switch controller logic (CPU, no probes needed for the decision tests)."""

import torch
import torch.nn as nn

from eos_switch.controllers import build_controller

POOL = [
    {"optimizer": "adamw", "lr": 1.0e-3, "weight_decay": 0.01},
    {"optimizer": "lion", "lr": 3.0e-4},
    {"optimizer": "sgd_momentum", "lr": 0.1, "momentum": 0.9},
]


def _ctrl(**over):
    cfg = {"type": "eos_switch", "pool": POOL, "check_every": 10, "k": 3, "margin_floor": 0.1}
    cfg.update(over)
    ctrl = build_controller(cfg)
    ctrl.setup(nn.Linear(6, 3), steps_per_epoch=5, total_epochs=4)
    return ctrl


def test_register_active_margin_triggers_after_k_lows():
    ctrl = _ctrl()
    assert ctrl._register_active_margin(0.5) is False  # healthy, resets
    assert ctrl._register_active_margin(0.05) is False  # 1
    assert ctrl._register_active_margin(0.05) is False  # 2
    assert ctrl._register_active_margin(0.05) is True   # 3 -> switch
    # a healthy reading resets the counter
    assert ctrl._register_active_margin(0.2) is False
    assert ctrl._low_margin_count == 0


def test_select_next_picks_max_lr_times_margin():
    ctrl = _ctrl(warmup_epochs=0)
    # adamw lr1e-3*margin0.9=9e-4; lion 3e-4*0.9=2.7e-4; sgd 0.1*0.005=5e-4
    margins = {"adamw": 0.9, "lion": 0.9, "sgd_momentum": 0.005}
    assert ctrl._select_next(margins, exclude=None) == "adamw"
    # if adamw excluded, sgd (5e-4) beats lion (2.7e-4)
    assert ctrl._select_next(margins, exclude="adamw") == "sgd_momentum"


def test_select_next_tiebreak_least_recently_used():
    ctrl = _ctrl(warmup_epochs=0)
    margins = {"adamw": 1.0, "lion": 1.0 * (1e-3 / 3e-4), "sgd_momentum": -10.0}
    # make adamw and lion tie on lr*margin, then prefer the LRU one
    margins = {"adamw": 1.0, "lion": 1e-3 / 3e-4, "sgd_momentum": -10.0}
    ctrl._last_used = {"adamw": 100, "lion": 10}  # lion less recently used
    assert ctrl._select_next(margins, exclude=None) == "lion"


def test_warmup_then_transition_records_switch():
    ctrl = _ctrl(warmup_epochs=1)  # 1 epoch * 5 steps = warmup boundary at step 5
    assert ctrl.phase == "warmup"
    opt0 = ctrl.begin_step(0)
    assert isinstance(opt0, torch.optim.SGD)
    assert ctrl.active_name == "sgd_momentum"
    # cross the boundary (no probe context -> uniform margins)
    ctrl.begin_step(5)
    assert ctrl.phase == "eos"
    assert len(ctrl.switch_events) == 1
    assert ctrl.switch_events[0].reason.startswith("warmup_exit")


def test_window_closes_after_fraction():
    ctrl = _ctrl(warmup_epochs=0, switch_until_frac=0.5)
    # total = steps_per_epoch(5) * total_epochs(4) = 20; window closes at step 10
    assert ctrl._window_open(0) is True
    assert ctrl._window_open(9) is True
    assert ctrl._window_open(10) is False
    assert ctrl._window_open(15) is False


def test_grad_sq_ema_updates_on_end_step():
    ctrl = _ctrl(warmup_epochs=0)
    model = ctrl.model
    opt = ctrl.begin_step(0)
    opt.zero_grad()
    model(torch.randn(8, 6)).square().mean().backward()
    opt.step()
    ctrl.end_step(0, loss=1.0)
    assert any(p in ctrl._grad_sq_ema for p in model.parameters())
