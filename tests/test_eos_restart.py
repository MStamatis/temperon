"""EoS-triggered warm-restart controller (arm K)."""

import torch.nn as nn

from eos_switch.controllers import build_controller


def _ctrl(**over):
    cfg = {"type": "eos_restart", "lr": 0.1, "min_lr": 0.0, "momentum": 0.9,
           "warmup_steps": 0, "check_every": 10, "restart_ratio": 0.1,
           "cycle_epochs": 5, "min_dwell_checks": 2, "switch_until_frac": 0.8}
    cfg.update(over)
    ctrl = build_controller(cfg)
    ctrl.setup(nn.Linear(4, 2), steps_per_epoch=10, total_epochs=10)  # 100 steps
    return ctrl


def test_warmup_ramp():
    ctrl = _ctrl(warmup_steps=10)
    ctrl.begin_step(0)
    assert abs(ctrl.active_lr - 0.1 * 1 / 10) < 1e-9


def test_cosine_decays_within_cycle():
    ctrl = _ctrl(warmup_steps=0, cycle_epochs=10)  # cycle length 100 steps
    ctrl.begin_step(0)
    hi = ctrl.active_lr
    ctrl.begin_step(50)  # mid -> ~0.05
    assert hi > ctrl.active_lr and abs(ctrl.active_lr - 0.05) < 0.02


def test_eos_restart_fires_when_edge_low_after_dwell():
    ctrl = _ctrl()
    ctrl._data = object()
    ctrl._probe_edge_ratio = lambda: (0.05, 1.0)  # deep inside stability
    ctrl.begin_step(10)  # check 1: dwell<2 -> no restart
    ctrl.begin_step(20)  # check 2: dwell<2 -> no restart
    assert ctrl.switch_events == []
    ctrl.begin_step(30)  # check 3: dwell>=2 + edge<0.1 -> RESTART
    assert len(ctrl.switch_events) == 1
    assert "eos_restart" in ctrl.switch_events[0].reason
    assert ctrl._cycle_start == 30


def test_no_restart_when_edge_high():
    ctrl = _ctrl()
    ctrl._data = object()
    ctrl._probe_edge_ratio = lambda: (0.5, 1.0)  # far from settling
    for s in (10, 20, 30, 40):
        ctrl.begin_step(s)
    assert ctrl.switch_events == []


def test_no_restart_after_window_close():
    ctrl = _ctrl()
    ctrl._data = object()
    ctrl._probe_edge_ratio = lambda: (0.0, 1.0)
    ctrl.begin_step(90)  # 90 > 0.8*100 -> final settle starts (lr at peak)
    lr90 = ctrl.active_lr
    ctrl.begin_step(99)  # later in the final cosine
    assert ctrl.switch_events == []   # no restarts after the window closes
    assert ctrl.active_lr < lr90      # final cosine is annealing
