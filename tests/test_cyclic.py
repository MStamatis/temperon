"""Cyclic-catapult controller (arm J): SGDR schedule + pool-per-restart."""

import math

import torch
import torch.nn as nn

from eos_switch.controllers import build_controller


def _ctrl(**over):
    cfg = {"type": "cyclic_catapult", "base_optimizer": "sgd_momentum",
           "lr": 0.1, "min_lr": 0.0, "n_cycles": 4}
    cfg.update(over)
    ctrl = build_controller(cfg)
    ctrl.setup(nn.Linear(8, 4), steps_per_epoch=10, total_epochs=8)  # 80 steps, 4 cycles of 20
    return ctrl


def test_lr_restarts_each_cycle():
    ctrl = _ctrl()
    # cycle length = 80/4 = 20 steps; peak at the start of each cycle
    ctrl.begin_step(0)
    assert abs(ctrl.active_lr - 0.1) < 1e-6           # cycle 0 start -> peak
    ctrl.begin_step(19)
    near_min = ctrl.active_lr
    ctrl.begin_step(20)
    assert ctrl.active_lr > near_min + 0.04            # restart -> catapult back up
    assert abs(ctrl.active_lr - 0.1) < 1e-3


def test_cosine_decay_within_cycle():
    ctrl = _ctrl()
    ctrl.begin_step(0)
    hi = ctrl.active_lr
    ctrl.begin_step(10)  # mid cycle (frac 0.5) -> ~0.05
    mid = ctrl.active_lr
    assert hi > mid and abs(mid - 0.05) < 0.02


def test_base_optimizer_adamw():
    ctrl = _ctrl(base_optimizer="adamw", lr=1e-3, momentum=0.0)
    opt = ctrl.begin_step(0)
    assert isinstance(opt, torch.optim.AdamW)
    assert abs(ctrl.active_lr - 1e-3) < 1e-9


def test_pool_per_restart_can_switch_optimizer():
    pool = [{"optimizer": "sgd_momentum", "lr": 0.1, "momentum": 0.9},
            {"optimizer": "adamw", "lr": 1e-3}]
    ctrl = _ctrl(pool=pool, n_cycles=4)
    model = ctrl.model
    # run through >1 cycle so the bandit selects at a restart
    for s in range(45):
        opt = ctrl.begin_step(s)
        opt.zero_grad(set_to_none=True)
        model(torch.randn(8, 8)).square().mean().backward()
        opt.step()
        ctrl.end_step(s, loss=1.0 / (s + 1))
    assert len(ctrl.restarts) >= 1
    assert ctrl.active_name in ("sgd_momentum", "adamw")
