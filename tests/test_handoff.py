"""Hand-off controller (arm P, Phase 6): SGD explorer -> SAM+Muon refiner."""

import pytest
import torch
import torch.nn as nn

from eos_switch.controllers import build_controller
from eos_switch.optimizers.muon import Muon


def _cfg(**over):
    cfg = {"type": "handoff", "handoff_frac": 0.5,
           "stage1": {"base_optimizer": "sgd_momentum", "lr": 0.1,
                      "weight_decay": 5e-4, "n_cycles": 2, "t_mult": 1.0},
           "stage2": {"lr": 0.01, "weight_decay": 0.2, "rho": 0.05,
                      "warmup_steps": 5}}
    cfg.update(over)
    return cfg


def _controller(**over):
    c = build_controller(_cfg(**over))
    c.setup(nn.Linear(8, 4), steps_per_epoch=10, total_epochs=8)
    # handoff_epoch = 4 -> handoff at step 40 of 80
    return c


def test_stage_progression():
    c = _controller()
    assert c.handoff_epoch == 4
    assert not c.sam
    opt = c.begin_step(0)
    assert isinstance(opt, torch.optim.SGD)
    assert c.active_name == "sgd_momentum"
    opt = c.begin_step(40)
    assert c.sam
    assert c.sam_active(40)
    assert c.active_name == "sam:muon"
    assert isinstance(opt.base_optimizer, Muon)
    assert len(c.switch_events) == 1
    assert c.switch_events[0].reason.startswith("handoff|transfer:momentum")
    c.begin_step(41)
    assert len(c.switch_events) == 1  # the hand-off happens exactly once


def test_momentum_transfer_is_raw_copy():
    torch.manual_seed(0)
    c = _controller()
    for step in range(3):  # build up real SGD momentum
        opt = c.begin_step(step)
        for group in opt.param_groups:
            for p in group["params"]:
                p.grad = torch.randn_like(p)
        opt.step()
    sgd = c.s1.active_optimizer
    bufs = {p: sgd.state[p]["momentum_buffer"].clone()
            for g in sgd.param_groups for p in g["params"]}
    opt = c.begin_step(40)
    muon = opt.base_optimizer
    n = 0
    for group in muon.param_groups:
        for p in group["params"]:
            assert torch.allclose(muon.state[p]["momentum_buffer"], bufs[p])
            n += 1
    assert n == len(bufs) > 0


def test_transfer_none_starts_cold():
    c = _controller(handoff_transfer="none")
    for step in range(3):
        opt = c.begin_step(step)
        for group in opt.param_groups:
            for p in group["params"]:
                p.grad = torch.randn_like(p)
        opt.step()
    opt = c.begin_step(40)
    muon = opt.base_optimizer
    for group in muon.param_groups:
        for p in group["params"]:
            assert "momentum_buffer" not in muon.state[p]


def test_stage2_lr_warmup_reentry():
    c = _controller()
    opt = c.begin_step(40)  # stage-2 local step 0 of warmup 5
    assert abs(opt.param_groups[0]["lr"] - 0.01 * 1 / 5) < 1e-9
    opt = c.begin_step(44)  # local step 4 -> full peak
    assert abs(opt.param_groups[0]["lr"] - 0.01) < 1e-9


def test_handoff_epoch_clamped_to_valid_range():
    lo = _controller(handoff_frac=0.0)
    assert lo.handoff_epoch == 1
    hi = _controller(handoff_frac=1.0)
    assert hi.handoff_epoch == 7  # total_epochs - 1


def test_resume_in_stage2_skips_retransfer():
    c = _controller()
    c.begin_step(40)
    sd = c.state_dict()
    c2 = build_controller(_cfg())
    c2.setup(nn.Linear(8, 4), steps_per_epoch=10, total_epochs=8)
    c2.load_state_dict(sd)
    assert c2.sam and c2._done_handoff
    opt = c2.begin_step(41)
    assert isinstance(opt.base_optimizer, Muon)
    assert c2.switch_events == []  # no duplicate hand-off on resume


def test_bad_transfer_mode_raises():
    c = build_controller(_cfg(handoff_transfer="teleport"))
    with pytest.raises(ValueError):
        c.setup(nn.Linear(8, 4), steps_per_epoch=10, total_epochs=8)


def test_end_step_and_epoch_route_to_active_stage():
    c = _controller()
    c.begin_step(0)
    c.end_step(0, 1.0)
    assert c.s1._loss_ema is not None
    c.begin_step(40)
    c.end_step(40, 0.5)
    assert c.s2._loss_ema is not None
    c.end_epoch(5, 0.7)  # epoch past hand-off routes to stage 2 (no crash)
