"""Free-HVP EoS catapult (arm N): the free same-batch sharpness + restart timing."""

import torch
import torch.nn as nn

from eos_switch.controllers import build_controller
from eos_switch.controllers.base import SwitchEvent
from eos_switch.optimizers.muon import Muon
from eos_switch.optimizers.sam import SAM


def test_free_sharpness_exact_on_quadratic():
    # L = 0.5 * sum(a_i * w_i^2): g = a*w, H = diag(a). For a quadratic the SAM
    # two gradients give g' - g = rho*H*g_hat EXACTLY, so last_sharpness must
    # equal g_hat^T H g_hat = sum(a_i * g_hat_i^2).
    torch.manual_seed(0)
    a = torch.tensor([0.5, 2.0, 1.0, 4.0, 0.25])
    w = nn.Parameter(torch.randn(5))
    opt = SAM(torch.optim.SGD([w], lr=0.0), rho=0.1)  # lr=0: isolate the estimate

    opt.zero_grad()
    (0.5 * (a * w * w).sum()).backward()
    g = (a * w).detach()
    ghat = g / g.norm()
    expected = float((a * ghat * ghat).sum())   # g_hat^T H g_hat

    opt.first_step()
    (0.5 * (a * w * w).sum()).backward()         # gradient at the perturbed point
    opt.record_sharpness()                       # measure BEFORE any clipping
    opt.second_step()

    assert opt.last_sharpness == __import__("pytest").approx(expected, rel=1e-4)


def test_clipping_after_record_does_not_bias_sharpness():
    # record_sharpness must read the UNCLIPPED g'; clipping afterwards (as the
    # loop does) must not change the recorded value.
    torch.manual_seed(0)
    a = torch.tensor([0.5, 2.0, 1.0, 4.0, 0.25])
    w = nn.Parameter(torch.randn(5))
    opt = SAM(torch.optim.SGD([w], lr=0.0), rho=0.1)
    opt.zero_grad(); (0.5 * (a * w * w).sum()).backward()
    opt.first_step()
    (0.5 * (a * w * w).sum()).backward()
    opt.record_sharpness()
    recorded = opt.last_sharpness
    torch.nn.utils.clip_grad_norm_([w], 0.01)    # aggressive clip AFTER record
    assert opt.last_sharpness == recorded        # unchanged
    assert recorded > 0                          # PSD Hessian -> positive


def test_sharpness_none_before_step():
    opt = SAM(torch.optim.SGD([nn.Parameter(torch.randn(3))], lr=0.0), rho=0.05)
    assert opt.last_sharpness is None


def test_controller_builds_sam_wrapped_muon():
    ctrl = build_controller({"type": "sam_eos_catapult", "base_optimizer": "muon",
                             "lr": 0.01, "rho": 0.05, "warmup_steps": 0})
    ctrl.setup(nn.Linear(8, 4), steps_per_epoch=10, total_epochs=10)
    assert ctrl.sam is True
    assert ctrl.active_name == "sam_eos:muon"
    opt = ctrl.begin_step(0)
    assert isinstance(opt, SAM) and isinstance(opt.base_optimizer, Muon)


def test_edge_ratio_reads_free_sharpness():
    ctrl = build_controller({"type": "sam_eos_catapult", "lr": 0.1, "momentum": 0.9,
                             "rho": 0.05})
    ctrl.setup(nn.Linear(8, 4), steps_per_epoch=10, total_epochs=10)
    ctrl._lr = 0.1
    ctrl._opt.last_sharpness = 19.0           # threshold = (2+1.8)/0.1 = 38
    edge, s = ctrl._probe_edge_ratio()
    assert s == 19.0
    assert edge == __import__("pytest").approx(0.5, rel=1e-6)   # 19 / 38


def _robust_ctrl(**over):
    cfg = {"type": "sam_eos_catapult", "base_optimizer": "muon", "lr": 0.01,
           "momentum": 0.9, "rho": 0.05, "warmup_steps": 0, "check_every": 1,
           "min_dwell_checks": 0, "flat_frac": 0.5, "lr_gate_frac": 0.3,
           "sharp_ema_beta": 0.0, "switch_until_frac": 0.95}
    cfg.update(over)
    ctrl = build_controller(cfg)
    ctrl.setup(nn.Linear(8, 4), steps_per_epoch=10, total_epochs=10)  # 100 steps
    return ctrl


def _run(ctrl, n, sharp_fn):
    lrs = []
    for step in range(n):
        ctrl._opt.last_sharpness = sharp_fn(step)   # the loop fills this each step
        ctrl.begin_step(step)
        lrs.append(ctrl.active_lr)
    return lrs


def test_constant_sharpness_no_lr_pinning_and_cap_restarts():
    # The OLD-bug regression: with sharpness that never flattens, lr must still
    # cosine-decay (NOT pin at peak) and restarts must come from the cap, not
    # every check.
    ctrl = _robust_ctrl(cycle_epochs=2.0)        # cycle_steps = 20
    lrs = _run(ctrl, 60, lambda s: 5.0)          # constant -> never flattens
    assert min(lrs[5:20]) < 0.5 * ctrl.peak_lr   # lr annealed within the first cycle
    assert 1 <= len(ctrl.switch_events) <= 5     # ~cap-period restarts, not every step
    assert all("cap" in e.reason for e in ctrl.switch_events)


def test_relative_flatten_triggers_adaptive_restart():
    # High baseline then a drop, once lr has annealed -> an adaptive (flatten)
    # restart fires before the cap.
    ctrl = _robust_ctrl(cycle_epochs=3.0)        # cap = 30 steps
    lrs = _run(ctrl, 29, lambda s: 10.0 if s < 5 else 1.0)
    assert any("flatten" in e.reason for e in ctrl.switch_events)


def test_lr_gate_blocks_restart_while_lr_high():
    # Flattened immediately, but the lr-gate must block the restart while lr is
    # still high (slow cosine, cap far away).
    ctrl = _robust_ctrl(cycle_epochs=10.0)       # cap = 100, slow anneal
    _run(ctrl, 8, lambda s: 10.0 if s == 0 else 1.0)
    assert len(ctrl.switch_events) == 0


def test_no_probe_overhead():
    ctrl = build_controller({"type": "sam_eos_catapult", "lr": 0.01})
    ctrl.setup(nn.Linear(8, 4), steps_per_epoch=10, total_epochs=10)
    assert ctrl.overhead_frac(1000.0) == 0.0   # sharpness is free, no HVP probe
