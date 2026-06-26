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


def test_low_sharpness_triggers_restart():
    # In-window, after the dwell, a low edge ratio must fire a catapult restart.
    ctrl = build_controller({"type": "sam_eos_catapult", "lr": 0.1, "momentum": 0.9,
                             "rho": 0.05, "warmup_steps": 0, "check_every": 1,
                             "restart_ratio": 0.1, "min_dwell_checks": 0,
                             "cycle_epochs": 5.0, "switch_until_frac": 0.9})
    ctrl.setup(nn.Linear(8, 4), steps_per_epoch=10, total_epochs=10)  # 100 steps
    ctrl._opt.last_sharpness = 0.0            # edge_ratio = 0 < restart_ratio
    before = len(ctrl.switch_events)
    ctrl.begin_step(10)                       # in-window, check fires
    assert len(ctrl.switch_events) == before + 1
    assert isinstance(ctrl.switch_events[-1], SwitchEvent)


def test_no_probe_overhead():
    ctrl = build_controller({"type": "sam_eos_catapult", "lr": 0.01})
    ctrl.setup(nn.Linear(8, 4), steps_per_epoch=10, total_epochs=10)
    assert ctrl.overhead_frac(1000.0) == 0.0   # sharpness is free, no HVP probe
