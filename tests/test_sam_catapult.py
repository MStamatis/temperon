"""SAM-catapult controller (arm L, stage 1): two-pass mechanics + schedule reuse."""

import torch
import torch.nn as nn

from eos_switch.controllers import build_controller
from eos_switch.optimizers.sam import SAM


def _ctrl(model=None, **over):
    cfg = {"type": "sam_catapult", "base_optimizer": "sgd_momentum",
           "lr": 0.1, "min_lr": 0.0, "n_cycles": 4, "rho": 0.05}
    cfg.update(over)
    ctrl = build_controller(cfg)
    ctrl.setup(model if model is not None else nn.Linear(8, 4),
               steps_per_epoch=10, total_epochs=8)  # 80 steps, 4 cycles of 20
    return ctrl


def test_returns_sam_optimizer_and_flag():
    ctrl = _ctrl()
    assert ctrl.sam is True
    assert ctrl.sam_freeze_bn is False
    assert ctrl.active_name.startswith("sam:")
    opt = ctrl.begin_step(0)
    assert isinstance(opt, SAM)
    assert isinstance(opt.base_optimizer, torch.optim.SGD)


def test_freeze_bn_flag_passthrough():
    assert _ctrl(sam_freeze_bn=True).sam_freeze_bn is True
    assert _ctrl().sam_freeze_bn is False


def test_perturbation_radius_equals_rho():
    # first_step normalizes the global gradient, so ||w - w0|| == rho exactly.
    torch.manual_seed(0)
    model = nn.Linear(8, 4)
    ctrl = _ctrl(model=model, lr=0.0, rho=0.1)
    opt = ctrl.begin_step(0)
    w0 = [p.detach().clone() for p in model.parameters()]
    opt.zero_grad(set_to_none=True)
    model(torch.randn(16, 8)).square().mean().backward()
    opt.first_step()
    disp = torch.cat([(p - p0).reshape(-1) for p, p0 in zip(model.parameters(), w0)])
    assert abs(disp.norm().item() - 0.1) < 1e-4


def test_restores_weights_after_two_pass():
    # base lr=0 -> second_step's base.step() is a no-op, isolating the restore.
    torch.manual_seed(0)
    model = nn.Linear(8, 4)
    ctrl = _ctrl(model=model, lr=0.0)
    opt = ctrl.begin_step(0)
    w0 = model.weight.detach().clone()
    opt.zero_grad(set_to_none=True)
    model(torch.randn(16, 8)).square().mean().backward()
    opt.first_step()
    assert (model.weight - w0).norm().item() > 1e-6   # perturbed away
    model(torch.randn(16, 8)).square().mean().backward()
    opt.second_step()
    assert torch.allclose(model.weight, w0, atol=1e-6)  # restored (lr=0)


def test_update_uses_perturbed_point_gradient():
    # On a quadratic, g(w0+e) != g(w0): the SAM step must move by -lr*g(w0+e).
    torch.manual_seed(0)
    model = nn.Linear(6, 3, bias=False)
    ctrl = _ctrl(model=model, lr=0.01, momentum=0.0, nesterov=False,
                 weight_decay=0.0, rho=0.1)
    x, y = torch.randn(32, 6), torch.randn(32, 3)

    def mse():
        return ((model(x) - y) ** 2).mean()

    w0 = model.weight.detach().clone()
    opt = ctrl.begin_step(0)
    opt.zero_grad(set_to_none=True)
    mse().backward()
    g_clean = model.weight.grad.detach().clone()

    opt.first_step()                       # w -> w0 + e ; grads zeroed
    mse().backward()                       # gradient at the perturbed point
    g_pert = model.weight.grad.detach().clone()
    opt.second_step()                      # w -> w0 - lr*g_pert

    delta = model.weight.detach() - w0
    assert torch.allclose(delta, -0.01 * g_pert, atol=1e-5)        # uses perturbed grad
    assert not torch.allclose(delta, -0.01 * g_clean, atol=1e-4)   # NOT the clean grad


def test_schedule_matches_cyclic():
    # The LR schedule must be identical to arm J for the same parameters.
    common = dict(base_optimizer="sgd_momentum", lr=0.1, min_lr=0.0,
                  n_cycles=4, t_mult=2.0, warmup_steps=5)
    sam = build_controller({"type": "sam_catapult", "rho": 0.05, **common})
    cyc = build_controller({"type": "cyclic_catapult", **common})
    for c in (sam, cyc):
        c.setup(nn.Linear(8, 4), steps_per_epoch=10, total_epochs=8)
    for step in (0, 3, 7, 19, 20, 40, 79):
        sam.begin_step(step)
        cyc.begin_step(step)
        assert abs(sam.active_lr - cyc.active_lr) < 1e-9


# --- stage 2: EoS-coupled rho ------------------------------------------------

def _eos_ctrl(model=None, **over):
    # warmup_steps=0 so checks engage immediately; sharp_ema_beta=0 -> ema is the
    # latest sharpness (deterministic); peak basis -> lr_basis is constant.
    cfg = {"type": "sam_catapult", "base_optimizer": "sgd_momentum", "lr": 0.1,
           "min_lr": 0.0, "n_cycles": 4, "rho": 0.05, "eos_rho": True,
           "warmup_steps": 0, "check_every": 10, "rho_min": 0.01, "rho_max": 0.20,
           "sharp_ema_beta": 0.0, "edge_lr_basis": "peak"}
    cfg.update(over)
    ctrl = build_controller(cfg)
    ctrl.setup(model if model is not None else nn.Linear(8, 4),
               steps_per_epoch=10, total_epochs=8)
    ctrl._data = object()  # non-None so begin_step's probe guard passes
    return ctrl


def test_eos_rho_auto_calibrates_to_base_on_first_probe():
    ctrl = _eos_ctrl()
    ctrl._probe_sharpness = lambda: 5.0
    opt = ctrl.begin_step(10)  # first check anchors target_ratio -> rho == base
    assert abs(ctrl._cur_rho - ctrl.base_rho) < 1e-9
    assert abs(opt.rho - ctrl.base_rho) < 1e-9
    assert abs(ctrl.target_ratio - 5.0 / ((2 + 2 * 0.9) / 0.1)) < 1e-9


def test_eos_rho_larger_when_flatter_smaller_when_sharper():
    ctrl = _eos_ctrl()
    sharp = {"v": 5.0}
    ctrl._probe_sharpness = lambda: sharp["v"]
    ctrl.begin_step(10)            # anchor at sharp=5 -> rho == base
    base = ctrl.base_rho
    sharp["v"] = 2.5               # flatter (further from edge) -> rho grows
    ctrl.begin_step(20)
    assert ctrl._cur_rho > base
    assert abs(ctrl._cur_rho - 2 * base) < 1e-9
    sharp["v"] = 10.0              # sharper (nearer the edge) -> rho shrinks
    ctrl.begin_step(30)
    assert ctrl._cur_rho < base


def test_eos_rho_clamped_to_bounds():
    ctrl = _eos_ctrl(rho_min=0.02, rho_max=0.08)
    sharp = {"v": 5.0}
    ctrl._probe_sharpness = lambda: sharp["v"]
    ctrl.begin_step(10)            # anchor rho == base (0.05)
    sharp["v"] = 1e-4             # near-flat -> huge rho -> clamp to rho_max
    ctrl.begin_step(20)
    assert abs(ctrl._cur_rho - 0.08) < 1e-9
    sharp["v"] = 1e4             # very sharp -> tiny rho -> clamp to rho_min
    ctrl.begin_step(30)
    assert abs(ctrl._cur_rho - 0.02) < 1e-9


def test_eos_rho_negative_curvature_uses_rho_min():
    ctrl = _eos_ctrl()
    sharp = {"v": 5.0}
    ctrl._probe_sharpness = lambda: sharp["v"]
    ctrl.begin_step(10)            # anchor on positive curvature
    sharp["v"] = -3.0            # non-convex noise, not a basin -> rho_min
    ctrl.begin_step(20)
    assert abs(ctrl._cur_rho - ctrl.rho_min) < 1e-9


def test_eos_rho_disabled_keeps_fixed_rho_and_no_checks():
    ctrl = _eos_ctrl(eos_rho=False)
    ctrl._probe_sharpness = lambda: 999.0  # must never be consulted
    opt = ctrl.begin_step(10)
    assert opt.rho == 0.05
    assert ctrl.checks == []


def test_eos_rho_fixed_target_ratio_respected():
    # A fixed target_ratio (the EoS edge = 1.0) must NOT be overwritten by the
    # first probe, and rho == base when edge_ratio == target_ratio.
    ctrl = _eos_ctrl(target_ratio=1.0)
    ctrl._probe_sharpness = lambda: 38.0  # edge_ratio = 38*0.1/(2+2*0.9) = 1.0
    ctrl.begin_step(10)
    assert ctrl.target_ratio == 1.0
    assert abs(ctrl._cur_rho - ctrl.base_rho) < 1e-9


def test_eos_rho_current_basis_rides_lr():
    # With the "current" basis, a lower live lr lifts the threshold and thus rho.
    ctrl = _eos_ctrl(edge_lr_basis="current", warmup_steps=0)
    ctrl._probe_sharpness = lambda: 4.0
    ctrl.begin_step(10)
    assert ctrl.checks[-1]["lr_basis"] == round(ctrl.active_lr, 6)
