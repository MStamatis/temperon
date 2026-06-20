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
