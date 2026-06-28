"""Speed knobs for SAM+Muon: NS precision/steps, large-only NS, periodic SAM."""

import pytest
import torch
import torch.nn as nn

from eos_switch.controllers import build_controller
from eos_switch.optimizers.muon import Muon, newton_schulz_orthogonalize
from eos_switch.optimizers.pool import build_optimizer


def test_ns_bf16_runs_and_roughly_orthogonal():
    torch.manual_seed(0)
    u = newton_schulz_orthogonalize(torch.randn(8, 16), steps=5, dtype=torch.bfloat16)
    assert torch.isfinite(u).all()
    gram = u.float() @ u.float().T            # rows ~orthonormal (loose bf16 tol)
    assert (gram - torch.eye(8)).abs().max() < 0.25


def test_muon_bad_ns_dtype_raises():
    with pytest.raises(ValueError):
        Muon([nn.Parameter(torch.randn(4, 4))], ns_dtype="fp16")


def test_min_numel_skips_ns_on_small_2d():
    torch.manual_seed(0)
    w = torch.randn(4, 4)
    g = torch.randn(4, 4)
    # min_numel huge -> small 2D param takes the plain momentum-SGD update -lr*g.
    wa = nn.Parameter(w.clone())
    oa = Muon([wa], lr=0.1, momentum=0.0, nesterov=False, min_numel=10**9)
    wa.grad = g.clone(); oa.step()
    assert torch.allclose(wa.detach(), w - 0.1 * g, atol=1e-6)
    # min_numel 0 -> Newton-Schulz path -> different (orthogonalized) update.
    wb = nn.Parameter(w.clone())
    ob = Muon([wb], lr=0.1, momentum=0.0, nesterov=False, min_numel=0)
    wb.grad = g.clone(); ob.step()
    assert not torch.allclose(wb.detach(), w - 0.1 * g, atol=1e-4)


def test_build_optimizer_forwards_muon_opts():
    o = build_optimizer("muon", [nn.Parameter(torch.randn(4, 4))], lr=0.01,
                        ns_steps=3, ns_dtype="bf16", muon_min_numel=50)
    grp = o.param_groups[0]
    assert grp["ns_steps"] == 3
    assert grp["ns_dtype"] == torch.bfloat16
    assert grp["min_numel"] == 50


def test_ns_opts_reach_muon_via_sam_controller():
    c = build_controller({"type": "sam_catapult", "base_optimizer": "muon", "lr": 0.01,
                          "n_cycles": 4, "rho": 0.05, "ns_dtype": "bf16", "ns_steps": 3,
                          "muon_min_numel": 1000})
    c.setup(nn.Linear(8, 4), steps_per_epoch=10, total_epochs=8)
    opt = c.begin_step(0)
    assert isinstance(opt.base_optimizer, Muon)
    grp = opt.base_optimizer.param_groups[0]
    assert grp["ns_steps"] == 3 and grp["ns_dtype"] == torch.bfloat16 and grp["min_numel"] == 1000


def test_sam_period_exposed_and_defaults_to_one():
    c2 = build_controller({"type": "sam_catapult", "base_optimizer": "muon", "lr": 0.01,
                           "n_cycles": 4, "rho": 0.05, "sam_period": 2})
    c2.setup(nn.Linear(8, 4), steps_per_epoch=10, total_epochs=8)
    assert c2.sam_period == 2
    c1 = build_controller({"type": "sam_catapult", "base_optimizer": "muon", "lr": 0.01,
                           "n_cycles": 4, "rho": 0.05})
    c1.setup(nn.Linear(8, 4), steps_per_epoch=10, total_epochs=8)
    assert c1.sam_period == 1
