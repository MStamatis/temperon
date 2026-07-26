"""SAM must work with ANY base optimizer from step 0.

Regression guard for the Phase-8 crash: SAM used to stash the perturbation in
`base_optimizer.state[p]`, and torch.optim.Adam/AdamW lazily initialize their
own slots with `if len(state) == 0`. A pre-populated state made them skip that
initialization and raise `KeyError: 'exp_avg'` on the very first SAM step.
SGD and Muon read their buffers with .get(), so they never hit it -- which is
why every vision arm passed while the first Adam-based arm died.
"""

import pytest
import torch
import torch.nn as nn

from eos_switch.optimizers.muon import Muon
from eos_switch.optimizers.sam import SAM


def _one_sam_step(base):
    torch.manual_seed(0)
    model = nn.Linear(6, 3)
    base_opt = base(model.parameters())
    sam = SAM(base_opt, rho=0.05)
    x, y = torch.randn(4, 6), torch.randn(4, 3)
    w0 = model.weight.detach().clone()

    nn.functional.mse_loss(model(x), y).backward()
    sam.first_step(zero_grad=True)
    perturbed = model.weight.detach().clone()
    nn.functional.mse_loss(model(x), y).backward()
    sam.second_step()
    sam.zero_grad()
    return model, w0, perturbed


@pytest.mark.parametrize("base", [
    lambda p: torch.optim.AdamW(p, lr=1e-3),
    lambda p: torch.optim.Adam(p, lr=1e-3),
    lambda p: torch.optim.SGD(p, lr=1e-2, momentum=0.9),
    lambda p: Muon(p, lr=1e-2),
], ids=["adamw", "adam", "sgd", "muon"])
def test_first_sam_step_works_on_fresh_optimizer(base):
    model, w0, perturbed = _one_sam_step(base)
    assert not torch.equal(w0, perturbed)              # climbed to w + e
    assert not torch.equal(model.weight.detach(), perturbed)  # restored+stepped
    assert torch.isfinite(model.weight).all()


def test_perturbation_does_not_leak_into_optimizer_state():
    """e_w must live in SAM's own dict, never in base_optimizer.state."""
    torch.manual_seed(0)
    model = nn.Linear(6, 3)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sam = SAM(opt, rho=0.05)
    nn.functional.mse_loss(model(torch.randn(4, 6)), torch.randn(4, 3)).backward()
    sam.first_step(zero_grad=True)
    assert sam._e_w, "perturbation was not stashed"
    for p in model.parameters():
        assert "e_w" not in opt.state[p]
    nn.functional.mse_loss(model(torch.randn(4, 6)), torch.randn(4, 3)).backward()
    sam.second_step()
    assert not sam._e_w, "perturbation was not cleared after second_step"
    # AdamW initialized its own slots normally.
    for p in model.parameters():
        assert {"exp_avg", "exp_avg_sq"} <= set(opt.state[p])


def test_two_consecutive_sam_steps_adamw():
    torch.manual_seed(0)
    model = nn.Linear(6, 3)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sam = SAM(opt, rho=0.05)
    x, y = torch.randn(4, 6), torch.randn(4, 3)
    for _ in range(2):
        nn.functional.mse_loss(model(x), y).backward()
        sam.first_step(zero_grad=True)
        nn.functional.mse_loss(model(x), y).backward()
        sam.record_sharpness()
        sam.second_step()
        sam.zero_grad()
    assert sam.last_sharpness is not None
    assert torch.isfinite(model.weight).all()
