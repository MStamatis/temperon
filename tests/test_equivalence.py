"""Equivalence: the `temperon` package must reproduce the research code.

This is what lets the published numbers transfer to the library without
re-running the benchmarks. Re-running them would be the weaker check anyway:
the 5-seed spread is +-0.0034, so a subtle bug would hide under seed noise,
whereas a trajectory comparison at fixed seed is exact.

Three things are compared, together covering everything the package does:

  * schedule semantics -- the SAM gate step and the rho ramp, against the
    `sam_catapult` controller;
  * SAM mechanics -- the perturbation geometry and the two-pass sequencing,
    against the `SAM` wrapper as `train/loop.py` drives it;
  * hand-off -- momentum transfer, against `HandoffController._copy_momentum`.

Run with the package on the path:

    ./eos.sh env PYTHONPATH=/workspace/package/src python -m pytest tests/test_equivalence.py -q
"""

import pytest
import torch
import torch.nn as nn

temperon = pytest.importorskip(
    "temperon", reason="package/src not on PYTHONPATH; see module docstring")
from temperon import Temperon  # noqa: E402

from eos_switch.controllers import build_controller  # noqa: E402
from eos_switch.controllers.handoff import HandoffController  # noqa: E402
from eos_switch.optimizers.sam import SAM  # noqa: E402

RHO = 0.05


def _model(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(6, 8), nn.Tanh(), nn.Linear(8, 3))


def _batches(n: int, seed: int = 1234):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(8, 6, generator=g), torch.randn(8, 3, generator=g))
            for _ in range(n)]


def _assert_identical(a: nn.Module, b: nn.Module, label: str) -> None:
    worst = 0.0
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert na == nb
        worst = max(worst, (pa - pb).abs().max().item())
    assert worst == 0.0, f"{label}: trajectories diverged, max |delta| = {worst:.3e}"


# --- 1. schedule semantics ----------------------------------------------------

@pytest.mark.parametrize("total,tail_frac", [
    (100, 0.3), (1000, 0.3), (12207, 0.3), (97, 0.5), (1550, 0.25), (64, 0.0),
])
def test_gate_opens_on_the_same_step(total, tail_frac):
    """Research `sam_start_frac` == 1 - package `tail_frac` must agree exactly;
    an off-by-one here would silently shift the whole allocation."""
    ctrl = build_controller({
        "type": "sam_catapult", "base_optimizer": "sgd_momentum", "lr": 0.1,
        "n_cycles": 1, "rho": RHO, "sam_start_frac": 1.0 - tail_frac,
    })
    ctrl.setup(_model(), steps_per_epoch=total, total_epochs=1)
    q = Temperon(torch.optim.SGD(_model().parameters(), lr=0.1),
               total_steps=total, tail_frac=tail_frac)

    assert ctrl._sam_start_step == q.tail_start
    for step in range(total):
        assert ctrl.sam_active(step) == q.sam_active(step), f"step {step}"


@pytest.mark.parametrize("ramp", [0, 1, 4, 400])
def test_rho_ramp_matches_in_the_sam_region(ramp):
    """rho is only read while SAM is on, so compare there (the package reports
    0.0 in the cheap phase, the controller leaves its last value)."""
    total, tail_frac = 1000, 0.3
    ctrl = build_controller({
        "type": "sam_catapult", "base_optimizer": "sgd_momentum", "lr": 0.1,
        "n_cycles": 1, "rho": RHO, "sam_start_frac": 1.0 - tail_frac,
        "sam_rho_ramp_steps": ramp,
    })
    ctrl.setup(_model(), steps_per_epoch=total, total_epochs=1)
    q = Temperon(torch.optim.SGD(_model().parameters(), lr=0.1),
               total_steps=total, tail_frac=tail_frac, rho=RHO,
               rho_ramp_steps=ramp)

    for step in range(q.tail_start, total):
        opt = ctrl.begin_step(step)
        assert opt.rho == pytest.approx(q.current_rho(step), abs=0.0, rel=0.0), \
            f"step {step}"


# --- 2. SAM mechanics ---------------------------------------------------------

def _research_run(model, base_opt, batches, tail_start, ramp):
    """Replicates train/loop.py: gated two-pass, base step when the gate is
    closed, rho ramped exactly as sam_catapult.begin_step does."""
    sam = SAM(base_opt, rho=RHO)
    loss_fn = nn.functional.mse_loss
    for step, (x, y) in enumerate(batches):
        sam_on = step >= tail_start
        if sam_on and ramp > 0:
            sam.rho = RHO * min(1.0, (step - tail_start + 1) / ramp)
        sam.zero_grad()
        loss_fn(model(x), y).backward()
        if sam_on:
            sam.first_step()                 # zero_grad=True, as in the loop
            loss_fn(model(x), y).backward()
            sam.second_step()
        else:
            base_opt.step()                  # loop.py: opt.base_optimizer.step()


def _package_run(model, base_opt, batches, total, tail_frac, ramp):
    q = Temperon(base_opt, total_steps=total, tail_frac=tail_frac, rho=RHO,
               rho_ramp_steps=ramp)
    loss_fn = nn.functional.mse_loss
    for x, y in batches:
        def closure(x=x, y=y):
            q.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            return loss
        q.step(closure)
    return q


def test_grad_norm_geometry_matches():
    """Both must use the same global L2 norm over all gradients -- a per-layer
    norm would change the perturbation direction, not just its size."""
    model = _model()
    x, y = _batches(1)[0]
    nn.functional.mse_loss(model(x), y).backward()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    research = SAM(opt, rho=RHO)._grad_norm()
    package = Temperon(opt, total_steps=1, tail_frac=1.0, rho=RHO)._grad_norm()
    assert torch.equal(research, package)


@pytest.mark.parametrize("base", ["sgd", "sgd_momentum", "adamw"])
def test_single_sam_step_is_bit_identical(base):
    makers = {
        "sgd": lambda p: torch.optim.SGD(p, lr=0.1),
        "sgd_momentum": lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9),
        "adamw": lambda p: torch.optim.AdamW(p, lr=1e-3),
    }
    batches = _batches(1)
    a, b = _model(), _model()
    _research_run(a, makers[base](a.parameters()), batches, tail_start=0, ramp=0)
    _package_run(b, makers[base](b.parameters()), batches, total=1,
                 tail_frac=1.0, ramp=0)
    _assert_identical(a, b, f"single SAM step ({base})")


@pytest.mark.parametrize("tail_frac,ramp", [(1.0, 0), (0.5, 0), (0.5, 4), (0.0, 0)])
def test_gated_trajectory_is_bit_identical(tail_frac, ramp):
    """The real test: 20 steps across the switch, comparing after every one."""
    n = 20
    batches = _batches(n)
    tail_start = int(round((1.0 - tail_frac) * n))
    a, b = _model(), _model()
    opt_a = torch.optim.SGD(a.parameters(), lr=0.1, momentum=0.9)
    opt_b = torch.optim.SGD(b.parameters(), lr=0.1, momentum=0.9)

    sam = SAM(opt_a, rho=RHO)
    q = Temperon(opt_b, total_steps=n, tail_frac=tail_frac, rho=RHO,
               rho_ramp_steps=ramp)
    loss_fn = nn.functional.mse_loss

    for step, (x, y) in enumerate(batches):
        sam_on = step >= tail_start and tail_frac > 0.0
        if sam_on and ramp > 0:
            sam.rho = RHO * min(1.0, (step - tail_start + 1) / ramp)
        sam.zero_grad()
        loss_fn(a(x), y).backward()
        if sam_on:
            sam.first_step()
            loss_fn(a(x), y).backward()
            sam.second_step()
        else:
            opt_a.step()

        def closure(x=x, y=y):
            q.zero_grad()
            loss = loss_fn(b(x), y)
            loss.backward()
            return loss
        q.step(closure)

        assert q.sam_active(step) == sam_on, f"gate disagrees at step {step}"
        _assert_identical(a, b, f"tail_frac={tail_frac} ramp={ramp} step={step}")


# --- 3. hand-off --------------------------------------------------------------

def test_momentum_transfer_matches_the_handoff_controller():
    """Same copy semantics: only params the tail optimizer owns, raw clone."""
    model_a, model_b = _model(), _model()
    old_a = torch.optim.SGD(model_a.parameters(), lr=0.1, momentum=0.9)
    new_a = torch.optim.SGD(model_a.parameters(), lr=0.01, momentum=0.9)
    old_b = torch.optim.SGD(model_b.parameters(), lr=0.1, momentum=0.9)
    new_b = torch.optim.SGD(model_b.parameters(), lr=0.01, momentum=0.9)

    loss_fn = nn.functional.mse_loss
    for m, o in ((model_a, old_a), (model_b, old_b)):
        for x, y in _batches(3):
            o.zero_grad()
            loss_fn(m(x), y).backward()
            o.step()

    n_research = HandoffController._copy_momentum(old_a, new_a)
    q = Temperon(old_b, total_steps=10, tail_frac=0.5, tail_optimizer=new_b)
    n_package = q._hand_off()

    assert n_research == n_package > 0
    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(new_a.state[pa]["momentum_buffer"],
                           new_b.state[pb]["momentum_buffer"])


def test_transfer_is_a_copy_not_an_alias():
    """Both implementations must clone: sharing the tensor would make the tail
    optimizer's first step corrupt the cheap optimizer's state."""
    model = _model()
    old = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    new = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    for x, y in _batches(2):
        old.zero_grad()
        nn.functional.mse_loss(model(x), y).backward()
        old.step()
    Temperon(old, total_steps=4, tail_frac=0.5, tail_optimizer=new)._hand_off()
    for p in model.parameters():
        assert new.state[p]["momentum_buffer"] is not old.state[p]["momentum_buffer"]
