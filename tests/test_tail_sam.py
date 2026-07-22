"""Tail-SAM (arm O, Phase 6): sam_start_frac gate, rho ramp, MSAM hooks."""

import torch
import torch.nn as nn

from eos_switch.controllers import build_controller


def _controller(**over):
    cfg = {"type": "sam_catapult", "base_optimizer": "muon", "lr": 0.01,
           "n_cycles": 1, "rho": 0.05}
    cfg.update(over)
    c = build_controller(cfg)
    c.setup(nn.Linear(8, 4), steps_per_epoch=10, total_epochs=8)  # 80 steps
    return c


def _loop_sam_on(controller, step: int) -> bool:
    """Replicates the training loop's sam_on decision (gate + period)."""
    sam_on = getattr(controller, "sam", False)
    if sam_on:
        gate = getattr(controller, "sam_active", None)
        if gate is not None:
            sam_on = bool(gate(step))
    if sam_on:
        period = getattr(controller, "sam_period", 1)
        sam_on = (period <= 1) or (step % period == 0)
    return sam_on


def test_default_is_full_sam_arm_L():
    c = _controller()
    assert c.sam_active(0) and c.sam_active(79)
    c.begin_step(0)
    assert c.active_name == "sam:muon"
    assert c.switch_events == []  # no sam_on event when SAM starts at step 0


def test_gate_opens_at_start_frac():
    c = _controller(sam_start_frac=0.5)
    assert not c.sam_active(39)
    assert c.sam_active(40)
    c.begin_step(0)
    assert c.active_name == "muon"  # cheap phase: no sam: prefix in logs
    c.begin_step(39)
    assert c.switch_events == []
    c.begin_step(40)
    assert c.active_name == "sam:muon"
    assert len(c.switch_events) == 1
    assert c.switch_events[0].reason.startswith("sam_on")
    c.begin_step(41)
    assert len(c.switch_events) == 1  # recorded exactly once


def test_loop_gate_emulation_counts():
    c = _controller(sam_start_frac=0.5)
    assert sum(_loop_sam_on(c, s) for s in range(80)) == 40
    full = _controller()
    assert sum(_loop_sam_on(full, s) for s in range(80)) == 80


def test_rho_ramp_linear_after_start():
    c = _controller(sam_start_frac=0.5, sam_rho_ramp_steps=4)
    opt = c.begin_step(40)
    assert abs(opt.rho - 0.05 * 0.25) < 1e-12
    opt = c.begin_step(41)
    assert abs(opt.rho - 0.05 * 0.50) < 1e-12
    opt = c.begin_step(43)
    assert abs(opt.rho - 0.05) < 1e-12
    opt = c.begin_step(60)
    assert abs(opt.rho - 0.05) < 1e-12  # capped at rho


def test_no_ramp_keeps_fixed_rho():
    c = _controller(sam_start_frac=0.5)
    opt = c.begin_step(40)
    assert opt.rho == 0.05


def test_state_roundtrip_preserves_sam_on():
    c = _controller(sam_start_frac=0.5)
    c.begin_step(40)
    sd = c.state_dict()
    c2 = _controller(sam_start_frac=0.5)
    c2.load_state_dict(sd)
    assert c2._sam_now and c2._sam_on_recorded
    assert c2.active_name == "sam:muon"
    c2.begin_step(41)
    assert c2.switch_events == []  # no duplicate sam_on event after resume


# --- MSAM (zero-extra-pass momentum perturbation, cheap phase only) ----------

def test_start_frac_one_is_msam_full_mode():
    # arm Q (MSAM literature baseline): SAM never fires, MSAM fires everywhere.
    c = _controller(sam_start_frac=1.0, msam_rho=0.3)
    assert not any(c.sam_active(s) for s in range(80))
    assert all(c.msam_active(s) for s in range(80))
    assert sum(_loop_sam_on(c, s) for s in range(80)) == 0
    c.begin_step(0)
    assert c.active_name == "muon" and c.switch_events == []


def test_msam_gate_only_pre_sam():
    c = _controller(sam_start_frac=0.5, msam_rho=0.1)
    assert c.msam_active(10)
    assert not c.msam_active(40)  # true SAM takes over in the tail
    off = _controller(sam_start_frac=0.5)
    assert not off.msam_active(10)  # default: MSAM off


def test_msam_perturb_direction_and_restore():
    torch.manual_seed(0)
    c = _controller(sam_start_frac=0.5, msam_rho=0.1)
    base = c._opt.base_optimizer
    params = [p for g in base.param_groups for p in g["params"]]
    bufs = {}
    for p in params:
        bufs[p] = torch.randn_like(p)
        base.state[p]["momentum_buffer"] = bufs[p].clone()
    w0 = {p: p.detach().clone() for p in params}
    c.msam_perturb()
    norm = torch.norm(torch.stack([b.norm(p=2) for b in bufs.values()]), p=2)
    for p in params:
        expected = w0[p] + 0.1 * bufs[p] / (norm + 1e-12)
        assert torch.allclose(p.detach(), expected, atol=1e-6)
    c.msam_restore()
    for p in params:
        assert torch.allclose(p.detach(), w0[p], atol=1e-6)


def test_msam_noop_without_momentum():
    c = _controller(sam_start_frac=0.5, msam_rho=0.1)
    params = [p for g in c._opt.base_optimizer.param_groups for p in g["params"]]
    w0 = {p: p.detach().clone() for p in params}
    c.msam_perturb()  # no momentum buffers yet -> must be a no-op
    for p in params:
        assert torch.equal(p.detach(), w0[p])
    c.msam_restore()  # safe on empty perturbation list
