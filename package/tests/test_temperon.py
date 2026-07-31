"""Temperon package tests (CPU-only)."""

import pytest
import torch
import torch.nn as nn

from temperon import Temperon, cosine_tail, wsd


def _model():
    torch.manual_seed(0)
    return nn.Linear(6, 3)


def _closure_factory(model, opt, calls):
    x, y = torch.randn(8, 6), torch.randn(8, 3)

    def closure():
        calls.append(1)
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()
        return loss

    return closure


# --- gate ---------------------------------------------------------------------

def test_tail_boundary_and_gate():
    q = Temperon(torch.optim.SGD(_model().parameters(), lr=0.1),
               total_steps=1000, tail_frac=0.3)
    assert q.tail_start == 700
    assert not q.sam_active(699)
    assert q.sam_active(700) and q.sam_active(999)


def test_tail_frac_extremes():
    p = list(_model().parameters())
    always = Temperon(torch.optim.SGD(p, lr=0.1), total_steps=100, tail_frac=1.0)
    assert always.sam_active(0) and always.tail_start == 0
    never = Temperon(torch.optim.SGD(p, lr=0.1), total_steps=100, tail_frac=0.0)
    assert not any(never.sam_active(s) for s in range(100))


def test_rho_ramp():
    q = Temperon(torch.optim.SGD(_model().parameters(), lr=0.1),
               total_steps=100, tail_frac=0.5, rho=0.05, rho_ramp_steps=4)
    assert q.current_rho(49) == 0.0                      # cheap phase
    assert q.current_rho(50) == pytest.approx(0.05 * 0.25)
    assert q.current_rho(53) == pytest.approx(0.05)
    assert q.current_rho(90) == pytest.approx(0.05)      # capped
    flat = Temperon(torch.optim.SGD(_model().parameters(), lr=0.1),
                  total_steps=100, tail_frac=0.5, rho=0.05)
    assert flat.current_rho(50) == 0.05


def test_invalid_arguments():
    p = list(_model().parameters())
    with pytest.raises(ValueError):
        Temperon(torch.optim.SGD(p, lr=0.1), total_steps=0)
    with pytest.raises(ValueError):
        Temperon(torch.optim.SGD(p, lr=0.1), total_steps=10, tail_frac=1.5)
    with pytest.raises(ValueError):
        Temperon(torch.optim.SGD(p, lr=0.1), total_steps=10, rho=-1)
    with pytest.raises(ValueError):
        Temperon(torch.optim.SGD(p, lr=0.1), total_steps=10, transfer="magic")


# --- the two passes -----------------------------------------------------------

def test_closure_called_once_cheap_twice_in_tail():
    model = _model()
    q = Temperon(torch.optim.SGD(model.parameters(), lr=0.1),
               total_steps=4, tail_frac=0.5)
    calls: list[int] = []
    closure = _closure_factory(model, q, calls)
    q.step(closure); q.step(closure)          # cheap: 1 call each
    assert len(calls) == 2
    q.step(closure); q.step(closure)          # tail: 2 calls each
    assert len(calls) == 6


@pytest.mark.parametrize("make", [
    lambda p: torch.optim.AdamW(p, lr=1e-3),
    lambda p: torch.optim.Adam(p, lr=1e-3),
    lambda p: torch.optim.SGD(p, lr=1e-2, momentum=0.9),
], ids=["adamw", "adam", "sgd"])
def test_sam_from_step_zero_on_a_fresh_optimizer(make):
    """Regression: the perturbation must not pre-populate Adam's state dict."""
    model = _model()
    q = Temperon(make(model.parameters()), total_steps=4, tail_frac=1.0, rho=0.05)
    closure = _closure_factory(model, q, [])
    w0 = model.weight.detach().clone()
    q.step(closure)
    assert not torch.equal(model.weight.detach(), w0)
    assert torch.isfinite(model.weight).all()


def test_weights_are_restored_before_the_base_step():
    model = _model()
    q = Temperon(torch.optim.SGD(model.parameters(), lr=0.0),  # lr=0: step is a no-op
               total_steps=1, tail_frac=1.0, rho=0.1)
    w0 = model.weight.detach().clone()
    q.step(_closure_factory(model, q, []))
    # With lr=0 the only way the weights move is a leftover perturbation.
    assert torch.allclose(model.weight.detach(), w0, atol=1e-7)
    assert not q._e_w


def test_perturbation_never_enters_optimizer_state():
    model = _model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    q = Temperon(opt, total_steps=2, tail_frac=1.0, rho=0.05)
    q.step(_closure_factory(model, q, []))
    for p in model.parameters():
        assert "e_w" not in opt.state[p]
        assert {"exp_avg", "exp_avg_sq"} <= set(opt.state[p])


def test_step_requires_a_closure():
    q = Temperon(torch.optim.SGD(_model().parameters(), lr=0.1), total_steps=2)
    with pytest.raises(ValueError):
        q.step(None)


# --- hand-off -----------------------------------------------------------------

def test_hand_off_switches_optimizer_at_the_boundary():
    model = _model()
    cheap = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    tail = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    q = Temperon(cheap, total_steps=4, tail_frac=0.5, tail_optimizer=tail)
    closure = _closure_factory(model, q, [])

    q.step(closure); q.step(closure)
    assert q.optimizer is cheap and not q._switched
    q.step(closure)
    assert q.optimizer is tail and q._switched


def test_hand_off_copies_the_momentum_buffer():
    """Checked at hand-off time: the tail optimizer's own step would otherwise
    update the buffer in place, so a post-step comparison proves nothing."""
    model = _model()
    cheap = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    tail = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    q = Temperon(cheap, total_steps=4, tail_frac=0.5, tail_optimizer=tail)
    closure = _closure_factory(model, q, [])
    q.step(closure); q.step(closure)

    before = {p: cheap.state[p]["momentum_buffer"].clone() for p in model.parameters()}
    q._hand_off()
    for p in model.parameters():
        assert torch.equal(tail.state[p]["momentum_buffer"], before[p])
        assert tail.state[p]["momentum_buffer"] is not cheap.state[p]["momentum_buffer"]


def test_momentum_transfer_changes_the_trajectory():
    """The public-API consequence: transferring vs starting cold diverge."""
    def run(transfer):
        torch.manual_seed(0)
        model = nn.Linear(6, 3)
        cheap = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        tail = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        q = Temperon(cheap, total_steps=4, tail_frac=0.5, tail_optimizer=tail,
                   transfer=transfer)
        torch.manual_seed(1)
        x, y = torch.randn(8, 6), torch.randn(8, 3)

        def closure():
            q.zero_grad()
            loss = nn.functional.mse_loss(model(x), y)
            loss.backward()
            return loss

        for _ in range(4):
            q.step(closure)
        return model.weight.detach().clone()

    assert not torch.allclose(run("momentum"), run("none"))


def test_transfer_none_starts_cold():
    model = _model()
    cheap = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    tail = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    q = Temperon(cheap, total_steps=2, tail_frac=0.5, tail_optimizer=tail,
               transfer="none")
    closure = _closure_factory(model, q, [])
    q.step(closure)
    q._hand_off()
    for p in model.parameters():
        assert "momentum_buffer" not in tail.state.get(p, {})


# --- checkpointing ------------------------------------------------------------

def test_state_dict_roundtrip():
    model = _model()
    q = Temperon(torch.optim.AdamW(model.parameters(), lr=1e-3),
               total_steps=10, tail_frac=0.5)
    closure = _closure_factory(model, q, [])
    for _ in range(6):
        q.step(closure)
    sd = q.state_dict()

    model2 = _model()
    q2 = Temperon(torch.optim.AdamW(model2.parameters(), lr=1e-3),
                total_steps=10, tail_frac=0.5)
    q2.load_state_dict(sd)
    assert q2.step_count == 6
    assert q2.sam_active() == q.sam_active()


# --- schedules ----------------------------------------------------------------

def test_wsd_shape_and_alignment_with_tail_frac():
    total, decay_frac = 1000, 0.3
    f = wsd(total, warmup_steps=100, decay_frac=decay_frac)
    assert f(0) == pytest.approx(0.01)
    assert f(99) == pytest.approx(1.0)
    assert f(500) == 1.0
    assert f(699) == 1.0
    assert f(850) == pytest.approx(0.5)
    assert f(1000) == pytest.approx(0.0)
    # The documented contract: same fraction => tail opens exactly at decay.
    q = Temperon(torch.optim.SGD(_model().parameters(), lr=0.1),
               total_steps=total, tail_frac=decay_frac)
    assert f(q.tail_start - 1) == 1.0 and f(q.tail_start) == pytest.approx(1.0)
    assert f(q.tail_start + 1) < 1.0


def test_cosine_tail_is_flat_then_anneals():
    f = cosine_tail(100, tail_frac=0.5)
    assert f(0) == 1.0 and f(49) == 1.0
    assert f(50) == pytest.approx(1.0)
    assert f(75) == pytest.approx(0.5, abs=0.02)
    assert f(100) == pytest.approx(0.0, abs=1e-6)


def test_wsd_rejects_bad_arguments():
    with pytest.raises(ValueError):
        wsd(0, 10, 0.3)
    with pytest.raises(ValueError):
        wsd(100, 10, 1.5)
    with pytest.raises(ValueError):
        cosine_tail(100, tail_frac=0.0)


def test_momentum_transfer_into_adam_tail_is_a_safe_no_op():
    # Seeding momentum_buffer into an Adam-family tail would leave its state
    # non-empty before the first step and crash slot init (KeyError:
    # 'exp_avg'). The hand-off must skip groups without a `momentum` hyper-
    # parameter and the tail must then step cleanly.
    model = _model()
    p = list(model.parameters())
    cheap = torch.optim.SGD(p, lr=0.1, momentum=0.9)
    tail = torch.optim.AdamW(p, lr=1e-3)
    opt = Temperon(cheap, total_steps=10, tail_frac=0.5,
                   tail_optimizer=tail, transfer="momentum")
    calls = []
    closure = _closure_factory(model, opt, calls)
    for _ in range(10):
        opt.step(closure)
    assert opt.optimizer is tail
    for state in tail.state.values():
        assert "momentum_buffer" not in state
        assert "exp_avg" in state
