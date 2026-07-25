"""Phase 7 LM pieces: WSD schedule, SAM gating, optimizer split, data bins."""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "experiments", "lm"))

from lm_model import GPT, GPTConfig  # noqa: E402
from train_lm import (OptimizerTeam, TokenBin, rho_now, sam_start_step,  # noqa: E402
                      wsd_mult)

from eos_switch.optimizers.muon import Muon  # noqa: E402
from eos_switch.optimizers.sam import SAM  # noqa: E402

TINY = GPTConfig(vocab_size=128, ctx=16, n_layer=2, n_head=2, n_embd=32)


def test_wsd_warmup_stable_decay():
    total, warmup, dfrac = 100, 10, 0.4  # decay starts at step 60
    assert wsd_mult(0, total, warmup, dfrac) == pytest.approx(0.1)
    assert wsd_mult(9, total, warmup, dfrac) == pytest.approx(1.0)
    assert wsd_mult(30, total, warmup, dfrac) == 1.0
    assert wsd_mult(59, total, warmup, dfrac) == 1.0
    assert wsd_mult(60, total, warmup, dfrac) == pytest.approx(1.0)
    assert wsd_mult(80, total, warmup, dfrac) == pytest.approx(0.5)
    assert wsd_mult(100, total, warmup, dfrac) == pytest.approx(0.0)
    # min_lr_frac floor
    assert wsd_mult(100, total, warmup, dfrac, min_frac=0.1) == pytest.approx(0.1)


def test_sam_start_modes():
    assert sam_start_step("off", 0.7, 1000) == 1001   # never
    assert sam_start_step("full", 0.7, 1000) == 0
    assert sam_start_step("tail", 0.7, 1000) == 700
    with pytest.raises(ValueError):
        sam_start_step("sometimes", 0.7, 1000)


def test_rho_ramp():
    assert rho_now(0.05, 700, 700, 0) == 0.05          # no ramp
    assert rho_now(0.05, 700, 700, 4) == pytest.approx(0.05 * 0.25)
    assert rho_now(0.05, 703, 700, 4) == pytest.approx(0.05)
    assert rho_now(0.05, 900, 700, 4) == pytest.approx(0.05)


def test_param_split_muon_gets_block_matrices_only():
    model = GPT(TINY)
    muon_p, adamw_p = model.param_split()
    # 4 matrices per block (qkv, attn.proj, fc, mlp.proj), all >= 2D.
    assert len(muon_p) == 4 * TINY.n_layer
    assert all(p.ndim >= 2 for p in muon_p)
    muon_ids = {id(p) for p in muon_p}
    assert id(model.wte.weight) not in muon_ids       # tied head stays in AdamW
    assert id(model.wpe.weight) not in muon_ids
    assert all(p.ndim == 1 for p in adamw_p if id(p) != id(model.wte.weight)
               and id(p) != id(model.wpe.weight))     # rest of AdamW is LN 1D
    # Full coverage, no overlap.
    all_ids = {id(p) for p in model.parameters()}
    assert muon_ids | {id(p) for p in adamw_p} == all_ids
    assert muon_ids & {id(p) for p in adamw_p} == set()


def test_model_forward_and_tying():
    model = GPT(TINY)
    x = torch.randint(0, TINY.vocab_size, (3, TINY.ctx))
    logits, loss = model(x, x)
    assert logits.shape == (3, TINY.ctx, TINY.vocab_size)
    assert torch.isfinite(loss)
    assert model.lm_head.weight is model.wte.weight


def test_optimizer_team_sam_two_pass():
    torch.manual_seed(0)
    model = GPT(TINY)
    muon_p, adamw_p = model.param_split()
    team = OptimizerTeam([
        Muon(muon_p, lr=0.01, weight_decay=0.0),
        torch.optim.AdamW(adamw_p, lr=1e-3),
    ])
    sam = SAM(team, rho=0.05)
    x = torch.randint(0, TINY.vocab_size, (2, TINY.ctx))

    _, loss = model(x, x)
    loss.backward()
    w0 = model.blocks[0].attn.qkv.weight.detach().clone()
    sam.first_step(zero_grad=True)
    perturbed = model.blocks[0].attn.qkv.weight.detach().clone()
    assert not torch.equal(w0, perturbed)             # climbed to w + e
    _, loss2 = model(x, x)
    loss2.backward()
    sam.second_step()
    team.zero_grad()
    stepped = model.blocks[0].attn.qkv.weight.detach()
    assert not torch.equal(stepped, perturbed)        # restored, then stepped
    assert not torch.equal(stepped, w0)               # a real update happened
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_token_bin_sampling_and_eval_windows(tmp_path):
    arr = (np.arange(5000) % 100).astype(np.uint16)
    path = str(tmp_path / "toy.bin")
    arr.tofile(path)
    tb = TokenBin(path, ctx=16, limit_tokens=3000)
    assert len(tb.data) == 3000
    gen = torch.Generator().manual_seed(0)
    x, y = tb.sample(4, gen, "cpu")
    assert x.shape == y.shape == (4, 16)
    assert torch.equal(x[:, 1:], y[:, :-1])           # y is x shifted by one
    n_windows = sum(xb.shape[0] for xb, _ in tb.eval_batches(4, "cpu"))
    assert n_windows == (3000 - 1) // 16
    with pytest.raises(ValueError):
        TokenBin(path, ctx=16, limit_tokens=10)       # too few tokens
