"""Phase 7: sharpness-budget allocation on a GPT-2-class LM (single GPU).

Three arms, identical model/data/schedule, differing ONLY in where the SAM
two-pass budget is spent:

  sam_mode: off   -- Muon+AdamW baseline (1 forward/backward per step)
  sam_mode: full  -- SAM on every step (2x passes all run long)
  sam_mode: tail  -- SAM only from sam_start_frac on, aligned to the WSD
                     decay phase (the LM analog of the Phase-6 hand-off:
                     the cheap phase ends BEFORE the anneal begins, and the
                     SAM tail owns the entire anneal)

Data-constrained multi-epoch pretraining: total_tokens budget over a
subset_tokens slice of WikiText-103 (repetition pressure is where SAM pays).
Optimizer split: Muon on the 2D transformer-block matrices, AdamW on
embeddings + 1D params. WSD (warmup-stable-decay) schedule for both.

Usage (inside the container):
  python experiments/lm/train_lm.py --config configs/lm_smoke.yaml
  python experiments/lm/train_lm.py --config configs/lm_muon.yaml --bench 60
  python experiments/lm/train_lm.py --config configs/lm_handoff.yaml --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lm_model import GPT, GPTConfig  # noqa: E402

from eos_switch.optimizers.muon import Muon  # noqa: E402
from eos_switch.optimizers.sam import SAM  # noqa: E402


class OptimizerTeam:
    """Present several optimizers as one (param_groups/state/step/zero_grad)
    so the loop-driven SAM wrapper can perturb across all of them with a
    single global-norm geometry, exactly as in the vision experiments."""

    def __init__(self, optimizers: list[torch.optim.Optimizer]) -> None:
        self.optimizers = optimizers
        self.param_groups = [g for opt in optimizers for g in opt.param_groups]
        self.state: defaultdict = defaultdict(dict)  # SAM's e_w stash only

    def step(self) -> None:
        for opt in self.optimizers:
            opt.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)


# --- schedule -----------------------------------------------------------------

def wsd_mult(step: int, total: int, warmup: int, decay_frac: float,
             min_frac: float = 0.0) -> float:
    """Warmup-Stable-Decay multiplier in [min_frac, 1]."""
    decay_start = int(round((1.0 - decay_frac) * total))
    if step < warmup:
        return (step + 1) / max(1, warmup)
    if step < decay_start:
        return 1.0
    span = max(1, total - decay_start)
    frac = (step - decay_start) / span
    return 1.0 - (1.0 - min_frac) * min(1.0, frac)


def sam_start_step(mode: str, start_frac: float, total: int) -> int:
    if mode == "off":
        return total + 1  # never
    if mode == "full":
        return 0
    if mode == "tail":
        return int(round(start_frac * total))
    raise ValueError(f"sam_mode must be off/full/tail, got {mode!r}")


def rho_now(base_rho: float, step: int, start: int, ramp: int) -> float:
    if ramp <= 0:
        return base_rho
    return base_rho * min(1.0, (step - start + 1) / ramp)


# --- data ---------------------------------------------------------------------

class TokenBin:
    def __init__(self, path: str, ctx: int, limit_tokens: int | None = None):
        data = np.memmap(path, dtype=np.uint16, mode="r")
        if limit_tokens is not None and limit_tokens < len(data):
            data = data[:limit_tokens]
        if len(data) < ctx + 1:
            raise ValueError(f"{path}: {len(data)} tokens < ctx+1")
        self.data = data
        self.ctx = ctx

    def sample(self, bs: int, gen: torch.Generator, device) -> tuple:
        ix = torch.randint(len(self.data) - self.ctx - 1, (bs,), generator=gen)
        x = torch.stack([torch.from_numpy(
            self.data[i:i + self.ctx].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(
            self.data[i + 1:i + 1 + self.ctx].astype(np.int64)) for i in ix])
        if str(device).startswith("cuda"):
            return (x.pin_memory().to(device, non_blocking=True),
                    y.pin_memory().to(device, non_blocking=True))
        return x.to(device), y.to(device)

    def eval_batches(self, bs: int, device):
        """Fixed non-overlapping ctx windows over the whole bin."""
        n = (len(self.data) - 1) // self.ctx
        for start in range(0, n, bs):
            idx = range(start, min(start + bs, n))
            x = torch.stack([torch.from_numpy(
                self.data[i * self.ctx:(i + 1) * self.ctx].astype(np.int64))
                for i in idx])
            y = torch.stack([torch.from_numpy(
                self.data[i * self.ctx + 1:(i + 1) * self.ctx + 1].astype(np.int64))
                for i in idx])
            yield x.to(device), y.to(device)


# --- main ---------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--output", default="results/phase7")
    ap.add_argument("--continue", dest="cont", action="store_true")
    ap.add_argument("--bench", type=int, default=0,
                    help="run N timed steps (after warmup) and exit; no files")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="dotted config overrides, e.g. sam_mode=full")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    for kv in args.set:
        key, val = kv.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = yaml.safe_load(val)
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision("high")

    ctx, bs = int(cfg["ctx"]), int(cfg["device_bs"])
    tok_per_step = ctx * bs
    total_steps = int(cfg["total_tokens"]) // tok_per_step
    subset = cfg.get("subset_tokens")

    data_dir = cfg.get("data_dir", "data/lm/wt103")
    train_bin = TokenBin(os.path.join(data_dir, "train.bin"), ctx,
                         int(subset) if subset else None)
    val_bin = TokenBin(os.path.join(data_dir, "val.bin"), ctx)

    m = cfg.get("model", {})
    gcfg = GPTConfig(ctx=ctx, n_layer=int(m.get("n_layer", 12)),
                     n_head=int(m.get("n_head", 12)),
                     n_embd=int(m.get("n_embd", 768)))
    model = GPT(gcfg).to(device)
    raw_model = model
    if cfg.get("compile", True) and device == "cuda":
        model = torch.compile(model)

    muon_params, adamw_params = raw_model.param_split()
    mcfg, acfg = cfg["muon"], cfg["adamw"]
    muon = Muon(muon_params, lr=float(mcfg["lr"]),
                momentum=float(mcfg.get("momentum", 0.95)), nesterov=True,
                weight_decay=float(mcfg.get("weight_decay", 0.0)),
                ns_dtype=mcfg.get("ns_dtype", "bf16"))
    adamw = torch.optim.AdamW(adamw_params, lr=float(acfg["lr"]),
                              betas=tuple(acfg.get("betas", (0.9, 0.95))),
                              weight_decay=float(acfg.get("weight_decay", 0.1)))
    team = OptimizerTeam([muon, adamw])
    for g in team.param_groups:
        g["base_lr"] = g["lr"]

    sam_mode = cfg.get("sam_mode", "off")
    sam = SAM(team, rho=float(cfg.get("rho", 0.05))) if sam_mode != "off" else None
    base_rho = float(cfg.get("rho", 0.05))
    sam_start = sam_start_step(sam_mode, float(cfg.get("sam_start_frac", 0.7)),
                               total_steps)
    ramp = int(cfg.get("sam_rho_ramp_steps", 0))
    warmup = int(cfg["warmup_steps"])
    decay_frac = float(cfg["decay_frac"])
    clip = float(cfg.get("grad_clip", 1.0))
    eval_every = int(cfg.get("eval_every", 250))
    autocast = torch.autocast(device_type=device, dtype=torch.bfloat16,
                              enabled=(device == "cuda"))

    gen = torch.Generator().manual_seed(seed)

    def train_step(step: int) -> float:
        mult = wsd_mult(step, total_steps, warmup, decay_frac,
                        float(cfg.get("min_lr_frac", 0.0)))
        for g in team.param_groups:
            g["lr"] = g["base_lr"] * mult
        sam_on = sam is not None and step >= sam_start
        x, y = train_bin.sample(bs, gen, device)
        with autocast:
            _, loss = model(x, y)
        loss.backward()
        if sam_on:
            sam.rho = rho_now(base_rho, step, sam_start, ramp)
            sam.first_step(zero_grad=True)   # perturb w/ raw grads, zero
            with autocast:
                _, loss2 = model(x, y)
            loss2.backward()
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), clip)
            sam.second_step()                # restore + base step
        else:
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), clip)
            team.step()
        team.zero_grad(set_to_none=True)
        return loss.item()

    @torch.no_grad()
    def evaluate() -> float:
        tot, n = 0.0, 0
        for x, y in val_bin.eval_batches(bs, device):
            with autocast:
                _, loss = model(x, y)
            tot += loss.item() * x.shape[0]
            n += x.shape[0]
        return tot / n

    # --- bench mode: measure step time for THIS config's sam_mode, exit ------
    if args.bench > 0:
        # 20 warmup steps: torch.compile/cudagraph settling must finish BEFORE
        # the timed window (a 2-pass SAM warmup executes the graph 2x more, so
        # a short warmup can bias the 1-pass arm slow, never the reverse).
        n_warm = 20
        print(f"bench: model={raw_model.num_params()/1e6:.0f}M tok/step={tok_per_step} "
              f"sam_mode={sam_mode} compile={cfg.get('compile', True)}")
        if sam is not None:
            sam_start = 0  # time the SAM-on region (that's the cost we're sizing)
        times = []
        for i in range(n_warm + args.bench):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = train_step(warmup + i)  # stable-phase lr
            if device == "cuda":
                torch.cuda.synchronize()
            if i >= n_warm:
                times.append(time.perf_counter() - t0)
        ts = sorted(times)
        med = ts[len(ts) // 2]
        p90 = ts[int(len(ts) * 0.9)]
        print(f"bench: {len(ts)} steps  median {med*1000:.1f} ms  "
              f"mean {sum(ts)/len(ts)*1000:.1f} ms  min {ts[0]*1000:.1f}  "
              f"p90 {p90*1000:.1f}  max {ts[-1]*1000:.1f} ms  "
              f"{tok_per_step/med:,.0f} tok/s")
        print(f"bench: projected {total_steps} steps at this rate: "
              f"{total_steps*med/3600:.2f} h")
        return

    # --- output dirs / resume -------------------------------------------------
    run_name = cfg.get("run_name") or os.path.splitext(os.path.basename(args.config))[0]
    out_dir = os.path.join(args.output, run_name, f"seed{seed}")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "evals.csv")
    ckpt_path = os.path.join(out_dir, "checkpoint.pt")

    start_step, wall_base = 0, 0.0
    if args.cont and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        raw_model.load_state_dict(ck["model"])
        muon.load_state_dict(ck["muon"])
        adamw.load_state_dict(ck["adamw"])
        gen.set_state(ck["gen"])
        start_step, wall_base = ck["step"], ck["wall_clock_s"]
        print(f"resumed from step {start_step} ({wall_base:.0f}s)")
    else:
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump({"cfg": cfg, "seed": seed, "total_steps": total_steps,
                       "tok_per_step": tok_per_step, "sam_start_step": sam_start,
                       "params_m": raw_model.num_params() / 1e6,
                       "argv": sys.argv}, f, indent=2)
        with open(csv_path, "w") as f:
            f.write("step,tokens,epoch,lr_mult,rho,sam_on,train_loss,"
                    "val_loss,val_ppl,step_ms,wall_clock_s\n")

    print(f"{run_name} seed{seed}: {raw_model.num_params()/1e6:.0f}M params, "
          f"{total_steps} steps x {tok_per_step} tok, sam_mode={sam_mode} "
          f"(sam_start={sam_start}), train pool "
          f"{len(train_bin.data):,} tok -> "
          f"{int(cfg['total_tokens'])/len(train_bin.data):.1f} passes")

    t_run = time.perf_counter()
    t_seg = t_run
    seg_steps = 0
    ema = None
    for step in range(start_step, total_steps):
        loss = train_step(step)
        ema = loss if ema is None else 0.98 * ema + 0.02 * loss
        seg_steps += 1
        if (step + 1) % eval_every == 0 or step == total_steps - 1:
            if device == "cuda":
                torch.cuda.synchronize()
            step_ms = (time.perf_counter() - t_seg) / seg_steps * 1000
            vl = evaluate()
            wall = wall_base + (time.perf_counter() - t_run)
            tokens = (step + 1) * tok_per_step
            mult = wsd_mult(step, total_steps, warmup, decay_frac)
            sam_on = int(sam is not None and step >= sam_start)
            row = (f"{step+1},{tokens},{tokens/len(train_bin.data):.3f},"
                   f"{mult:.4f},{(sam.rho if sam_on else 0.0):.4f},{sam_on},"
                   f"{ema:.4f},{vl:.4f},{math.exp(vl):.2f},{step_ms:.1f},"
                   f"{wall:.1f}\n")
            with open(csv_path, "a") as f:
                f.write(row)
            print(f"step {step+1}/{total_steps} val_loss {vl:.4f} "
                  f"ppl {math.exp(vl):.1f} sam={sam_on} {step_ms:.0f} ms/step "
                  f"wall {wall:.0f}s")
            torch.save({"model": raw_model.state_dict(), "muon": muon.state_dict(),
                        "adamw": adamw.state_dict(), "gen": gen.get_state(),
                        "step": step + 1, "wall_clock_s": wall}, ckpt_path)
            t_seg = time.perf_counter()
            seg_steps = 0

    vl = evaluate()
    wall = wall_base + (time.perf_counter() - t_run)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"final_val_loss": vl, "final_val_ppl": math.exp(vl),
                   "total_tokens": total_steps * tok_per_step,
                   "wall_clock_s": wall, "sam_mode": sam_mode}, f, indent=2)
    print(f"done: final val_loss {vl:.4f} ppl {math.exp(vl):.2f} wall {wall:.0f}s")


if __name__ == "__main__":
    main()
