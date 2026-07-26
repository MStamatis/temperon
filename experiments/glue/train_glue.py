"""Phase 8: sharpness-budget allocation for LM FINE-TUNING (GLUE).

This is where SAM is *known* to pay for language models (Bahri et al., ACL 2022,
arXiv 2110.08529 -- largest gains when task data is limited), and where its 2x
cost is the only adoption barrier. So it is the natural target for the
allocation law: same three arms, differing ONLY in where the SAM budget goes.

  --arm off   : plain AdamW (1 pass/step)
  --arm full  : SAM every step (2 passes all run long)
  --arm tail  : SAM only from the decay start on (our allocation)

Schedule is WSD (warmup -> stable -> linear decay) for ALL arms, so the tail arm
gets a *fresh anneal* owned entirely by SAM -- the property that made vision
arm P win and arm O fail, kept identical across vision / LM / fine-tuning.

Because 'off' and 'tail' are bit-identical before the switch, running both on a
seed gives a FREE run-to-run noise floor (see phase7 notes).

Usage (inside the container):
  python experiments/glue/train_glue.py --task rte --arm tail --seed 42
  python experiments/glue/train_glue.py --task rte --arm off --smoke
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lm.train_lm import rho_now, sam_start_step, wsd_mult  # noqa: E402

from eos_switch.optimizers.sam import SAM  # noqa: E402

# GLUE subsets where SAM's published gains are largest (small training sets)
# and where a full sweep is minutes, not hours.
TASKS = {
    "rte":  dict(keys=("sentence1", "sentence2"), n_labels=2, metric="acc"),
    "mrpc": dict(keys=("sentence1", "sentence2"), n_labels=2, metric="acc_f1"),
    "stsb": dict(keys=("sentence1", "sentence2"), n_labels=1, metric="pearson"),
    "cola": dict(keys=("sentence", None),         n_labels=2, metric="mcc"),
}


def compute_metric(kind: str, preds: np.ndarray, labels: np.ndarray) -> dict:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import f1_score, matthews_corrcoef
    if kind == "pearson":
        return {"pearson": float(pearsonr(preds, labels)[0]),
                "spearman": float(spearmanr(preds, labels)[0]),
                "score": float(pearsonr(preds, labels)[0])}
    if kind == "mcc":
        m = float(matthews_corrcoef(labels, preds))
        return {"mcc": m, "score": m}
    acc = float((preds == labels).mean())
    out = {"acc": acc, "score": acc}
    if kind == "acc_f1":
        out["f1"] = float(f1_score(labels, preds))
        out["score"] = (acc + out["f1"]) / 2  # GLUE convention for MRPC
    return out


def build_loaders(task: str, tok, cfg, seed: int):
    from datasets import load_dataset
    spec = TASKS[task]
    ds = load_dataset("nyu-mll/glue", task)
    k1, k2 = spec["keys"]

    def enc(batch):
        args = (batch[k1],) if k2 is None else (batch[k1], batch[k2])
        return tok(*args, truncation=True, max_length=int(cfg["max_len"]),
                   padding="max_length")

    keep = ["input_ids", "attention_mask", "label"]
    ds = ds.map(enc, batched=True)
    ds = ds.remove_columns([c for c in ds["train"].column_names if c not in keep])
    ds.set_format("torch")
    g = torch.Generator().manual_seed(seed)
    train = DataLoader(ds["train"], batch_size=int(cfg["batch_size"]),
                       shuffle=True, generator=g, drop_last=True)
    val = DataLoader(ds["validation"], batch_size=64, shuffle=False)
    return train, val


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--arm", required=True, choices=["off", "full", "tail"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", default="configs/glue_base.yaml")
    ap.add_argument("--output", default="results/phase8")
    ap.add_argument("--smoke", action="store_true",
                    help="2 epochs on 256 examples; no result files")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    spec = TASKS[args.task]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    name = cfg["model"]
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(
        name, num_labels=spec["n_labels"],
        problem_type="regression" if spec["n_labels"] == 1 else None,
    ).to(device)

    train_dl, val_dl = build_loaders(args.task, tok, cfg, args.seed)
    epochs = 2 if args.smoke else int(cfg["epochs"])
    steps_per_epoch = len(train_dl) if not args.smoke else min(len(train_dl), 8)
    total_steps = epochs * steps_per_epoch

    decay_frac = float(cfg["decay_frac"])
    warmup = max(1, int(round(float(cfg["warmup_frac"]) * total_steps)))
    # The tail owns the whole decay phase: SAM turns on exactly at decay start.
    start_frac = 1.0 - decay_frac
    sam_start = sam_start_step(args.arm, start_frac, total_steps)
    base_rho = float(cfg["rho"])
    ramp = int(cfg.get("rho_ramp_frac", 0) * total_steps)

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]),
                            weight_decay=float(cfg["weight_decay"]))
    for g in opt.param_groups:
        g["base_lr"] = g["lr"]
    sam = SAM(opt, rho=base_rho) if args.arm != "off" else None
    clip = float(cfg.get("grad_clip", 1.0))

    @torch.no_grad()
    def evaluate() -> dict:
        model.eval()
        P, L = [], []
        for batch in val_dl:
            labels = batch.pop("label")
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            P.append((logits.squeeze(-1) if spec["n_labels"] == 1
                      else logits.argmax(-1)).float().cpu())
            L.append(labels.float())
        model.train()
        return compute_metric(spec["metric"], torch.cat(P).numpy(),
                              torch.cat(L).numpy())

    model.train()
    step = 0
    t0 = time.perf_counter()
    history = []
    for ep in range(epochs):
        for i, batch in enumerate(train_dl):
            if args.smoke and i >= steps_per_epoch:
                break
            mult = wsd_mult(step, total_steps, warmup, decay_frac)
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * mult
            labels = batch.pop("label").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            if spec["n_labels"] == 1:
                labels = labels.float()

            loss = model(**batch, labels=labels).loss
            loss.backward()
            if sam is not None and step >= sam_start:
                sam.rho = rho_now(base_rho, step, sam_start, ramp)
                sam.first_step(zero_grad=True)
                # NOTE: dropout masks differ between the two passes (standard
                # SAM-on-transformers practice; we do not re-seed).
                model(**batch, labels=labels).loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                sam.second_step()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
        m = evaluate()
        wall = time.perf_counter() - t0
        sam_on = int(sam is not None and step > sam_start)
        history.append({"epoch": ep, "step": step, "sam_on": sam_on,
                        "wall_clock_s": round(wall, 1), **m})
        print(f"ep{ep} step{step} sam={sam_on} "
              + " ".join(f"{k} {v:.4f}" for k, v in m.items() if k != "score")
              + f" | {wall:.0f}s")

    if args.smoke:
        print("smoke ok")
        return

    best = max(h["score"] for h in history)
    final = history[-1]
    out_dir = os.path.join(args.output, args.task, args.arm, f"seed{args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump({"task": args.task, "arm": args.arm, "seed": args.seed,
                   "final_score": final["score"], "best_score": best,
                   "wall_clock_s": final["wall_clock_s"],
                   "total_steps": total_steps, "sam_start_step": sam_start,
                   "history": history, "cfg": cfg}, f, indent=2)
    print(f"done: {args.task}/{args.arm}/seed{args.seed} final {final['score']:.4f} "
          f"best {best:.4f} wall {final['wall_clock_s']:.0f}s")


if __name__ == "__main__":
    main()
