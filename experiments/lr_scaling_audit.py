"""Audit whether OptiRoulette's LR-scaling actually fires during real training.

Runs CIFAR-100 / ResNet-18 with the *stock* OptiRoulette default profile
(17-epoch SGD warmup, then epoch-granularity roulette), wrapping
compatibility.adjust_lr to record every (old, new, lr_in, lr_out) call.
"""
from __future__ import annotations
import argparse
import torch
import torch.nn.functional as F
from optiroulette import OptiRoulette
from eos_switch.data import GPUCifar
from eos_switch.train.models import build_model
from eos_switch.train.seed import make_generator, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    data = GPUCifar("cifar100", device=dev)
    model = build_model("resnet18", 100).to(dev).to(memory_format=torch.channels_last)

    opt = OptiRoulette(model.parameters(), seed=args.seed)  # STOCK default profile
    print("default warmup:", opt.warmup_optimizer, "warmup_epochs:", opt.warmup_epochs,
          "drop_after_warmup:", opt.drop_after_warmup)
    print("active pool:", opt.active_names)
    print("compatibility (LR-scaling) active:", opt.compatibility is not None)

    cur_epoch = {"e": -1}
    calls = []
    if opt.compatibility is not None:
        _orig = opt.compatibility.adjust_lr
        def wrapped(old, new, lr):
            out = _orig(old, new, lr)
            rec = {"epoch": cur_epoch["e"], "from": old, "to": new,
                   "lr_in": lr, "lr_out": out, "changed": out != lr}
            calls.append(rec)
            return out
        opt.compatibility.adjust_lr = wrapped

    gen = make_generator(args.seed, dev)
    for epoch in range(args.epochs):
        cur_epoch["e"] = epoch
        opt.on_epoch_start(epoch)
        for bi, (x, y) in enumerate(data.train_batches(args.batch, gen)):
            x = x.contiguous(memory_format=torch.channels_last)
            opt.on_batch_start(bi)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
        opt.on_epoch_end(val_acc=0.0)
        print(f"epoch {epoch:>2}  phase={opt.phase:<8} opt={opt.active_optimizer_name:<8} "
              f"lr={float(opt.param_groups[0]['lr']):.3e}")

    print("\n==== adjust_lr CALL LOG (every optimizer switch) ====")
    for r in calls:
        flag = "  <-- LR SCALED!" if r["changed"] else ""
        print(f"  epoch {r['epoch']:>2}  {r['from']:>6} -> {r['to']:<8} "
              f"lr_in={r['lr_in']:.3e} lr_out={r['lr_out']:.3e}{flag}")
    n = len(calls); ch = sum(c["changed"] for c in calls)
    print(f"\nSUMMARY: {n} adjust_lr calls during training, {ch} actually changed the LR.")
    print("=> LR-scaling FIRED" if ch else "=> LR-scaling NEVER fired (de-facto no-op)")


if __name__ == "__main__":
    main()
