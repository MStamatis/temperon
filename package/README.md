# Temperon

**Pay for sharpness-aware minimization only where it pays you back.**

*Tempering* is the controlled heat treatment that follows quenching and sets a
metal's final toughness. This package governs the training equivalent: what
happens during the last anneal.

[SAM](https://arxiv.org/abs/2010.01412) improves generalization but doubles the
cost of every training step, which is why most people skip it. Temperon runs SAM
for a contiguous tail aligned to your learning-rate decay instead of the whole
run. In our experiments that reaches full-time-SAM quality for roughly a third
less wall-clock — and spreading the same SAM budget uniformly across training
does *worse* than either extreme.

> Status: **pre-release** (`0.1.0.dev0`), published alongside a paper in
> preparation. The API may still change. Single-GPU, bf16/fp32 tested;
> DDP and gradient accumulation are not supported yet.

```bash
pip install temperon
```

## Usage

```python
import torch
from temperon import Temperon, wsd

base = torch.optim.AdamW(model.parameters(), lr=2e-5)
total = len(loader) * epochs

opt = Temperon(base, total_steps=total, tail_frac=0.3, rho=0.05)
sched = torch.optim.lr_scheduler.LambdaLR(base, wsd(total, 250, decay_frac=0.3))

for batch in loader:
    def closure():
        opt.zero_grad()
        loss = loss_fn(model(batch.x), batch.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        return loss

    loss = opt.step(closure)   # 1 pass in the cheap phase, 2 in the tail
    sched.step()
```

The closure follows the `torch.optim.LBFGS` convention: zero the gradients,
compute the loss, call `backward()`, return the loss. Temperon calls it once per
step during the cheap phase and twice during the tail. Nothing else in your
training code changes.

### The one thing that matters

`tail_frac` must be **aligned to your learning-rate decay**, so the tail owns a
complete anneal. This is not a detail: in our ablations a SAM tail bolted onto
the middle of an ongoing cycle gained nothing at all (flat in tail length,
~1pp below full SAM), while a tail owning a fresh anneal matched full-time SAM.
Pass the same fraction to `wsd(..., decay_frac=f)` and `Temperon(tail_frac=f)`
and the alignment is exact.

### Cheap explorer, expensive refiner

You can also switch optimizers at the same boundary — a cheap explorer for most
of training, then a stronger optimizer owning the SAM tail:

```python
from temperon import cosine_tail

explorer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
refiner  = Muon(model.parameters(), lr=0.01, weight_decay=0.2)

opt = Temperon(explorer, total_steps=total, tail_frac=0.57,
             tail_optimizer=refiner, transfer="momentum")
sched = torch.optim.lr_scheduler.LambdaLR(explorer, cosine_tail(total, 0.57))
```

`transfer="momentum"` carries `momentum_buffer` across the switch.

## When *not* to use this

Measured limits, stated because they define where the method applies:

- **A cheap optimizer wins below its own ceiling.** Temperon never accelerates an
  accuracy target that plain SGD or Adam can already reach; it accelerates
  targets that only expensive methods reach at all. If your target is modest,
  use the cheap optimizer.
- **Saturated tasks gain nothing.** Where the frontier is reached early, the
  late anneal is not what is limiting you.
- **Language-model pretraining is out of scope.** In near-single-epoch
  pretraining there is little overfitting for SAM to prevent, and we measured
  no useful gain; under heavy data repetition SAM was actively worse. SAM's
  established language gains are in *fine-tuning*, which is where this package
  is aimed for LMs.

## Citing

See [CITATION.cff](https://github.com/MStamatis/temperon/blob/main/CITATION.cff).
The research code and the full experimental record — including the negative
controls this claim rests on — live in the same repository.

## License

MIT.
