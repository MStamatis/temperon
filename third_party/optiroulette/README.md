# OptiRoulette Optimizer

This repository accompanies the paper "OptiRoulette Optimizer: A New Stochastic
Meta-Optimizer for up to 5.3x Faster Convergence".

A standalone, pip-installable PyTorch meta-optimizer that brings OptiRoulette's training logic to any project:
- random optimizer switching
- warmup -> roulette phase handling
- optimizer pool with active/backup swapping
- compatibility-aware replacement
- learning-rate scaling rules when switching
- momentum/state transfer on swap

The default behavior is loaded from the bundled `optimized.yaml` profile (same optimizer pool logic used in this project).

## Research Highlights

Based on the current paper draft, OptiRoulette is a stochastic meta-optimizer
that combines:
- warmup optimizer locking
- randomized sampling from an active optimizer pool
- compatibility-aware LR scaling during optimizer transitions
- failure-aware pool replacement

Reported mean test accuracy vs a single-optimizer AdamW baseline:

| Dataset | AdamW | OptiRoulette | Delta |
|---|---:|---:|---:|
| CIFAR-100 | 0.6734 | 0.7656 | +9.22 pp |
| CIFAR-100-C | 0.2904 | 0.3355 | +4.52 pp |
| SVHN | 0.9667 | 0.9756 | +0.89 pp |
| Tiny ImageNet | 0.5669 | 0.6642 | +9.73 pp |
| Caltech-256 | 0.5946 | 0.6920 | +9.74 pp |

Additional paper-reported highlights:
- Target-hit reliability: in the reported 10-seed suites, OptiRoulette reaches
  key validation targets in 10/10 runs, while the AdamW baseline reaches none
  of those targets within budget.
- Faster time-to-target on shared milestones (example: Caltech-256 @ 0.59,
  25.7 vs 77.0 epochs), with budget-capped lower-bound speedups up to 5.3x for
  non-attained baseline targets.
- Paired-seed analysis is positive across datasets, except CIFAR-100-C test
  ROC-AUC, which is not statistically significant in the current 10-seed study.

## Install

```bash
pip install OptiRoulette
```

## Examples

- [CIFAR-100 demo notebook](examples/quick_cifar100_optiroulette.ipynb)
- [Tiny-ImageNet demo notebook](examples/quick_tiny_imagenet_optiroulette.ipynb)

## Quick Use

```python
import torch
from optiroulette import OptiRoulette

model = torch.nn.Linear(128, 10)
optimizer = OptiRoulette(model.parameters())

for epoch in range(5):
    optimizer.on_epoch_start(epoch)

    for batch_idx in range(100):
        optimizer.on_batch_start(batch_idx)
        optimizer.zero_grad()
        x = torch.randn(32, 128)
        y = torch.randint(0, 10, (32,))
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()

    # pass validation accuracy for warmup plateau logic (optional)
    optimizer.on_epoch_end(val_acc=0.6)
```

## API

```python
from optiroulette import (
    OptiRoulette,
    OptiRouletteOptimizer,
    PoolConfig,
    get_default_config,
    get_default_seed,
    get_default_optimizer_specs,
    get_default_pool_setup,
    get_default_roulette_config,
)
```

## Configuration Reference

For a full settings guide (constructor arguments, `optimizer_specs`,
`pool_config`, warmup/roulette options, and defaults precedence), see:
- `docs/configuration.md`

For package maintainers (release/publish steps), see:
- `docs/release.md`

### Defaults behavior

`OptiRoulette(model.parameters())` uses:
- default optimizer specs from bundled `optimized.yaml`
- default roulette settings from bundled `optimized.yaml`
- default pool config + active/backup names from bundled `optimized.yaml`
- default LR scaling rules from bundled `optimized.yaml`
- default optimizer RNG seed from bundled `optimized.yaml` (`system.seed`, fallback `42`)

If you provide manual optimizer/pool settings, those are used instead of defaults:

```python
optimizer = OptiRoulette(
    model.parameters(),
    optimizer_specs={"adam": {"lr": 1e-3}},
)
```

Manual custom pool example (only your chosen optimizers are used):

```python
optimizer = OptiRoulette(
    model.parameters(),
    optimizer_specs={
        "adam": {"lr": 1e-3},
        "adamw": {"lr": 8e-4, "weight_decay": 0.01},
        "lion": {"lr": 1e-4, "betas": (0.9, 0.99)},
    },
    active_names=["adam", "adamw"],
    backup_names=["lion"],
)
```

Optional: override pool behavior too:

```python
optimizer = OptiRoulette(
    model.parameters(),
    optimizer_specs={
        "adam": {"lr": 1e-3},
        "adamw": {"lr": 8e-4, "weight_decay": 0.01},
        "lion": {"lr": 1e-4, "betas": (0.9, 0.99)},
    },
    pool_config={
        "num_active": 2,
        "num_backup": 1,
        "failure_threshold": -0.2,
        "consecutive_failure_limit": 3,
    },
    active_names=["adam", "adamw"],
    backup_names=["lion"],
)
```

## Third-Party Dependencies

This package depends on `pytorch-optimizer` for additional optimizer implementations.
See `THIRD_PARTY_LICENSES.md` for a short third-party license notice.

## Disclaimer

The OptiRoulette name refers exclusively to a machine-learning optimizer and has no
affiliation, sponsorship, or technical relation to roulette manufacturers, casinos,
or any physical/software gambling products or services.
