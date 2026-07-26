# Temperon

Research code and archival record for **Temperon** — a study of *where* to spend
an expensive training mode. Named after *tempering*, the controlled heat
treatment applied after quenching that sets a metal's final toughness, because
that is exactly what the method governs: what happens during the last anneal.

The finding, in one sentence:

> Sharpness-Aware Minimization pays for its second pass only in the final
> anneal — allocating the SAM budget to a scheduled tail reaches full-time-SAM
> quality for roughly a third less wall-clock on three of four vision datasets
> and on a GPT-2-class language model. The fourth dataset is a measured
> boundary case, not an omission.

**What this is not.** This is *not* super-convergence. The method never
accelerates an accuracy target that a cheap optimizer can already reach; it
accelerates targets that only expensive methods reach at all. See
[Honest scope](#honest-scope) — that boundary is measured, stated, and part of
the result.

Everything runs inside a Docker container on a single NVIDIA RTX 5090
(Blackwell, sm_120, 32 GB VRAM). All numbers below come from runs in this
repository; see [REPRODUCE.md](REPRODUCE.md) for the exact commands.

## Headline results

Vision, 5 seeds per arm (`42, 1181241943, 958682846, 271828, 314159`),
wide ResNet-110, 100 epochs. "Hand-off" = cyclic SGD explorer for 43 epochs,
one scheduled switch, then a SAM+Muon tail owning a fresh cosine anneal.

Both full-time-SAM baselines are shown on every dataset: **SAM+SGD** is the
published recipe ([Foret et al., 2021](https://arxiv.org/abs/2010.01412))
re-run here, and **SAM+Muon** is this project's own stronger combination.
Cells give final test accuracy (5-seed mean ± sd) and, below, the wall-clock
median to the hardest target that any method reaches, with the fraction of
seeds that reach it.

| dataset (target) | Temperon | SAM+SGD *(published, re-run)* | SAM+Muon *(ours, full-time)* |
|---|---|---|---|
| **CIFAR-100** (0.82) | **0.8295 ± 0.0034**<br>4364s · 5/5 | 0.8232 ± 0.0031<br>5356s · 4/5 | 0.8292 ± 0.0021<br>6624s · 5/5 |
| **Tiny ImageNet** (0.69) | 0.7003 ± 0.0033<br>**6602s** · 5/5 | **0.7027 ± 0.0032**<br>10935s · 5/5 | 0.6838 ± 0.0019<br>never |
| **CIFAR-10** (0.968) | **0.9695 ± 0.0008**<br>**4418s** · 5/5 | 0.9668 ± 0.0005<br>4646s · 1/5 | 0.9694 ± 0.0008<br>8052s · 5/5 |
| **SVHN** (0.98) | **0.9807 ± 0.0005**<br>8035s · 5/5 | 0.9798 ± 0.0004<br>6692s · 3/5 | 0.9804 ± 0.0003<br>**4820s** · 5/5 |

Read that as: Temperon **ties the best full-time-SAM recipe on accuracy
everywhere** (Welch p = 0.87 / 0.29 / 0.94 / 0.28) while reaching the hard
target **−34%** sooner on CIFAR-100 and **−40%** sooner on Tiny ImageNet. On
SVHN it does not win on time — the boundary case documented below. Against the
published SAM+SGD recipe specifically it is ahead on accuracy on three of four
datasets and reaches the target in more seeds on two.

The **allocation control** matters more than the amount: uniform periodic SAM
at equal-or-greater budget reaches only 0.8183 ± 0.0016 on CIFAR-100, i.e.
1.12pp below the hand-off (p=0.001) while spending more time.

Language model (GPT-2 124M, WikiText-103, 400M-token budget, 1 seed): the tail
arm matches full-time SAM (3.3051 vs 3.3101 val loss, inside the measured noise
floor) at **−29% wall-clock**, and beats it by **0.063 nats at equal
wall-clock**. Whether SAM is worth using at all for LM *pretraining* is a
separate question, and our answer is "marginally, and not under heavy data
repetition" — see [Phase 7](#phases).

## Relation to published results

Every baseline above was **re-run inside this pipeline** — same architecture,
data, augmentation, epoch budget, seeds and GPU. That is not laziness about
citing numbers, it is a requirement: the claim is about *wall-clock*, and
wall-clock cannot be compared across papers, implementations and hardware. An
accuracy taken from another paper has no time attached to it.

The published method in the table above is therefore SAM itself, re-run here
rather than quoted. For external calibration, the SAM paper
([Foret et al., 2021](https://arxiv.org/abs/2010.01412), Table 1) reports 12.8%
CIFAR-100 error for SAM and 16.1% for its SGD baseline on WideResNet-28-10 —
trained for **1800 epochs with AutoAugment**. This work trains a wide
ResNet-110 for **100 epochs** with flip/cutout/colour-jitter, so the two are
not comparable and no cross-paper accuracy claim is made. The useful read is a
sanity one: our in-pipeline SAM baselines land near that paper's 1800-epoch
*SGD* number while training ~18x fewer epochs, which is evidence that the
baselines being beaten here are not weak ones.

**Known gap.** The closest published *allocation* method, late-phase SAM
(below), has not yet been re-run in this pipeline; `arm_O` is not a faithful
stand-in for it, since it switches SAM on mid-cycle over a Muon-catapult base
rather than over plain SGD with a standard schedule. Until that arm exists,
the comparison against that specific paper is argued, not measured.

The published result this work builds on directly is **late-phase SAM**
([arXiv:2410.10373](https://arxiv.org/abs/2410.10373)), which showed that SAM
applied only late can match full SAM. That paper fixes the switch point, uses
SGD only, and reports no wall-clock recipe; the contribution here is the
allocation *shape* (a tail owning a fresh anneal, versus the same budget spread
uniformly or bolted mid-cycle), a Muon refiner, and the wall-clock framing.
Closest prior art for the cheap-to-expensive hand-off itself is SWATS
([Keskar & Socher, 2017](https://arxiv.org/abs/1712.07628)), which switches
Adam to SGD on a convergence trigger rather than allocating a sharpness budget.

## Honest scope

Measured limits, stated because they define where the method applies:

- **The cheap optimizer wins below its own ceiling.** On CIFAR-100 plain SGD
  reaches 0.80 in 1747s; the hand-off needs 4096s. SGD never reaches 0.82.
- **The loss band is narrow and predictable**: it is the last ~1pp below the
  cheap method's ceiling, in every dataset tested.
- **Saturated tasks (SVHN) show no time win** — there the full recipe's early
  cycles already reach the frontier.
- **Data scarcity is not the same axis as task difficulty.** Forcing 20 passes
  over a 20M-token slice made SAM *worse*, not better (Phase 7b).

## Repository map

This repo is the record of an eight-phase study, not a single experiment. Old
arms are kept deliberately: the negative results are the controls that support
the main claim.

```
src/eos_switch/
  probes/        # curvature & stability measurement (fp32, no autocast/TF32)
  optimizers/    # muon, SAM, sharp_muon, OptiRoulette adapter, state transfer
  controllers/   # fixed / cyclic / sam_catapult / handoff / eos_switch / ...
  data/          # GPU-resident CIFAR/SVHN/TinyImageNet with on-GPU augmentation
  train/         # models, seeding, training loop (drives SAM's two passes)
  report/        # run logging, plots, markdown report
configs/         # one YAML per experiment arm; configs/bench/ = the 4x5 matrix
configs/bench/   # generated by experiments/gen_bench_configs.py
experiments/     # run.py, launch_grid.py, analyze_5seed.py, lm/, glue/
tests/           # pytest, CPU-only
third_party/     # vendored OptiRoulette source (MIT, same author)
```

Results land in `results/<run_name>/seed<seed>/`: `epochs.csv`, `steps.jsonl`,
`probes.jsonl`, `summary.csv` (first-hit milestones), `meta.json` (config +
switch events), `metrics.json` (final test metrics), `nvidia_smi.txt`.

**On the name `eos_switch`.** The Python package, the container and the working
directory still carry the project's original name: the study began as an
investigation of *edge-of-stability-aware optimizer switching*. That hypothesis
failed (see Phase 3 and Phase 4 below) and what survived — a single scheduled
hand-off into a SAM-owned anneal — is a different method, named Temperon. The
internal name is kept so that the archived code matches the runs that produced
the published numbers; renaming it would gain nothing and break that match.

## Phases

Each phase's conclusion, including the failures — they are why the final method
looks the way it does.

- **Phase 0-1** — scaffolding, Docker, OptiRoulette adapter, probe library
  (lambda_max, batch sharpness, preconditioned sharpness, stability margin).
- **Phase 2** — ablation arms A-E. Warmup explains most of the apparent gain;
  the catapult hypothesis was not supported.
- **Phase 3** — EoS-aware switching controller (arm F), three iterations. All
  failed to beat a tuned baseline: either inert or plateau→thrash→diverge.
- **Phase 4** — per-layer Muon pool with a UCB bandit (arm G). Best controlled
  arm of its era but still below the tuned baseline; per-layer specialization
  was weak and seed-dependent.
- **Phase 5** — SAM+Muon-catapult found by 2x2 ablation: SAM buys
  generalization, the catapult buys epoch-speed, and the two are separable.
  Single-pass sharpness surrogates (SharpMuon) failed to recover SAM.
- **Phase 6** — **the paper.** Sharpness-budget allocation: `arm_O` (tail-SAM
  on a catapult continuation) *fails*, `arm_P` (hand-off into a fresh anneal)
  wins. That contrast is the evidence that *what precedes the tail* matters.
  Controls: uniform periodic SAM, and MSAM as a literature rival (eliminated).
- **Phase 7** — LM transfer (`experiments/lm/`). The allocation law transfers;
  SAM itself is marginal for pretraining at 8 passes and harmful at 20.
- **Phase 8** — LM fine-tuning on GLUE (`experiments/glue/`), where SAM's
  language-model gains are established in the literature and its 2x cost is
  the adoption barrier. In progress.

## Setup

Host requirements: Docker (Desktop) with the NVIDIA container runtime. Python,
uv and all dependencies live in the image.

```bash
docker compose build dev          # PyTorch >= 2.7 on the cu128 index (sm_120)
docker compose up -d dev
docker compose exec dev uv sync --frozen
docker compose exec dev python experiments/check_env.py
```

If you see `no kernel image is available for execution on the device`, the
installed torch wheels lack sm_120 support — rebuild against the cu128 index.

`./eos.sh <command>` forwards a command into the running container (starting
it if needed), which is how every run in this repo was launched. On Windows
PowerShell, `.\eos.ps1 <command>` is the identical counterpart.

## Running

```bash
./eos.sh python experiments/run.py --config configs/arm_P_handoff.yaml
./eos.sh python -m pytest tests/ -q
```

On Windows PowerShell, the same two commands:

```powershell
.\eos.ps1 python experiments/run.py --config configs/arm_P_handoff.yaml
.\eos.ps1 python -m pytest tests/ -q
```

Batch size is fixed at 128 in all comparative vision arms
(Edge-of-Stochastic-Stability depends on it).

See **[REPRODUCE.md](REPRODUCE.md)** for the command behind each table above.

## Planned library

The method will ship as a small standalone library in this repository
(`pip install temperon`, `import temperon`), whose CI asserts that it
reproduces the trajectories of the research code archived here. It is not
published yet; until then this repository is the reference implementation.

## Citing and license

MIT (see [LICENSE](LICENSE)). Citation metadata in
[CITATION.cff](CITATION.cff). The vendored OptiRoulette under `third_party/`
is MIT by the same author.
