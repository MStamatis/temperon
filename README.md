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
Cells give final test accuracy (5-seed mean ± sd) and, below, the time to the
hardest target that any method reaches, with the fraction of seeds reaching it.

**How the times are obtained.** Not by reading a stopwatch on the training
runs. Raw wall-clock on a workstation is not a property of the method: a GPU
that is also driving a monitor loses 20–28% of every epoch to the desktop, and
we measured the same optimizer costing anywhere from 14.3 to 21.9 s/epoch
depending only on when the grid happened to be scheduled. Each figure below is
therefore **epochs-to-target × calibrated seconds-per-epoch** — the first
factor measured from the runs and immune to load, the second measured
separately on an idle GPU by
[`calibrate_cost.py`](experiments/calibrate_cost.py). That is also the only
form another reader can use, since their seconds-per-epoch will differ from
ours. Raw wall-clock is kept alongside in
[`analysis_costmodel.txt`](results/latesam/analysis_costmodel.txt).

| dataset (target) | Temperon | SAM+SGD *(published, re-run)* | SAM+Muon *(ours, full-time)* |
|---|---|---|---|
| **CIFAR-100** (0.82) | **0.8295 ± 0.0034**<br>4643s · 5/5 | 0.8232 ± 0.0031<br>4679s · 4/5 | 0.8292 ± 0.0021<br>7190s · 5/5 |
| **Tiny ImageNet** (0.69) | 0.7003 ± 0.0033<br>**6359s** · 5/5 | **0.7027 ± 0.0032**<br>9333s · 5/5 | 0.6838 ± 0.0019<br>never |
| **CIFAR-10** (0.968) | **0.9695 ± 0.0008**<br>**4849s** · 5/5 | 0.9668 ± 0.0005<br>4990s · 1/5 | 0.9694 ± 0.0008<br>7371s · 5/5 |
| **SVHN** (0.98) | **0.9807 ± 0.0005**<br>6948s · 5/5 | 0.9798 ± 0.0004<br>7029s · 3/5 | 0.9804 ± 0.0003<br>**5084s** · 5/5 |

Read that as: Temperon **ties the best full-time-SAM recipe on accuracy
everywhere** (Welch p = 0.87 / 0.29 / 0.94 / 0.28) while reaching the hard
target **−35%** sooner on CIFAR-100, **−34%** on CIFAR-10 and **−32%** on Tiny
ImageNet. On SVHN it does not win on time — the boundary case documented below.
Against the published SAM+SGD recipe specifically it is ahead on accuracy on
three of four datasets (+0.63pp p=0.016, +0.27pp p<0.001, +0.09pp p=0.015) and
reaches the target in more seeds on two.

Against that same recipe it is **level on time — within 3% on all four
datasets** — which is the more interesting number, because it is not a
coincidence. See [the exchange rate](#the-exchange-rate) below.

The **allocation control** matters more than the amount: uniform periodic SAM
at equal-or-greater budget reaches only 0.8183 ± 0.0016 on CIFAR-100, i.e.
1.12pp below the hand-off (p=0.001) while spending more time.

Language model (GPT-2 124M, WikiText-103, 400M-token budget, 1 seed): the tail
arm matches full-time SAM (3.3051 vs 3.3101 val loss, inside the measured noise
floor) at **−29% wall-clock**, and beats it by **0.063 nats at equal
wall-clock**. Whether SAM is worth using at all for LM *pretraining* is a
separate question, and our answer is "marginally, and not under heavy data
repetition" — see [Phase 7](#phases).

## The exchange rate

Skipping SAM before the switch buys a fixed credit: `43 × (cost of a SAM epoch
− cost of a cheap epoch)`. What the run does with that credit is a choice, and
it is the whole method.

Handing the tail to Muon spends it. Muon costs **1.50×** a SAM+SGD epoch —
1.505 / 1.509 / 1.495 / 1.497 across the four datasets, a constant we did not
expect to be that flat — and the premium very nearly cancels the credit:

| dataset | tail | credit | premium | net time | accuracy |
|---|---|---|---|---|---|
| CIFAR-100 | Muon | −1439s | +1294s | **−2.0%** | **+0.63pp** |
| CIFAR-10 | Muon | −1470s | +1319s | **−2.1%** | **+0.27pp** |
| SVHN | Muon | −2151s | +1907s | **−2.4%** | **+0.09pp** |
| Tiny ImageNet | SGD | −2942s | **+0** | **−32.6%** | −0.24pp |

*(vs the full-time SAM+SGD baseline of that dataset;
[`analyze_allocation.py`](experiments/analyze_allocation.py))*

Three datasets land within half a point of each other at −2%, which is not a
coincidence: they are paying the same 1.50× premium out of the same credit.
Tiny ImageNet keeps an SGD tail — the same optimizer as its baseline, so its
premium is zero by construction — and the credit survives as a third off the
wall-clock instead. The accuracy column tracks it exactly: banked as time,
accuracy goes slightly down; spent on Muon, it comes back as accuracy.

So the method is not "SAM, but faster". It converts a fixed allocation credit
into accuracy at a near-constant rate, and the rate is predictable before you
run anything: measure your refiner's cost per epoch, compare it to the credit,
and you know which side of the trade you are on.

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

### The closest rival, measured

The published result this work builds on directly is **late-phase SAM**
([arXiv:2410.10373](https://arxiv.org/abs/2410.10373)), which showed that SAM
applied only late can match full SAM. It is re-run here at our own tuned
hyper-parameters and a matched SAM budget (`configs/latesam/`, 5 seeds), and it
does two things at once — one for us and one against.

**For the thesis.** It reproduces full-time SAM+SGD to within noise (0.8234 ±
0.0011 vs 0.8232 ± 0.0031, p=0.896) while paying for SAM on 57 of 100 epochs
instead of all of them: 3022s against 4679s, **−35%**. An independent method,
in our pipeline, arriving at the allocation law from the other direction.

**Against us.** At the 0.82 target it is **34% faster than Temperon** (3022s vs
4643s). If 0.82 is what you need on CIFAR-100, use it, not this.

What it cannot do is go higher. Its accuracy is 0.8234 ± 0.0011 — a tighter
band than any other arm here — and it reaches 0.83 in **0 of 5 seeds**, as does
every other SGD-refined arm. Temperon reaches 0.83 in 3 of 5 at 4870s. The two
methods are not competing for the same point on the frontier.

### Which part of Temperon earns that

Temperon differs from late-phase SAM in two ways at once — a Muon refiner, and
a cyclic explorer handing over to a fresh anneal — so `arm_P_sgdtail` swaps
*only* the refiner back to SGD and changes nothing else. It lands at 0.8210 ±
0.0006. The verdict is unambiguous in both directions:

- **The Muon refiner is the accuracy contribution**: +0.85pp (p=0.005), with
  every other element of the method held fixed.
- **The allocation shape is not.** At a fixed SGD refiner our shape is 0.25pp
  *worse* than a plain cosine switched mid-schedule (p=0.005) and needs seven
  more epochs to reach 0.82. Earlier drafts of this README claimed the shape as
  a contribution; that claim was wrong and has been removed.

The refiner sets the tier and nothing else does: every SGD-refined arm lands in
0.8210–0.8234, every Muon-refined arm in 0.8292–0.8295.

Closest prior art for the cheap-to-expensive hand-off itself is SWATS
([Keskar & Socher, 2017](https://arxiv.org/abs/1712.07628)), which switches
Adam to SGD on a convergence trigger rather than allocating a sharpness budget.

## Honest scope

Measured limits, stated because they define where the method applies:

- **The cheap optimizer wins below its own ceiling.** On CIFAR-100 plain SGD
  reaches 0.80 in 1567s; Temperon needs 4416s. SGD never reaches 0.82.
- **A cheaper allocation wins below *its* ceiling too.** Late-phase SAM reaches
  0.82 34% sooner than Temperon. The win here starts above 0.8234, which is
  where every SGD-refined method stops.
- **The loss band is narrow and predictable**: it is the last ~1pp below the
  cheaper method's ceiling, in every dataset tested.
- **Saturated tasks (SVHN) show no time win against SAM+Muon**, which reaches
  0.98 by epoch 45 there — the accuracy is worth so little on an easy dataset
  that the early cycles already arrive. Against the published SAM+SGD recipe
  SVHN behaves like the others (−2.4%); it is only the Muon baseline it cannot
  outrun.
- **Data scarcity is not the same axis as task difficulty.** Forcing 20 passes
  over a 20M-token slice made SAM *worse*, not better (Phase 7b).
- **SAM does not help RoBERTa fine-tuning at any ρ we tested.** ρ=0.05 hurts
  (RTE −2.24pp, p=0.046); ρ=0.02 and ρ=0.01 stop the damage but never turn it
  into a gain (RTE +0.14pp p=0.845, MRPC +0.27pp p=0.519). The Phase 8 result
  is that the *tail* beats *full-time* SAM at −36% time — not that SAM helps.
- **Part of that Phase 8 margin was ρ, and we checked.** Re-running the tail at
  ρ=0.02, where full-time SAM is no longer harmful, shrinks the advantage on
  every task and removes it on one:

  | tail vs full | ρ=0.05 | ρ=0.02 |
  |---|---|---|
  | MRPC | +1.31pp (p=0.002) | +1.11pp (p=0.023) |
  | STSB | +0.44pp (p=0.004) | +0.24pp (p=0.063) |
  | RTE | +2.74pp (p=0.047) | −0.43pp (p=0.516) |

  So the **accuracy** advantage over full-time SAM is partly a ρ=0.05 artifact,
  clearly surviving only on MRPC. RTE's whole margin was the tail harming less
  — unsurprising for the task with the largest measured noise floor (0.0181,
  dev set 277 examples), flagged as fragile before this was run.

  What survives at both ρ is the **equivalence at a third of the cost**: the
  tail is never worse than full-time SAM on any task at either ρ, while paying
  for SAM on 30% of steps (−36% wall-clock). That is the same shape as the
  Phase 7 LM result, and it is the claim this repository makes for fine-tuning
  — not that the tail is more accurate.

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
