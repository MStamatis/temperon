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

## What Temperon does

One training run, two stages, one scheduled switch. On vision (the reference
recipe, `configs/arm_P_handoff.yaml`):

1. **Explorer — epochs 1–43: plain SGD.** Nesterov SGD (lr 0.1, wd 5e-4) on a
   cyclic cosine schedule. No SAM anywhere in this stage, so every epoch costs
   the cheap price (~17s vs ~76s on CIFAR-100). The particular schedule is not
   load-bearing: a single cosine matches the cyclic one (p=0.61).
2. **The switch — epoch 43, scheduled, not adaptive.** One hand-off at a fixed
   43% of the epoch budget. Momentum carries over; the incoming optimizer gets
   a 200-step LR warmup and SAM's ρ ramps from zero over 400 steps, so the
   switch never shocks the loss. The switch point is a measured quality/cost
   dial: moving it to epoch 80 lands at 0.8249 ± 0.0022 (−0.42pp vs full
   SAM+Muon, p=0.015) for 43% less total compute (`arm_P_handoff_e80`).
3. **Refiner — epochs 44–100: SAM+Muon owning a fresh cosine anneal.** Muon
   (lr 0.01, wd 0.2) wrapped in SAM's two-pass ascent–descent (ρ=0.05),
   annealed to zero over the remaining 57 epochs. All of the SAM budget is
   spent here, and this stage is the measured accuracy contribution (+0.85pp
   over the same run with an SGD refiner).

Two rules generalize across every task we measured: **spend SAM only in the
tail**, and **let the tail own the entire final anneal** — switching mid-decay
(arm O) loses, switching at the decay boundary wins. The base optimizer of the
tail is per-task: Muon where it buys a tier (CIFAR-10/100, SVHN), SGD where it
does not (Tiny ImageNet), Muon under WSD for GPT-2 pretraining (SAM switches
on at 70% of steps, nothing else changes), AdamW for GLUE fine-tuning.

Everything runs inside a Docker container on a single NVIDIA RTX 5090
(Blackwell, sm_120, 32 GB VRAM). All numbers below come from runs in this
repository; see [REPRODUCE.md](REPRODUCE.md) for the exact commands.

## Headline results

Vision, 5 seeds per arm (`42, 1181241943, 958682846, 271828, 314159`),
wide ResNet-110 (the He et al. depth-110 topology at 4× width: stages
64/128/256, 27.6M parameters vs the original's 1.7M — so no external
"ResNet-110" number is comparable to these), 100 epochs. "Hand-off" = cyclic SGD explorer for 43 epochs,
one scheduled switch, then a SAM+Muon tail owning a fresh cosine anneal.

Both full-time-SAM baselines are shown on every dataset: **SAM+SGD** is the
published recipe ([Foret et al., 2021](https://arxiv.org/abs/2010.01412))
re-run here, and **SAM+Muon** is this project's own stronger combination.
**Late-phase SAM** ([arXiv:2410.10373](https://arxiv.org/abs/2410.10373)) is
the closest published rival, re-run at a matched SAM budget on all four
datasets — it belongs in this table precisely because it beats us in some
cells.
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

| dataset (target) | Temperon *(ours)* | late-phase SAM *(rival, re-run)* | SAM+SGD *(published, re-run)* | SAM+Muon *(ours, full-time)* |
|---|---|---|---|---|
| **CIFAR-100** (0.82) | **0.8295 ± 0.0034**<br>4643s · 5/5 | 0.8234 ± 0.0011<br>**3022s** · 5/5 | 0.8232 ± 0.0031<br>4679s · 4/5 | 0.8292 ± 0.0021<br>7190s · 5/5 |
| **Tiny ImageNet** (0.69) | 0.7003 ± 0.0033<br>6359s · 5/5 | 0.7020 ± 0.0028<br>**5750s** · 5/5 | **0.7027 ± 0.0032**<br>9333s · 5/5 | 0.6838 ± 0.0019<br>never |
| **CIFAR-10** (0.968) | **0.9695 ± 0.0008**<br>4849s · 5/5 | 0.9688 ± 0.0006<br>**3121s** · 4/5 | 0.9668 ± 0.0005<br>4990s · 1/5 | 0.9694 ± 0.0008<br>7371s · 5/5 |
| **SVHN** (0.98) | **0.9807 ± 0.0005**<br>6948s · 5/5 | 0.9798 ± 0.0005<br>**5003s** · 3/5 | 0.9798 ± 0.0004<br>7029s · 3/5 | 0.9804 ± 0.0003<br>5084s · 5/5 |

Read that as: Temperon **ties the best full-time-SAM recipe on accuracy
everywhere** (Welch p = 0.87 / 0.29 / 0.94 / 0.28) while reaching the hard
target **−35%** sooner on CIFAR-100, **−34%** on CIFAR-10 and **−32%** on Tiny
ImageNet. On SVHN it does not win on time — the boundary case documented below.

The rival column holds the fastest time in every row, and what the rows mean
splits three ways. On CIFAR-100 and CIFAR-10 the mid target is not where
Temperon competes: the rival never reaches the tier above it — 0.83 in 0/5
seeds (Temperon 3/5), 0.97 in 0/5 (Temperon 2/5, and only Muon-refined arms
at all) — and that tier is the win. On SVHN the rival's 5003s is the median
of only the 3 seeds that reach 0.98 at all, its final sits 0.09pp below
Temperon (p=0.025), and SAM+Muon reaches 0.98 by epoch 45 in every seed — the
robust choice there. On Tiny ImageNet there is no tier above (SAM+Muon
*loses* there) and the rival's row is a plain defeat for us — see
[the closest rival](#the-closest-rival-measured).
Against the published SAM+SGD recipe specifically Temperon is ahead on accuracy
on three of four datasets (+0.63pp p=0.016, +0.27pp p<0.001, +0.09pp p=0.015)
and reaches the target in more seeds on two.

Against that same recipe it is **level on time — within 3% on all four
datasets** — which is the more interesting number, because it is not a
coincidence. See [the exchange rate](#the-exchange-rate) below.

The **allocation control** matters more than the amount: uniform periodic SAM
at equal-or-greater budget reaches only 0.8183 ± 0.0016 on CIFAR-100, i.e.
1.12pp below the hand-off (p=0.001) while spending more time.

Language model (GPT-2 124M, WikiText-103, 400M-token budget, 1 seed,
back-to-back runs on the same GPU):

| arm | val loss | ppl | wall-clock |
|---|---|---|---|
| Muon, no SAM | 3.3182 | 27.61 | **3079s** |
| SAM+Muon, full-time | 3.3101 | 27.39 | 5335s |
| **SAM tail** *(ours = the late-phase allocation)* | **3.3051** | **27.25** | 3796s · **−29%** |

The tail matches full-time SAM (the 0.005 gap is inside the measured noise
floor) at **−29% wall-clock**, and beats it by **0.061 nats at equal
wall-clock**. There is no separate rival column for the LM, and the reason is
worth stating: the tail arm here *is* the late-phase allocation — one WSD
schedule, no explorer, no restarts, SAM switched on for the final 30% of steps
— run with Muon as the base optimizer. On this task the two methods coincide,
and the merged form is what wins; what this phase adds is that the allocation
law transfers to LM pretraining, to Muon, and to a smaller SAM budget (30%
here vs 57% in the vision runs). Whether SAM is worth using at all for LM
*pretraining* is a separate question, and our answer is "marginally, and not
under heavy data repetition" — see [Phase 7](#phases).

## The exchange rate

Skipping SAM before the switch buys a fixed credit: `43 × (cost of a SAM epoch
− cost of a cheap epoch)`. What the run does with that credit is a choice, and
it is the whole method.

Handing the tail to Muon spends it. Muon costs **1.50×** a SAM+SGD epoch —
1.504 / 1.508 / 1.495 / 1.497 across the four datasets, a constant we did not
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

**For the thesis.** It reproduces — or beats — full-time SAM+SGD on all four
datasets while paying for SAM on 57 of 100 epochs instead of all of them:
within noise on CIFAR-100 (0.8234 vs 0.8232, p=0.896), Tiny ImageNet (0.7020
vs 0.7027, p=0.720) and SVHN (0.9798 vs 0.9798, p=0.846), and clearly above
it on CIFAR-10 (+0.20pp, p<0.001, where the published recipe reaches the
0.968 target in only 1 of 5 seeds). An independent method, in our pipeline,
arriving at the allocation law from the other direction.

**Against us.** It is the fastest measured route to the mid target on every
dataset: 0.82 on CIFAR-100 **34% sooner** than Temperon (3022s vs 4643s),
0.968 on CIFAR-10 **36% sooner** (3121s vs 4849s, though in 4 of 5 seeds),
0.69 on Tiny ImageNet 10% sooner, and at SVHN's 0.98 its 3-of-5 median
(5003s) numerically edges even SAM+Muon's all-seed 5084s. If the mid target
is all you need, use it, not this.

What it cannot do is reach the tier the Muon refiner buys. On CIFAR-100 it
never reaches 0.83 (0/5 seeds; Temperon 3/5, at 4870s). On CIFAR-10 it never
reaches 0.97 (0/5; Temperon 2/5 — a target only Muon-refined arms touch at
all). On SVHN its final sits 0.09pp below Temperon (p=0.025). On three of
four datasets the two methods are not competing for the same point on the
frontier — the exception is next.

**On Tiny ImageNet it wins outright, and we say so.** There it matches
Temperon's final accuracy (0.7020 ± 0.0028 vs 0.7003 ± 0.0033, p=0.42) and
reaches every late target sooner at calibrated cost: 0.68 at 5344s vs 6156s,
0.69 at 5750s vs 6359s, 0.70 in 4/5 seeds vs 3/5. The reason is structural.
Tiny ImageNet is the one dataset where the Muon refiner does not help — full
SAM+Muon lands 1.8pp *below* both, at 0.6838 — so Temperon's tail there is
plain SAM+SGD and there is no higher tier for it to retreat to. Where the
expensive refiner buys nothing, the simpler allocation is the better method;
what Temperon keeps on that dataset is the early game (0.65 at 1322s vs 4533s,
the cyclic explorer's head start) and nothing at the top.

Beyond vision the rivalry dissolves rather than continues: the LM and GLUE
tail arms *are* the late-phase allocation — a single schedule with SAM
switched on late, no explorer, no restarts — run with our optimizers and
budget, so there is no separate rival arm to compare against there. See the
language-model paragraph under [Headline results](#headline-results).

### Which part of Temperon earns that

Temperon differs from late-phase SAM in two ways at once — a Muon refiner, and
a cyclic explorer handing over to a fresh anneal — so two ablations swap one
piece each while holding everything else fixed: `arm_P_sgdtail` puts an SGD
refiner behind our explorer (0.8210 ± 0.0006), and `arm_T_singlecos` keeps the
Muon refiner but flattens the explorer's four warm restarts into a single
cosine (0.8285 ± 0.0027). The verdict is unambiguous in every direction:

- **The Muon refiner is the accuracy contribution**: +0.85pp (p=0.005), with
  every other element of the method held fixed.
- **The allocation shape is not.** At a fixed SGD refiner our shape is 0.25pp
  *worse* than a plain cosine switched mid-schedule (p=0.005) and needs seven
  more epochs to reach 0.82. Earlier drafts of this README claimed the shape as
  a contribution; that claim was wrong and has been removed.
- **Neither are the restarts.** With the Muon refiner fixed, the single-cosine
  explorer matches the cyclic one on accuracy (+0.10pp for cyclic, p=0.61) and
  hits 0.80 and 0.82 at the same epochs (91 and 94). It also ties full-time
  SAM+Muon (−0.07pp, p=0.66). Any cheap schedule that lands the hand-off works;
  the restarts are an implementation choice, not a mechanism.

The refiner sets the tier and nothing else does: every SGD-refined arm lands in
0.8210–0.8234, every Muon-refined arm in 0.8285–0.8295.

Closest prior art for the cheap-to-expensive hand-off itself is SWATS
([Keskar & Socher, 2017](https://arxiv.org/abs/1712.07628)), which switches
Adam to SGD on a convergence trigger rather than allocating a sharpness budget.

## Honest scope

Measured limits, stated because they define where the method applies:

- **The cheap optimizer wins below its own ceiling.** On CIFAR-100 plain SGD
  reaches 0.80 in 1567s; Temperon needs 4416s. SGD never reaches 0.82.
- **A cheaper allocation wins below *its* ceiling too.** Late-phase SAM
  reaches the mid target sooner than Temperon on every dataset (−34% on
  CIFAR-100, −36% on CIFAR-10). The win here starts above the SGD-refined
  ceiling — 0.8234 on CIFAR-100, 0.9688 on CIFAR-10 — which no arm without a
  Muon refiner crosses.
- **Where the expensive refiner buys nothing, the rival wins outright.** On
  Tiny ImageNet SAM+Muon is 1.8pp *worse* than SAM+SGD, so there is no higher
  tier to reach — and late-phase SAM matches Temperon's accuracy there (p=0.42)
  while reaching 0.68/0.69/0.70 sooner (−13%/−10%/−7% at calibrated cost).
  Temperon's win condition is a dataset where the refiner actually buys a tier;
  Tiny ImageNet is the measured counterexample.
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
src/eos_switch/  # the project's original name, kept on purpose -- see the note below
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
