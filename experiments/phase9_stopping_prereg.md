# Phase 9 pre-registration: a calibrated stopping rule for the hand-off

Written 2026-07-29, before any replay result was computed. The offline replay
tool is `analyze_stopping.py`; this file locks the rule, the parameter grid,
and the read-out criteria first, in that order.

## Question

Temperon's one load-bearing hyper-parameter is where the hand-off happens
(`tail_frac`): 0.43 on vision, 0.70 on LM/GLUE, both hand-picked. Can a single
online rule, watching only the explorer's own progress curve, reproduce those
choices — removing the hyper-parameter and explaining why those values were
right?

## The rule (primary form, dimensionless)

At each eval point t during the cheap phase, over the best-so-far envelope of
the validation metric:

```
rate(t)     = slope of the metric over the trailing window of W evals
avg_rate(t) = (metric gain from t_min to t) / (t - t_min)

FIRE when  rate(t) < c * avg_rate(t)  for K consecutive evals,
subject to t in [t_min, t_max].
```

"Switch when the explorer's current rate falls below fraction `c` of its own
historical average rate." The rule is dimensionless — the same `c` can be
asked to work on CIFAR accuracy and LM loss (sign flipped) — which is the
property the manual fractions lack.

Secondary form (vision-only): FIRE when rate(t) < θ_credit, with θ_credit =
0.46pp / 37 epochs ≈ 0.0125 pp/epoch — the marginal value of one tail epoch
measured between the two c100 switch points we ran (epoch 43 → 0.8295, epoch
80 → 0.8249). This form carries physical units from the measured
quality-vs-tail-length curve; it is not expected to transfer beyond vision.

## Parameter grid (locked)

- window W: 5 and 8 evals
- patience K: 2 and 3
- c: {0.10, 0.15, 0.20, 0.25, 0.30, 0.40}
- guardrails: t_min = 15% of budget, t_max = 85%
- envelope: running max (vision acc) / running min (LM loss), to keep cyclic
  oscillation from firing the rule mid-cycle

## Replay substrates (all existing logs; zero GPU)

- **Vision, primary**: `strongsgd` bench trajectories (smooth single
  schedule — matches the arm-T single-cosine explorer Phase 9 will use),
  4 datasets × 5 seeds.
- **Vision, secondary**: `cyclicj` bench trajectories (the arm-P explorer),
  same grid, envelope-smoothed.
- **LM**: `lm_muon` evals.csv (pure Muon under WSD, eval every 250 steps).
- **GLUE: excluded** — 3-epoch fine-tuning gives too few eval points for any
  slope estimate; declared out of scope before running anything.

Known limitation, stated up front: the pure-cheap 100-epoch runs are a proxy
for "the explorer kept going". They have the full-budget schedule, not the
explorer's own 43-epoch one, so firing epochs are approximate. The offline
stage selects or kills the rule; only live GPU validation can confirm it.

## Read-out criteria (locked before results)

For a single (W, K, c) cell to count, the SAME cell must satisfy all of:

1. **c100 consistency (hard constraint)**: median firing epoch across seeds in
   [35, 55]. We *know* switching at 43 beats switching at 80 there
   (0.8295 vs 0.8249, and 0.83 in 3/5 vs 0/5 seeds); a rule that fires at
   ~75+ contradicts measurement and is dead regardless of anything else.
2. **LM consistency (hard constraint)**: firing point within [0.55, 0.80] of
   the token budget. The measured-good hand-off is at 0.70 (aligned to the WSD
   decay boundary); firing long before 0.55 or after 0.85 contradicts the
   Phase 7 result.
3. **Cross-seed stability**: within one dataset, seed-to-seed spread of the
   firing epoch ≤ 12 epochs (else the rule is noise-driven).

Outcomes, in the order they will be reported:

- **SUPPORTED**: ≥1 grid cell passes 1–3. The rule replaces the
  hyper-parameter on the evidence we have; its c10/svhn/tiny firing epochs
  become *predictions* for a later live run.
- **PREDICTIVE-ONLY**: cells pass 1–2 but their c10/svhn/tiny firing epochs
  scatter far from 43 (>±12 epochs). Not failure — the manual 0.43 was never
  shown optimal per-dataset — but the rule then *requires* the live run
  before any claim.
- **FAILED**: no cell passes the two hard constraints with one `c`. The
  honest conclusion is that one dimensionless threshold does not exist, the
  fixed schedule stands, and Phase 9 stops at this document.

## What is deliberately NOT claimed

- No oracle-gap estimate: with only two measured switch points on one
  dataset, the in-hindsight optimum per seed is unknowable offline. A
  switch-point sweep (GPU) would be needed first; it is not part of this
  stage.
- No claim about the refiner-choice decision (Muon vs SGD tail); that is a
  separate question with n=1 evidence (tiny) and is not tested here.
