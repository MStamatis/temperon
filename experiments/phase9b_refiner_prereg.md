# Phase 9b pre-registration: can the refiner choice be read at the hand-off?

Written 2026-07-29, before the 5-seed replay ran. This is the "who" question
— the surviving descendant of the optimizer-selection line — restricted to
the one decision where optimizer identity measurably matters: does the tail
go to SAM+Muon or SAM+SGD?

**Disclosure**: before locking this file, seed42 envelope values at epochs
21/43 were previewed while designing the signals (they are what showed that
the naive gap points the wrong way). The other four seeds are unseen. With
n=4 datasets this was always a consistency screen, not a hypothesis test;
the seed42 peek weakens it further and is stated so it cannot be unstated.

## Labels (measured, 5-seed finals, sammuon − samsgd)

- c100 **+0.60pp** → Muon tier YES
- c10  **+0.26pp** → YES
- svhn **+0.06pp** → YES (weak)
- tiny **−1.89pp** → NO

## Signals (locked; all from bench `muon` / `strongsgd` epochs.csv,
best-so-far envelopes, matched seeds, h = epoch 43 of 100, h2 = epoch 21)

Observable at hand-off time:
- **S1** gap: muon(h) − sgd(h)
- **S2** ratio: muon(h) / sgd(h)
- **S3** second-window relative progress: (muon(h)−muon(h2)) − (sgd(h)−sgd(h2))

Diagnostic only (uses information unavailable online — an upper bound on
whether the information exists at all):
- **D** muon(h) / sgd(final)

## Read-out (locked)

A signal SEPARATES if tiny's per-seed [min,max] does not overlap the union
of the other three datasets' ranges. Direction required for a usable rule:
tiny must sit LOW (Muon relatively weaker where it fails to buy the tier).

- **SIGNAL-CANDIDATE**: ≥1 observable signal separates in the required
  direction → justifies designing a live probe rule (still needs validation
  on datasets outside these four).
- **INVERTED**: a signal separates but in the opposite direction (tiny high).
  Recorded as an n=4 curiosity, NOT promoted to a rule — "Muon looks best
  early exactly where it fails late" is the kind of association four points
  cannot support.
- **NO-SIGNAL**: nothing separates. The refiner choice cannot be read from
  cheap early observables; it stays a per-task measured decision, and the
  package keeps `tail_optimizer` explicit.

If even D fails to separate, the stronger conclusion holds: the information
does not exist in the cheap trajectories at any price, mirroring the Phase 9
timing result.

---

## Outcome (added after the replay; the lock is commit d3746e9)

Replay executed 2026-07-29, `results/analysis_refiner.txt`, 4 datasets × 5
seeds. **NO-SIGNAL for a single-run rule — but the information exists, and a
two-run form works.**

- **S1 overlaps** (tiny [+0.178, +0.203] vs c100 up to +0.178 — touching).
- **S2 separates INVERTED**: tiny 1.41–1.51 vs everything else ≤ 1.30. Early
  Muon dominance is *largest* exactly where SAM+Muon later fails. Recorded as
  the n=4 curiosity the prereg said it would be; not promoted to a rule.
- **S3 overlaps.**
- **D separates cleanly in the required direction**: tiny 0.904–0.922 vs all
  others ≥ 0.958 — no overlap, threshold ~0.94. Muon's hand-off plateau as a
  fraction of the cheap optimizer's *final* accuracy is the discriminator:
  where Muon stalls below ~94% of the task's SGD ceiling, it will not buy a
  tier and the tail should stay SGD.

The unifying reading: the refiner answer is locked behind the same door as
the switch timing — it requires knowing where the final anneal lands, which
Phase 9 showed the cheap trajectory cannot forecast. **As a single-run online
decision the refiner choice is unreadable; as a two-run protocol it is
practical**: anyone who already has a full baseline run of the cheap
optimizer (the normal situation when tuning) can run a 43%-budget probe of
the expensive one and apply the D<0.94 test. Validation on datasets outside
these four is still required before shipping that as a package default;
until then `tail_optimizer` stays explicit.
