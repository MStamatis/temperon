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
