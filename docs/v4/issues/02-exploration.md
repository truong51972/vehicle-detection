> Status update (2026-11-12): v4 implementation and exploratory full training/evaluation are complete. The matched Raw comparison does not improve overall AP. These are local issue documents, not newly created GitHub issues. Proposed independent pilot/confirmation gates remain unperformed; see [results](../RESULTS.md).

# [v4 / P1] Validate Canny edge support, bounded detail and pixel-only routing

## Goal

Test the mechanism behind Canny-guided detail before committing compute to S20
training. Edge visibility alone is not evidence of detector benefit.

## Work

- [ ] Inspect #30/#45/#47 consumed calibration groups and representative weak
  vehicle boundaries, road-marking false positives and small two-wheeler failures.
- [ ] Freeze exact unused exploration IDs (up to 512), an independent group-disjoint
  confirmation reserve, support counts, native metrics, compute budget and hashes
  before inference. Stop if provenance or support is insufficient.
- [ ] Keep four fixed arms: Raw; configured adaptive Canny-guided detail; ungated
  Canny-guided detail; bounded global-luminance-detail control with identical
  amount/sigma/protection/offset bound and mask=1. Do not use the unbounded legacy
  BGR `sharpen()` as a supposedly matched control. No parameter grid search.
- [ ] Use paired original IDs/GT and the same detector. Report five-class native
  mAP50:95/mAP50, class/subgroup AP and support, TP/FP examples, routing, edge
  density, changed pixels and matched one-thread CPU policy mean/p95.
- [ ] Freeze gates before measuring: >= +0.2 pp primary gain, no class worse than
  -0.5 pp, at least 30 GT in 20 source groups per protected motorcycle/bicycle class,
  CPU p95 <= 25 ms/image at original BDD resolution. State missing support explicitly.
- [ ] Advance at most one fixed candidate to one independent confirmation with
  unchanged gates. Stop weak directions; no tuning on confirmation/development.
- [ ] Save a concise decision and exact config/checkpoint hashes before preparation.

## Acceptance / boundaries

`01_experiments.ipynb` remains a read-only fixed-sample pixel preview, including
Skip examples; its 64 calibration images are not assumed unused or held-out.
Any new pilot execution belongs to an explicitly declared saved plan, not an
implicit Run-All side effect. No per-image AP, GT-based routing, synthetic-score
substitution or final-test access. Auto-threshold/multiscale Canny is a future
bounded hypothesis, not an unmeasured winner bundled into this search.

**References:** v4 parent issue; #30, #45, #47; `notebooks/experiments/v4/README.md`.
**Status:** pending measurements; all thresholds and gates are proposed, not validated.
