> Status update (2026-11-12): v4 implementation and exploratory full training/evaluation are complete. The matched Raw comparison does not improve overall AP. These are local issue documents, not newly created GitHub issues. Proposed independent pilot/confirmation gates remain unperformed; see [results](../RESULTS.md).

# [Epic / v4] Explore Canny-guided edge detail and paired YOLO26n training/evaluation

## Goal

Build v4 as an edge-detection-focused experiment while keeping the independent
notebook style of v0–v3. Investigate whether Gaussian-smoothed Canny support and
bounded luminance detail improve five-class vehicle detection. Preserve failed
outcomes; v4 is an experiment identity, not a validated improvement.

## Implementation available locally

- `notebooks/experiments/v4/{01_experiments,02_train_prepare,03_train,04_eval}.ipynb`
  and `v4.yaml`, matching current main's lifecycle and native training recipe.
- Minimal `canny_edges`, bounded `canny_detail` and pixel-only adaptive routing in
  `src/vehicle_detection/preprocess.py`; shared offline export and native APIs reused.
- Four-panel read-only pixel preview and 13-cell Raw-vs-policy comparison support.
- No real preview, preparation, training, inference or AP claim has been made.

## Work

- [ ] Complete bounded exploration and one independent confirmation only if justified.
- [ ] Freeze one evidence-backed policy, or document a negative conclusion and stop.
- [ ] Prepare/train v4 only after the explicit scientific decision; preserve v0–v3.
- [ ] Evaluate the matched v4 pair and report the Raw/v4 2×2 only in `compare.ipynb`.
- [ ] Commit concise evidence, hashes and limitations; update v4 README honestly.

## Boundaries / acceptance

- Notebook-first `01 → 02 → 03 → 04`; no upstream notebook execution, custom
  trainer, policy-search framework or new experiment registry.
- Use exact frozen five-class IDs, labels, negatives, geometry and JPEG95; final
  holdout sealed. Runtime routing uses pixels only, never labels/scenario metadata.
- Current main recipe: pinned YOLO26n, seed 2026, S20 9,781 training images,
  calibration 6,985 for checkpoint selection, development 13,975 for reporting,
  full fine-tune, 50 epochs, SGD, batch 32, 640, geometry-only augmentation.
  Historical freeze=11/20-epoch results are distinct evidence.
- Native mAP50:95 primary; per-class/cohort support, false positives, halos and
  matched CPU mean/p95 mandatory. A negative conclusion is acceptable.
- Focused operator/export tests, Ruff and fresh-kernel notebook checks pass.

**References:** #41, #45, #47; `notebooks/experiments/v4/README.md`.
**Execution status:** implementation only; research/training/evaluation pending.
