# Phase 1 baseline and Phase 2 research

[Issue #1](https://github.com/truong51972/vehicle-detection/issues/1) spans both
phases. It remains open after the Phase 1 PR; its full-study requirements are not
completed by a small integration pilot. PR #2 references the issue without an
automatic closing keyword.

## Phase 1: verified notebook pilot

Completion means these concrete artifacts and behaviors are verified:

- [x] Two executed, inspectable notebooks with real BDD2018 data and inline figures.
- [x] Official BDD100K/Scalabel conversion followed by native Ultralytics conversion,
  with explicit coordinate adaptation, class mapping and selected-box roundtrips.
- [x] Safe offline export: untouched original reference, JPEG95-only raw control,
  gamma+JPEG95, aligned IDs/labels/shapes/splits and verified export repair/reuse.
- [x] Independent actual native YOLO fine-tuning and best/last checkpoints on
  64 train / 16 tune images for three epochs, with 16 exposed integration test images.
- [x] Native evaluation of all nine trained/input pairs and inline controlled results.
- [x] Fresh-kernel checkpoint and completed-evaluation reuse, with identity guards
  and explicit deep verification / forced reevaluation options.
- [x] Full-run annotation/header preflight and sampled per-variant storage estimates;
  helper tests, notebook smoke, lint and source/wheel builds.

This verifies feasibility and a reproducible starting workflow. Full fine-tuning,
robustness benchmarks and research-grade accuracy claims are not Phase 1 gates.
The PR remains draft for owner review of the recorded JPEG closeout evidence.

## Phase 2: full BDD100K fine-tuning and controlled benchmark

Remaining work is tracked here under Issue #1 and is deliberately unchecked:

- [ ] Freeze a study protocol, compute/storage budget and untouched final test set.
  Exclude all exposed IDs in `notebooks/test_exposure.json`; make model/recipe
  choices on train-derived tune only.
- [ ] Run the full original-raw baseline first with appropriate training duration;
  preserve independent native checkpoints and evaluation identities.
- [ ] Add the same-encoding raw control and selected preprocessing variants at full
  scale, with aligned IDs/labels/splits and separate offline preparation/storage costs.
- [ ] Calibrate adaptive thresholds on train/tune, freeze parameters and evaluate
  the filter/skip policy. Existing defaults are uncalibrated heuristics.
- [ ] Evaluate adverse conditions and robustness, including carefully scoped natural
  condition slices and synthetic stress tests. Avoid tuning on final test outcomes.
- [ ] Run the frozen-checkpoint input comparisons and separately trained model
  comparisons with native AP, class/condition support, confusion/PR and failure analysis.
  Add representative mean/p50/p95 costs with warm-up and synchronized CUDA timing;
  measure online end-to-end latency separately if claiming deployment performance.
- [ ] Repeat independent seeds and quantify uncertainty (including paired/bootstrap
  analysis where appropriate); report accuracy/cost tradeoffs and limitations.

Create separately scoped implementation work for these items before executing a
full study. Do not mark Issue #1 complete until the agreed research requirements
and their evidence have been reviewed.
