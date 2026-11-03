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
Phase 1 was merged in PR #2; its executed pilot evidence remains in that history.

## Phase 2: YOLO26n Raw / Adaptive study

[Parent #4](https://github.com/truong51972/vehicle-detection/issues/4) supersedes
this document's former full-data YOLO11 plan. Its sequential implementation issues
are:

- #5: audit and freeze four disjoint data roles; run untouched YOLO26n on Raw
  development validation, with semantic COCO→vehicle mapping and native metrics.
- #6: calibrate Adaptive using calibration only; create representative nested
  S10/S20 training subsets and matched Raw/Adaptive exports.
- #11: independently screen synthetic degradations/restoration on calibration
  with frozen YOLO26n metrics before proposing a supported v3 policy; preserve
  clean evaluation and v1/v2 evidence.
- #7: progressive partial fine-tuning of neck/head with backbone frozen; scale
  paired subsets only when development-validation evidence justifies it.
- #8: lock protocol and execute the final 3×2 model/input benchmark once.

The new notebook sequence lives in `notebooks/phase2/`. Notebook 01 handles the
data contract; notebook 02 calibrates Adaptive and exports matched nested pilots;
notebook 03 initially runs pretrained validation and will be extended with #7.
No optimizer steps are part of #5 or #6.

Official train supplies training pool, calibration and development validation.
Official val minus prior exposures and disclosed exclusions supplies final test.
Final-test pixels/performance stay closed during development. Image-content
leakage checks against final test remain deferred to its controlled release.

The two original notebooks and [PR #3 execution snapshot](PHASE2_RESULTS.md)
preserve historical YOLO11 evidence independently. They are not YOLO26 outcomes.
See [issue #5 evidence and limitations](ISSUE5_RESULTS.md) for the current protocol.
See [issue #6 evidence and limitations](ISSUE6_RESULTS.md) for nested subsets and
the frozen Adaptive policy. Quality proxies alone do not establish detection gains.
The [PR review follow-up](ISSUE6_REVIEW.md) introduces a separately measured v2
policy with minimum-support gates and retains the original executed v1 notebook.
