# Issue #7: paired S10 implementation and completed training

[Issue #7](https://github.com/truong51972/vehicle-detection/issues/7) requires
completed paired training and development evidence before data-scale/model
selection. The guarded S10 runner is available in
[notebook 03](../notebooks/archive/phase2/03_finetuning_benchmark.ipynb). Both full S10
branches completed on 2026-11-03; see the [results report](ISSUE7_RESULTS.md) for
development metrics, checkpoint provenance, convergence and resource costs.
Issue #7 remains open for the development matrix, uncertainty and selection work.

## Frozen inputs and initial budget

- Preserve issue #5 roles and the existing pretrained development Raw baseline.
- Use issue #6 v2's shared **4,890-image S10** and byte-identical labels. Verify
  each label against frozen boxes, source hashes, geometry, export hashes and
  exact membership before training. Reject final-test IDs before decoding.
- Retain **Gamma-or-Skip**, policy ID
  `1ac3fb9bb94b132f70d3e168b5ef15a54b616444e535b9a3bb265682a6fce0ef`, as selected
  by issue #11's frozen no-v3 decision. No synthetic augmentation is added.
- Start each branch independently from the same pinned `yolo26n.pt`. Native
  semantic class-name transfer handles COCO→six-class training initialization;
  trained six-class validation uses the native validator directly.
- Derive the freeze boundary from the loaded YOLO26n YAML: **11 backbone modules**
  in Ultralytics 8.4.198. Native training freezes parameters and BatchNorm stats.
- Match 20 epochs, seed 2026, batch 16, imgsz 640, FP32, AdamW, LR 0.001→0.00001
  with cosine decay, three warmup epochs, effective nominal batch 64. Disable
  early stopping for equal budgets. Disable image augmentation, including the
  optional Albumentations defaults. Both select on the same **13,975-image Raw
  development set**, using native rectangular validation batches of 32,
  confidence 0.001, IoU 0.7, max detections 300 and `nms=False`.

These are initial S10 execution settings, not a completed expansion/selection
protocol. Changes require a new experiment root. S20/S40 are rejected by this
runner until a separate evidence-based expansion protocol is implemented.

## Checks and records

Preparation freezes data/policy/decision/checkpoint identities, absolute paths,
settings, native defaults, implementation hash, package versions and hardware.
The two fresh six-class models must have exactly the same initial state hash.
The runner checks native loader membership, frozen parameters and buffers,
finite/nonzero neck and detection gradients, and actual weight changes after a
nonzero-learning-rate step. Zero-LR warmup is allowed. An automatic OOM batch-size
reduction fails the protocol instead of silently changing a paired budget.

Native `results.csv` retains loss, AP, precision/recall and LR histories.
`freeze_audit.json` lists module/parameter names and counts. Per-epoch JSON records
optimizer steps, native validation metrics, training seconds, training pipeline
images/second, wall time including validation and peak allocated GPU memory.
Pipeline throughput includes GPU work; isolated loader throughput is still pending.
Completion requires all declared epochs and matching pair budgets. Checkpoints,
curves, audits and epoch records remain under gitignored `runs/issue7-s10-v2/`.
Completed checkpoint/result hashes are verified before cache reuse.

Resume requires the same experiment/code/environment and this run's hash-receipted,
unstripped native optimizer checkpoint. Each epoch publishes a separate atomic
resume snapshot and receipt before deleting the previous snapshot; interruption
during checkpoint writing retains the previous resumable epoch. Wrong architectures, changed settings,
changed YAMLs/exports, stale paths and symlinks are rejected. Only an epoch-boundary
checkpoint can resume; work after that boundary is repeated. Native Ultralytics
stores EMA weights and FP16 optimizer state without complete RNG/loader state, so
resume is **not bitwise uninterrupted continuation**. Epoch records identify
resumed segments. Native best/last checkpoints are stripped after completion; completed runs are reused.

## Execution evidence and next work

The GPU integration smoke uses four real frozen S10 images and two real
development images at imgsz 320, batch 2, two epochs per branch. It intentionally
interrupts after the first durable checkpoint, checks resume identity, then
resumes both branches. This verifies the native training lifecycle, six-class
head, gradients/updates and frozen backbone on an RTX 5060 Ti. It is not a fair
full S10 experiment or model-selection result; its tiny-sample AP is not reported
as study evidence. Compact provenance is in [smoke.json](issue7/smoke.json); full
logs and artifacts remain under `runs/issue7-smoke-v4/`.

Fixture tests exercise role isolation before pixel access, policy/label identity,
symlink/YAML rejection, backbone buffers, gradients/updates, zero-LR warmup,
changed-batch rejection, native paired completion/cache reuse and resume guards.
Their generated-image metrics are test fixtures, never research evidence.

Implementation verification: **89 tests passed**, Ruff lint/format, source/wheel build and
`git diff --check` passed. Both real-data GPU smoke branches passed interruption/
resume with the final implementation. The four saved issue #5 baseline code
cells and outputs remain byte-for-byte equivalent as notebook JSON values. New
training cells remain unexecuted in the saved notebook; the full run used the same
`train_branch` implementation through an external sequential launcher.
Full deep verification of both 4,890-image S10 exports and the 13,975-image Raw
development export also passed with the final source. The prepared identities
and paired YAMLs are under `runs/issue7-s10-v2/`. Both branches subsequently
completed 20 epochs and 1,746 optimizer steps without interruption. Compact
full-run evidence is in [s10_results.json](issue7/s10_results.json).

Still required before closing #7:

- Record that expansion/model-selection rules were not declared before S10
  outcomes became visible. These outcomes are exploratory for scale selection;
  do not present a retrospective rule as predeclared. Declare rules before any
  further experiments used to confirm an expansion or selection decision.
- Export/evaluate Adaptive development inputs and complete pretrained/Raw-trained/
  Adaptive-trained × Raw/Adaptive-input validation, with fixed supported-class
  and group uncertainty, weather/time/scene slices and convergence/resource review.
- Justify symmetric S20 expansion or stopping, then freeze final checkpoint(s),
  policy and data scale for #8. The final test remains unopened.
