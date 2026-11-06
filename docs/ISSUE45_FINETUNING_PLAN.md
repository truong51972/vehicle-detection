# Issue #45 — post-pilot exploratory S20 extension

Declared after the failed pretrained pilot and before any v2 preparation/training.
The [review request](https://github.com/truong51972/vehicle-detection/pull/46#issuecomment-6680595496)
explicitly authorizes one matched fine-tuning experiment. Commit this plan,
`configs/v2.yaml`, the unchanged operator and lifecycle notebooks before execution;
record that commit and timestamps in the execution report. Keep #45 open until
the three-arm comparison is complete. No new run has executed when this plan is
committed. Execution status belongs in a separate report, not edits to this plan.

## Hypothesis and interpretation fixed before training

The frozen pretrained detector may respond differently to local-shadow images than
a detector fine-tuned on that same policy's outputs. Matching training and inference
may reduce distribution mismatch. Competing hypotheses are no useful difference,
lost contrast/clipping/noise amplification, or worse generalization to development.
One real run tests this mechanism; it does not select a policy for release.

The original pilot **failed** its improvement, regression, support, uncertainty and
mean-latency gates. Its files under `docs/issue45/` and declaration remain immutable.
The pilot declaration was hash-bound before inference but not separately committed;
this limitation remains. This extension follows a new user request and cannot
retroactively pass those gates. Training is not justified by a claimed pretrained
win, and its runtime cost remains a drawback even if accuracy improves.

## Frozen arm, sources and compute

`v2` is an **exploratory local-shadow detector/policy pair**. Move the existing
`local_shadow` function verbatim from the pilot to `preprocess.py` and re-export
it for historical callers. No parameter changes: maximum side 512 INTER_AREA,
grayscale Gaussian sigma 16, gain
`clip(sqrt(0.25 / max(illumination / 255, 0.0625)), 1, 2)`, INTER_LINEAR gain
resize, multiply original BGR, clip then cast uint8, JPEG95 exactly once.
`recipes: {local_shadow: {}}` exposes no new tuning knobs; this plan and committed
operator source bind its constants. Two deterministic pixel fixtures authenticate
the original implementation before the move. Current native wrappers also compute
their normal quality diagnostics; direct timing must include that work equally.

Reuse the frozen S20 **9,781 images / 5,915 groups**, including all S10 IDs,
and **13,975 development images**, with original five-class GT and all negatives.
Shared split SHA256:
`b7a6aba1d1f72e1b1e3eceda8e8427d41f44408067fad63ee68479137714cb96`.
Sorted S20 IDs:
`ca35e8665e4b7961379b75bf96c697210f29f0fd17c6a3a172c9de6903bea1bc`.
Verify exact IDs and every label byte against both existing arms before training.
Original sources, memberships and completed v0/v1 runs are read-only. Final-test
pixels remain sealed; source annotations may be inspected only as metadata.
No independent confirmation is used for this exploratory extension.

Prepare once at `runs/v2/data/`; independently initialize once from pinned
`weights/yolo26n.pt`, SHA256
`9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`.
Native `train_experiment` retains seed 2026, 20 epochs, 640, batch 16, FP32 AdamW,
frozen backbone 11, identical optimizer/schedule and disabled augmentations, with
**shared v0 Raw development checkpoint selection**. No additional seeds, epochs,
candidate recipes or hyperparameter changes. Resume only explicitly after a real
interruption; record it as a deviation. Never silently overwrite an existing run.

Evaluate the selected v2 checkpoint on all matching v2-processed development IDs,
using identical FP32/native NMS-free settings and the existing evaluation recovery
`workers=0`; training stays at 8 workers. Save native per-image matching arrays
for subgroup/error analysis without assigning AP to an individual image. Existing
v0/v1 metric caches remain valid and unchanged.

## Prespecified comparison and reporting

Primary descriptive comparison: policy-matched **development, single seed**
mAP50:95 and mAP50 for v0/v1/v2. Report all class AP, native precision/recall,
selected epoch, full 20-epoch histories, best/last/model/config/data hashes and
preparation/training/evaluation wall times. A worse result completes the experiment.
Any positive point difference remains exploratory because development selected
checkpoints, only one seed was trained and the pilot informed this extension.
No success threshold converts development into independent confirmation.

For subgroup/error comparisons, retain all GT in each image and use the pilot's
unchanged definitions: brightness <0.12 (dark), >=0.12 (bright), sharpness <50
(pixel-defined blur), and images containing original motorcycle/bicycle boxes
smaller than 32² pixels. Compute source-group/instance support before diagnostic
inference. Unsupported classes remain absent/inconclusive, never zero-imputed.
Do not resample or retune thresholds to obtain favorable subgroup results.

Existing v0/v1 full evaluations saved no native matching arrays. If needed, permit
**one diagnostic replay per existing arm**, over its saved processed development
images, to separate `runs/issue45-s20/diagnostics/{v0,v1}/` outputs. Reuse the v2
evaluation's matches. These replays are needed for comparable subgroup/error
statistics and must reproduce the original aggregate AP; they never retrain or
overwrite v0/v1. Maximum additional diagnostic inference: **27,950 image passes**;
v2 main evaluation: **13,975 passes**. No additional detector inference is budgeted.

Use native dataset/subgroup AP, per-class support and fixed confidence .25 / IoU .50
TP/FP counts, with deterministic illustrative GT frames, not per-image AP. Subgroup
macro averages represented classes and is separate from the primary five-class
score. No new bootstrap claim is required for this single-seed development study.

Direct preprocessing timing uses the same previously consumed 256 exploration
images for all native policies: eight CPU threads, warmup, three repeats, rotated
order; exclude source reads, JPEG, detector time and verification equally. This
is a new matched cost comparison, separate from the original failed-pilot timings.
No parameters can change based on timing or development results.

Keep compact native args/history/metrics/plots and report evidence under
`docs/issue45-s20/`; large data, model weights and matching arrays remain ignored.
Run full lightweight/study tests, Ruff and fresh-kernel smoke checks. Preserve
the original pilot evidence byte-for-byte and document which source hashes refer
to its original commit `5945f1f`, rather than rewriting them for moved/fixed code.

Notebook opt-ins: v2 `02` `ALLOW_PREPARATION=True`; `03` `ALLOW_TRAINING=True`,
`RESUME=False`; `04` `ALLOW_INFERENCE=True`. Defaults inspect saved outputs or
state that execution is pending. Downstream notebooks never execute upstream ones.
Record exact diagnostic/report commands and their execution status separately.
