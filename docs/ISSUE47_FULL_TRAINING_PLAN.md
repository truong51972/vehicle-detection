# Issue #47 follow-up — full S20 training-resolution experiment

**Status: finalized for guarded implementation and synthetic verification only.
Real data preparation, pixel acquisition, inference and GPU training remain gated.**
Review baseline: merged PR #50, `6432849d17a46994751ea655aac6d1bab2375f70`.
[Issue #51](https://github.com/truong51972/vehicle-detection/issues/51) carries
the design and [implementation approval/review](https://github.com/truong51972/vehicle-detection/issues/51#issuecomment-6687645032).
The user requested finalization and implementation after that review. Commit this
protocol, notebook, settings and verified checks before any real acquisition;
this implementation turn performs no real acquisition even after checks pass.

## Review and purpose

[PR #50](https://github.com/truong51972/vehicle-detection/pull/50) used 2,048
training images, five epochs (three warmup), and 1,027 already-consumed exploration
images. At fixed inference 960, training 960 added +0.913351 AP percentage points;
its conditional interval was [-0.081883, +1.778887], and supported dark-bus AP
fell 7.766145 points. Both failed gates remain failed. See the immutable
[results](ISSUE47_TRAINING_RESULTS.md), [original plan](ISSUE47_TRAINING_PLAN.md)
and [repeat declaration](ISSUE47_TRAINING_REPEAT_PLAN.md).

The requested full experiment addresses whether longer, full-S20 adaptation adds
benefit beyond inference resizing. It is a **new, single-seed Raw resolution
study**, not an adaptive preprocessing policy, a retry of PR #50, or independent
confirmation. The earlier prohibition on automatic S20 promotion remains intact:
only a separately reviewed and finalized scope can authorize this new experiment.
Issue #47 remains open; no v3 policy/config or release is implied.

Parent: [Issue #47](https://github.com/truong51972/vehicle-detection/issues/47).

Twenty epochs covers the established S20 budget, not guaranteed convergence.
Earlier S20 Raw development peaked at epoch 9 and declined by epoch 20; inspect
the full histories rather than presume more training must help. Changes from
PR #50 include population size, epoch schedule, and checkpoint-selection/evaluation
population. Do not attribute a pilot-to-full difference solely to training size.

## Frozen data and leakage boundaries

Use `configs/data.yaml` and `configs/splits.csv.gz` without resplitting. The split
SHA256 is `b7a6aba1d1f72e1b1e3eceda8e8427d41f44408067fad63ee68479137714cb96`.
ID digests below use the repository's `digest(sorted(ids))`, not a new serializer.

- Training: **all 9,781 S20 images / 5,915 leakage groups**, 73 negatives;
  IDs `ca35e8665e4b7961379b75bf96c697210f29f0fd17c6a3a172c9de6903bea1bc`.
- Selection and matrix: the unchanged **13,975 development_validation images /
  8,258 leakage groups**, 121 negatives; IDs
  `bf2957e6f237c3c2373f9913117933e6de0ae3c7d3bcf5aa700362efa0e71ff3`.
- Preserve original five-class order: car, truck, bus, motorcycle, bicycle.
  Preserve exact GT, geometry, negatives and Raw JPEG95 bytes for both arms.
- Before acquisition, verify memberships and train/development separation by ID,
  leakage group, original source group and source-image SHA256. Audit each allowed
  role against other roles using saved metadata only; stop on any overlap or
  mismatch. Do not open confirmation/final sources to perform this audit.
- Calibration-exploration is not added to training or this matrix. PR #50's
  consumed 1,027 images stay consumed; no additional evaluation on them is planned.
  All reserved confirmation pixels, its 1,025-image plan, and final-test pixels,
  headers, predictions and source hashing remain unopened.
- Development is reused for checkpoint selection and this fixed evaluation,
  never threshold/augmentation/epoch/resolution search. A new enhancement or
  adaptive policy would still need its own exploration and independent confirmation
  before a separately authorized full arm. This study does not consume that reserve.

After approval, copy verified historical v0 train/development exports once to an
owned new study directory; both arms share those same owned files. Verify copied
image/label hashes and original label geometry against existing receipts. Do not
hard-link writable files or let native label caches touch historical exports.
If prerequisites are missing, stop and declare the preparation needed; do not
silently regenerate old artifacts or substitute inputs.

## Matched training

Exactly two fresh native Ultralytics YOLO26n runs: `imgsz=640` and `imgsz=960`.
Each independently loads `weights/yolo26n.pt`, SHA256
`9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`;
neither loads a historical trained checkpoint or the other arm.

Use pinned Ultralytics **8.4.198**, the locked study environment and existing
`TRAIN_SETTINGS`. Both arms use seed 2026, deterministic=True, **20 epochs**,
**batch 16**, workers 0, device 0, FP32/amp=False, AdamW, freeze boundary 11,
lr0=.001, lrf=.01, cosine schedule, warmup_epochs=3, warmup_bias_lr=0,
weight_decay=.0005, nbs=64, cls_remap=True, rect=False, fraction=1,
cache=False, compile=False, channels_last=False, val=True and patience=0.
Retain the native warmup/accumulation behavior (nominal nbs/batch is 4), optimizer
defaults and EMA; no hand-adjusted learning rate or compute-matched epoch change.

All existing detection augmentations remain disabled: HSV, geometry, flips, BGR,
mosaic/mixup/cutmix/copy-paste, multi-scale, close_mosaic and extra augmentations.
Validate the effective saved args and initialized trainable parameter names in
both runs. Only **training imgsz** may differ scientifically between arms;
output paths/names are administrative differences. Record all other resolved
defaults, software/hardware versions and data order rather than relying on memory.

Run sequentially, 640 then 960, on the same otherwise idle GPU. Reject batch
auto-reduction, OOM/NaN recovery, AMP fallback, optimizer auto-selection, early
stopping, automatic retries or resume. An interrupted/recovered run is incomplete,
preserved and disclosed; a repeat needs a prospective addendum. Identical seeds
do not promise bitwise equality across image shapes or software/hardware changes.

## Checkpoint selection and fixed 2×2 evaluation

**Final selection: shared Raw development at a fixed 640 validation resolution
for both arms**, every epoch, batch 32, workers 0, FP32, rectangular/NMS-free,
with the same AP settings below. This preserves the established Raw-development
selection principle while keeping selection resolution independent of training
resolution. Use native fitness/best.pt selection; capture native best-checkpoint
epoch/train_metrics on each save before stripping, then authenticate against the
native history and final checkpoint metrics. This preserves native tie handling
without guessing from identical AP rows or stripped `epoch=-1`. Native CSV uses
six significant digits; metric comparisons allow that serialization precision.
Retain last.pt and all 20 epochs for convergence reporting.
Fixed-640 selection may favor checkpoints suited to 640; conclusions are
conditional on that rule, including D-B at 960.

Important implementation constraint: ordinary `YOLO.train(imgsz=960)` also builds
its validation loader at 960. Calling the existing `train_experiment` unchanged
therefore does not implement this protocol. The notebook uses the testable native
callback in `full_resolution_training.install_selection` to configure the
validation args **and loader** at fixed 640 before the
first validation, leaving training tensors/settings untouched. Native rectangular
validation uses extra `pad=.5`, producing a 672-pixel maximum tensor edge at nominal
640. Selection instead uses the native `YOLODataset` with **pad=0.0** in both arms:
the maximum actual tensor edge is 640, with stride-aligned shorter edges. This is
a prospective selection-padding change, not exact historical selection replication.
The inference matrix retains native rectangular pad=.5 at each nominal resolution;
record the actual shapes rather than label padded tensors as literal 640/960.
Reuse native
loader/validator/fitness/checkpointing; do not introduce a custom trainer or an
epoch-by-epoch training loop. Native terminal best-checkpoint validation must
also remain at 640. Record this override separately from train args and assert
actual train/validation tensor shapes and IDs. A focused tiny native smoke check
must establish this behavior before real acquisition. If it cannot be achieved
cleanly with the pinned API, return to design review before training; do not
silently select each arm at its own resolution or substitute final-epoch selection.

Freeze one selected checkpoint per arm, then perform exactly these four explicit
full-development evaluations, in this predeclared order:

1. Train 640, infer 640: **A**.
2. Train 960, infer 960: **D**.
3. Train 640, infer 960: **B**.
4. Train 960, infer 640: **C**.

Each checkpoint is identical in its two cells. No cell-specific epoch selection,
winner selection, confidence tuning, TTA, slicing or additional inference size.
Explicitly pass every evaluation setting rather than inherit checkpoint imgsz:
batch 32, workers 0, FP32/half=False/quantize=32, rect=True, conf=.001,
iou=.7, max_det=300, nms=False, agnostic_nms=False, augment=False, seed 2026,
deterministic=True. Use existing `ExperimentValidator`/saved native matching stats;
check NMS-free mode through `on_val_end`, retaining PR #50's API correction.
Verify exact loader IDs, GT counts and native AP agreement with saved-match AP.

Primary contrast: **D - B**, incremental training benefit at fixed inference 960.
Also report **C - A** at fixed inference 640, inference-only changes **B - A**
and **D - C**, and interaction **(D - B) - (C - A)**. D - A changes both factors
and cannot isolate training-resolution benefit. The primary estimates the benefit
under this fixed checkpoint-selection procedure, not at a common final epoch.

## Metrics, uncertainty and prospective interpretation

For every cell report five-class native mAP50:95, mAP50, each class's AP50:95/AP50,
precision/recall and GT/image/original-source-group support. Emphasize motorcycle
and bicycle separately. Report class and total **TP/FP/FN at conf >=.25, matching
IoU .50**, using existing saved native matches and GT totals (FN=GT-TP); do not
raise the AP validation conf to .25. Native F1-selected P/R describes a different
operating point. Retain negative-image FP counts and deterministic gain/loss IDs;
AP gains alone do not establish improved two-wheeler recall at .25.

Repeat the already fixed dark (<.12 brightness), bright and original-GT small
two-wheeler (<32² area) image cohorts on development using existing feature units.
Compute features once from allowed Raw development exports after approval,
independent of predictions; retain all GT in each cohort. Explicitly revisit
dark-bus behavior. Cohort AP is neither AP-small nor per-image AP; saved class
counts do not identify individual missed GT boxes. No new threshold grid or route.

For D-B and each fixed dark/bright comparison, use **1,000** paired seed-2026
**original-source-group** bootstrap draws, frozen before outcomes
(supply source_group as the grouping key). Fixed all-five-class macro requires
>=5 original source groups per class; record valid/missing-class draws without
imputation or redraw. Save individual class intervals. Unsupported cohorts/classes
are inconclusive, absent AP stays absent, and five groups is only an engineering
minimum. These intervals condition on two selected models and reused development;
they omit training-seed/selection uncertainty and multiple-comparison correction.

Final prospective **research follow-up** criteria, reusing PR #50's numerical
engineering tolerances on this newly declared population (never rescoring PR #50):

- D-B >=+.2 AP points, supported all-five-class macro and positive interval lower bound.
- Motorcycle/bicycle mean AP gain >=+.5 points with neither point estimate declining;
  report each interval and fixed-confidence counts even if this criterion passes.
- No supported whole-development class loss >2 points or dark/bright class loss
  >1 point (support >=5 source groups). Sparse findings remain inconclusive.
- Both complete 20 epochs with matched settings/batch; 960 training wall <=3×640,
  peak allocated VRAM <=14 GiB, and D inference mean <=1.5×B at the same input size.

Record every failed/unsupported condition; do not collapse contradictory AP,
TP/FN or subgroup evidence into a success label. A sparse dark/bright class,
motorcycle/bicycle interval spanning zero, or a positive AP change accompanied by
worse fixed-confidence TP/FN for those classes or dark-bus makes the follow-up
decision inconclusive unless another criterion establishes a negative outcome.
Report GT counts, distinct source groups and class intervals for dark-bus and
both two-wheeler classes even when the macro criterion passes. Passing only prioritizes a
separate reviewed follow-up. Failure stops this fixed study without rescue arms;
neither outcome releases a policy, proves convergence/generalization, consumes
confirmation/final data or closes the broader adaptive-preprocessing issue.

## Compute, memory and cost review

- Training: **391,240 image exposures**, 195,620 per arm; **12,240 minibatches per
  arm** at batch 16, including the last partial batch (no drop_last). This is
  **19.1035×** the completed PR #50 repeat's 20,480 exposures. Twenty epochs also
  changes the cosine schedule relative to the five-epoch diagnostic.
- At equal batches, 960 has **2.25× square input pixels**; this is not an exact
  time/VRAM multiplier. PR #50 measured 121.692/210.278 s and 2.347/5.240 GiB
  allocated at 640/960 on the recorded RTX 5060 Ti 16 GiB. Simple exposure scaling
  gives **38.75/66.95 min**, not measured full-run costs. Earlier full-S20 640
  training with per-epoch development validation took about 34–35 min at workers 8.
  Workers 0, more batches/labels, full-development validation, warmup and clock/I/O
  effects limit extrapolation. Budget approximately **40 min / 65–80 min** for
  the two arms and a **~3 GPU-hour planning ceiling** including preparation/evaluation
  margin. This is not a measurement or a guarantee; the per-arm abort ceilings
  below are separate limits and can exceed that planning estimate in aggregate.
- Validation is not free: **559,000 image passes** for 2×20 development selection
  passes, **27,950** for one native terminal best validation per arm, and **55,900**
  for the four matrix cells: **642,850 total planned validation image passes**.
  Each batch-32 pass has 437 batches; no hidden diagnostic/model-check evaluations.
  Bootstrap/cohort reports reuse saved matches and add CPU cost, not inference.
  The fixed 1,000 draws for each of all/dark/bright total 3,000 CPU resamples;
  their full-development wall time and RAM are unmeasured and excluded from the
  GPU-hour estimate. Do not reduce draws after viewing intervals.
- Hard stop: **90 min at 640 / 150 min at 960** at batch boundaries;
  do not use native time-based training to silently shorten the 20-epoch budget.
  These are abort ceilings, not predicted runtimes. Preserve incomplete evidence
  and require review before spending more. Observe peak allocated and reserved
  VRAM separately; >14 GiB allocated stops the arm without changing batch/precision.
- One owned train+development image copy is approximately **2.75 GiB**, based
  on current historical export file sizes; labels/caches, checkpoints and four
  saved-match files add space. Require **20 GiB free** before copying; record actual
  disk use, host RAM and CUDA/GPU state. Full-development saved stats/bootstrap may
  be memory-intensive; report/stop resource failure rather than subsample silently.

Measure per-arm full wall time (including validation), native cumulative CSV time,
epoch durations, selected/final epoch, train/val losses, AP trajectory, LR and
allocated/reserved VRAM. Plot convergence against both epochs and cumulative
seconds; differing input sizes can change loss scales. Report peaks and final
trends; do not extend training based on these curves.

For each matrix cell save phase wall time, native preprocess/inference/postprocess
ms/image, measured image throughput and synchronized validation-batch mean/p95
with batch counts. Use the same machine/process settings, native warmup, and
actual rectangular shapes; separate initialization/warmup from steady inference.
Compare inference-size cost within each checkpoint and checkpoint cost at fixed
size. Batch p95 is not per-image p95; one sequential pass has order/cache/thermal
limitations and is not deployment latency. No extra latency runs, CPU filter
benchmark, TensorRT/export or batch-size search is included.

## Minimal notebook workflow and evidence

Implement one independent research notebook:
`notebooks/research/issue47/03_full_resolution_training.ipynb`, with ordered
review/preflight → owned preparation → two native train calls → four native val
calls → saved-evidence report. Default Restart & Run All reads only verified
compact evidence (or clearly reports not acquired), with no model/data construction.
Preparation/training/inference require separate explicit opt-ins and the committed
approved plan. Each section reads saved prerequisites, so reports can be rerun
without training. No new registry, service, dependency, trainer framework or
generic experiment abstraction is needed.
Keep the committed notebook unchanged: after running Configuration, set the
intended `ALLOW_* = True` in an attached Jupyter console, then execute that stage's
cell. Re-running Configuration resets all flags. Saved source edits are rejected
by the frozen-input guard; no flags need to be committed as True.

Use new owned `runs/issue47-s20-resolution-20e/` and
`docs/issue47-s20-resolution-20e/` roots. Existing/incomplete acquisition blocks
reruns; never delete an attempt to make a guard pass. Save run/commit/environment,
membership and split/GT/export/config/code/initialization hashes before training.
Save wall/VRAM receipts immediately after each arm and timing/checkpoint receipts
after each cell, before downstream reporting. Retain both best/last checkpoint
hashes, resolved train/selection/evaluation settings, histories, all four metrics,
matching arrays, uncertainty draws, failures and executed notebooks/logs. Commit
compact JSON/CSV/plots/settings; weights, pixels, caches and large arrays stay ignored.

Preserve all existing plans/results, PR #50's failed attempt and repeat, notebooks,
models and native caches. At acquisition preflight record hashes of historical
tracked evidence/configs/notebooks and size/mtime inventories of historical local
runs/weights; compare afterward without opening protected pixels. State that
size/mtime preservation is not a cryptographic rehash of all old images.

The new notebook compares resolutions within this study only. Cross-version
v0/v1/v2/v3 S20 presentation remains exclusively in `experiments/compare.ipynb`;
do not insert a fabricated v3 row or change that notebook for this study. Add only
navigation links to the new research notebook/plan when implemented.

## Review and acceptance gates

- [x] Review/finalize this new exploratory scope, development reuse, fixed-640
  checkpoint selection, primary contrast, criteria, timing limits and cost budget.
  Approval: the linked design review authorizes implementation only. The latest
  user instruction authorizes this final protocol and commit; no real acquisition
  is performed in this turn. The committed revision and validation receipt bind it.
- [x] After approval, implement only the new notebook and any small necessary
  reuse; preserve old runner/constants/results and compare behavior.
- [x] Before real acquisition, focused native synthetic checks prove fixed
  validation size/loader and frozen checkpoint use, matching settings/IDs,
  overwrite/recovery rejection and hash-verified read-only defaults. Run focused
  tests, Ruff and fresh-kernel notebook smoke; commit the approved implementation.
  [Implementation receipt](issue51/implementation.json) binds the protocol,
  notebook and module. [Validation receipt](issue51/validation.json) records the
  commands, preservation check and limits; this is software evidence only.
- [ ] Only then explicitly opt in to real preparation, 2×20-epoch training and
  the four-cell matrix. Disclose interruption/failed cells; no invented completion.
- [ ] Publish all requested metrics, convergence/cost receipts, integrity checks
  and a candid positive/negative/inconclusive decision; keep #47 open.

Review sources: [S20 membership/settings/timing](ISSUE43_EXECUTION.md),
[shared training settings](../src/vehicle_detection/training_settings.py),
[native training integration](../src/vehicle_detection/finetuning.py),
[saved-match reporting](../src/vehicle_detection/evaluation_stats.py),
[Ultralytics train](https://docs.ultralytics.com/modes/train/), and
[Ultralytics validation](https://docs.ultralytics.com/modes/val/).
Current online docs support the native approach; the installed pinned 8.4.198
source and focused checks govern API behavior. No package upgrade is proposed.
