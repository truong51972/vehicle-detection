# Issue #47 — bounded resolution-aware training diagnostic

Commit this declaration, notebook and implementation before acquisition. This
follows PR #49's measured resolution sensitivity; its literature/evidence review
and all v0/v1/v2 and PR #48/#49 evidence remain immutable. This is a training
diagnostic, not an adaptive preprocessing policy or an S20/development result.
No enhancement, photometric change, parameter grid or new exploration is added.

## Population and ownership

Select whole leakage groups from the frozen S20 training membership, ordered by
`digest([2026, "issue47-resolution-training", group])`, stopping at the first
prefix with at least 2,048 images, at most 2,176. Never select by class, quality
or measured outcome. Retain original five-class labels and negatives. Verify
all train/evaluation ID, leakage-group, source-group and source-hash separation
and original annotation geometry before training. Require all five classes in
at least five source groups in both populations; otherwise stop without expanding.

Copy existing Raw JPEG95 S20 training exports and PR #48 Skip evaluation exports
to new owned files under `runs/issue47-resolution-training/study/data`. Verify
copied bytes, original labels, decoded geometry and the historical Skip export
receipt. No original source, new exploration, confirmation, development or final
pixels are read. Annotation metadata is allowed; no reserved confirmation plan
is read. Save exact membership, hashes and environment before training.
Historical exports/models/metrics remain untouched, including loader caches.
Existing/incomplete attempts block re-execution; no automatic retries or resume.

Evaluation uses exactly the consumed 1,027 PR #48 images / 649 source groups,
ID hash `d925d907e4ea2da0ff1301a418d7a5a194ff0f4c114f73aba2adb0b0e8c45084`.
This population already informed method choice, so intervals are exploratory,
conditional and not independent confirmation. No outcome-selected subset is added.

## Fixed training and inference budget

Two independent native Ultralytics YOLO26n runs from the same SHA256-pinned
COCO checkpoint: training `imgsz=640` and `960`, seed 2026, **five epochs**,
batch 16, workers 0, FP32/no AMP, AdamW, frozen backbone boundary 11, and otherwise
unchanged shared `TRAIN_SETTINGS` (including disabled augmentation, nbs 64,
three warmup epochs and cosine schedule). Keep both batch sizes identical; OOM
ends the study rather than silently changing it. No pretrained/S20 model reuse.
Five epochs is a short adaptation diagnostic, not a convergence comparison.

Use the **fixed final-epoch `last.pt`** for both arms. Set `val=False`; native
Ultralytics still validates at the final epoch and in final evaluation, using
only the consumed exploration population. These validation results do not select
the reported checkpoint. No development-based checkpoint selection occurs here.
Full S20 would still require the established shared Raw development selection.

At most 21,760 training image exposures (2 × 5 × 2,176). Abort an arm after
1,800 seconds at a batch boundary; record incomplete execution without retry.
Evaluate both final checkpoints at both fixed inference sizes 640/960: exactly
four explicit native evaluations, 4,108 image passes, plus native terminal
training validation (at most two passes per arm, 4,108 further image passes).
No extra size, epoch budget, seed, augmentation or rescue arm. Native FP32,
batch 32, workers 0, rect=True, confidence .001, IoU .7, max_det 300 and NMS-free
evaluation remain fixed. Stop after this matrix regardless of outcome.

## Prospectively fixed interpretation

Primary contrast: **960-trained vs 640-trained, both evaluated at 960**. Report
the full 2×2 matrix to distinguish training resolution from inference resizing.
Use native five-class mAP50:95 (primary), mAP50, per-class and historical
dark/bright/contrast/sharpness/small-two-wheeler image cohorts, original-source-
group support, confidence-.25 / IoU-.50 TP/FP/FN and deterministic gain/loss
example IDs. Cohort AP retains all GT, is not AP-small or per-image AP, and does
not identify individual missed GT boxes. No GT/metadata-based runtime routing.

Use the existing seed-2026 100-draw paired original-source-group bootstrap:
fixed all-five-class macro, no missing-class imputation/redraw. Save class
intervals and excluded draws. Rare-class intervals crossing zero remain
inconclusive; a macro pass cannot establish either rare class separately.

A useful *training-resolution* signal requires all of:

- Incremental primary gain >=0.2 AP percentage points and positive primary
  interval lower bound with all five supported classes.
- Motorcycle/bicycle mean incremental AP50:95 gain >=0.5 points, neither class
  point estimate declining, and >=5 source groups each. Report their individual
  uncertainty regardless of this engineering gate.
- No supported primary-class loss >2 points or dark/bright class loss >1 point
  (support >=5 source groups); sparse cohort/classes remain inconclusive.
- Completed five-epoch budgets with identical batch/settings apart from size;
  960 training wall time <=3×640, peak allocated VRAM <=14 GiB, and native
  inference mean at the same 960 input <=1.5× the 640-trained checkpoint.

Save training histories/settings/checkpoint hashes, wall times, peak allocated
VRAM, native preprocessing/inference/postprocessing means and synchronized
validation-batch mean/p95. Batch p95 is not image p95; sequential training/timing
and one seed limit conclusions. There is no added offline filter or new CPU
preprocessing overhead. Never equate the timings with deployment latency.

A pass prioritizes a later separately declared group-disjoint study, **not** a
policy freeze, independent confirmation or full S20 promotion. A failure stops
this fixed diagnostic without tuning or rejecting every training mechanism.
Issue #47 stays open because resolution alone does not implement adaptive
preprocessing. Default notebook execution reads only compact committed evidence.
