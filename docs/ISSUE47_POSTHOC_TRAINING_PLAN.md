# Issue #47 / PR #53 — explicitly requested post-hoc S20/960 training

The user explicitly requested one exploratory Adaptive V3 fine-tuning run after
the completed negative acceptance review. This authorizes this separate study
despite the earlier stop-before-training decision. It does **not** change that
decision, its thresholds, candidate code, data or evidence, or establish independent
confirmation. Everything here stays inside PR #53. No other PR, S100 or v4.

## Fixed question and interpretation

Does training on the **unchanged `adaptive_local` input policy** improve the
detector's response to that policy, compared with the Raw-trained PR #52 detector?
The original failed primary/uncertainty/two-wheeler/dark-bus gates stay failed.
No policy selection, threshold fitting, seed search, confirmation or release occurs.
This single-seed experiment reuses development for checkpoint selection/evaluation;
all contrasts are **post-hoc descriptive exploration**, not independent evidence.

Commit this plan, implementation, notebook and focused safeguards before any
preparation/training. Use new owned `runs/issue47-posthoc-adaptive-960` and
`docs/issue47-posthoc-adaptive-960` roots. Existing attempts block retries/resume;
preserve incomplete attempts and failures. Saved-only reporting can continue
without repeating training/inference, but never hide an acquisition interruption.

## Data and unchanged preprocessing

Reuse the canonical split table and PR #52's hash-verified saved rows, Raw JPEG95
export receipt and Raw-development feature cache. Require exact canonical ID,
source/content group, source SHA256, geometry and original five-class GT identities.
There are **9,781 S20 training images / 5,915 leakage groups / 73 negatives** and
**13,975 development images / 8,258 leakage groups / 121 negatives**. Sorted-ID
digests remain `ca35e8665e4b7961379b75bf96c697210f29f0fd17c6a3a172c9de6903bea1bc`
and `bf2957e6f237c3c2373f9913117933e6de0ae3c7d3bcf5aa700362efa0e71ff3`.

Apply the frozen `vehicle_detection.v3_adaptive.apply(image, "adaptive_local")`
to original decoded S20/development sources, followed by one JPEG95 encode.
Do not apply it to already re-encoded Raw exports. Verify original source hashes
and geometry, verify a fresh Raw JPEG95 encoding equals PR #52's saved bytes,
and copy exact original label bytes. Skip must yield Raw-identical bytes.
Keep the ±12/clipping limitations and original feature thresholds unchanged.

Own all new Adaptive train/development and Raw development files; no hard links
or native cache writes to historical exports. Preparation uses four CPU threads
with one OpenCV thread each; this is export throughput, not a new cost benchmark.
Guard source access to exactly the approved training/development paths. Calibration,
reserved confirmation and final-test pixels/headers/hashes/predictions stay sealed.
Read annotations only through already verified saved metadata; do not resplit.

## Exactly one Adaptive training run

Independently initialize the same pinned COCO YOLO26n as PR #52:
`weights/yolo26n.pt`, SHA256
`9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`.
Do **not** continue from the Raw-trained checkpoint. Use pinned Ultralytics
8.4.198 and PR #52's recorded NumPy/OpenCV/Torch versions on this host.

Use unchanged `TRAIN_SETTINGS` with imgsz960, 20 epochs, batch16, workers0,
device0, seed2026, deterministic=True, FP32/amp=False, AdamW lr0=.001/lrf=.01,
cosine, warmup3, weight_decay=.0005, nbs64, freeze11, cls_remap=True,
rect=False, patience0 and all existing detection augmentations disabled.
Compare every resolved training argument against PR #52's saved train960 args;
only data/project/name/save_dir paths may differ. Verify identical initialized
trainable parameter names. Keep native optimizer, EMA, scheduling and data order.

Reuse the unchanged `full_resolution_training.install_selection` callback:
every epoch and terminal best validation uses the same **owned Raw development
images at actual640 / pad0 / batch32 / FP32**, with native fitness/best selection.
Adaptive training tensors remain square960. Capture native selected epoch before
checkpoint stripping, all 20 CSV rows, best/last hashes, every observed validation
pass and actual tensor shape. Preserve PR #52's Raw best checkpoint (epoch10,
SHA256 `b13ec9339f7a58d55dd53c46f7e0e24330a889a761f3ee8cc4a3e9764573b2cb`).

Retain the same batch/recovery/finite-loss/14GiB safeguards and 150-minute
training abort ceiling supplied by that callback. No automatic retry, resume,
AMP fallback, OOM batch reduction, shorter schedule or final-epoch substitution.
PR #52 measured 91.176 minutes / 5.239GiB allocated for Raw960; that is a planning
reference, not a timing guarantee. Require >=20GiB free disk before preparation.

## Fixed full-development 2×2 evaluation

All cells use the same 13,975 original IDs/GT and **nominal960 / native pad.5**,
FP32, batch32, workers0, conf.001, IoU.7, max_det300, quantize32, NMS-free,
seed2026, deterministic=True, no augmentation/TTA. Freeze one selected checkpoint
per training arm before evaluating; no cell-specific model or epoch choice.

1. **Raw/Raw (A)**: PR #52 Raw-trained checkpoint on owned Raw development.
2. **Raw/Adaptive (B)**: the same Raw checkpoint on Adaptive development.
3. **Adaptive/Raw (C)**: newly selected Adaptive checkpoint on Raw development.
4. **Adaptive/Adaptive (D)**: the same Adaptive checkpoint on Adaptive development.

Rerun A as a control, not Raw training. Require its native AP50:95/AP50 and all
class/cohort metrics to reproduce PR #52's saved 960/960 results within absolute
1e-6; a mismatch invalidates the comparison and stops further cells. All cells
verify loader IDs, original GT, actual shapes, checkpoint hashes, NMS-free mode
and native/saved-match AP agreement. Reuse the existing native validator/matcher.

Primary descriptive training contrast **D-B** holds Adaptive evaluation inputs
fixed. Report **C-A** at Raw evaluation inputs; preprocessing effects **B-A** and
**D-C**; joint pipeline contrast **D-A**; and interaction **(D-B)-(C-A)**.
D-A changes both training and evaluation preprocessing and cannot isolate either.

Report mAP50:95/mAP50 and each class's AP50:95/AP50, emphasizing motorcycle and
bicycle separately; fixed-conf .25 / matching-IoU .50 TP/FP/FN and original
class/group support. Report all/dark/bright/small-two-wheeler image cohorts using
the preserved PR #52 Raw-JPEG feature cache/GT for comparability, not new outcome
thresholds. Policy routing itself always uses original decoded-source features.
Retain every GT in cohort images; this is neither AP-small nor per-image AP.
No new bootstrap grid or significance/confirmation claim is added: these are
one-seed, reused-development point estimates under the fixed selector.

## Budget, cost and reporting

- New training: **195,620 image exposures**, 12,240 minibatches across 20 epochs.
- Raw-development selection: **279,500 image passes** plus **13,975 terminal**.
- Explicit four-cell matrix: **55,900 image passes**. Total new native validation
  passes: **349,375 images**. CPU-only reporting adds no detector inference.
- Record preprocessing/export wall time and route distribution separately from
  training wall (including selection/terminal validation), per-epoch/cumulative
  native time, loss/AP curves, selected epoch and peak allocated/reserved VRAM.
  Compare with the verified PR #52 Raw960 cost; order/host/cache effects remain.
- Retain native per-cell stage timings, phase wall, synchronized batch mean/p95
  and actual shapes. Batch p95 is not per-image p95 or a deployment SLA.

One independent research notebook defaults to compact read-only evidence and
explicit preparation/training/inference opt-ins. Preserve frozen 01–04 and their
historical negative records. Add the post-hoc matrix to `compare.ipynb` only from
the new study's verified committed results, clearly separated from the failed
calibration pilot and historical S20/640. No released `configs/v3.yaml` is created.

Persist cost immediately after training and metrics/timing after each cell,
before subsequent reporting. Commit compact JSON/CSV/plots/settings/route support
and data/code/checkpoint hashes. Keep original pixels, exports, models, native
matching arrays, detailed logs and executed notebooks ignored. Preserve all prior
tracked evidence and historical local image/model size/mtime inventories. Run
focused tests, full tests, Ruff and fresh-kernel read-only checks; update **PR #53**
with measured findings and limitations. Earlier frozen results remain unchanged.
