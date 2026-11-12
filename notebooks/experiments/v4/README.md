# v4 — Canny-guided detail

**Status:** local BDD100K preparation, 50-epoch training and development evaluation completed. The matched batch16 Raw comparison is negative for v4; no AP improvement is claimed. See [results](../../../docs/v4/RESULTS.md).

**Hypothesis:** Increasing local luminance detail near stable Canny edges may help
vehicle boundaries without sharpening all background texture. It may also amplify
road markings, create halos or lose weak small-object boundaries; detector outcomes,
not attractive edge maps, decide whether the hypothesis is useful.

## Policy

[v4.yaml](v4.yaml) owns the recipe, pinned initialization checkpoint, export workers
and every native training/evaluation argument, just like v0–v3. Shared splits and
five-class taxonomy stay in [configs/data.yaml](../../../configs/data.yaml).

1. Convert original BGR to grayscale `Y`. Gaussian smoothing (sigma 1.0), followed
   by `cv2.Canny(50, 150, apertureSize=3, L2gradient=True)`, produces binary edges.
   Both run at **original resolution**; thresholds are Sobel-gradient units. Canny
   is a pixel-derived support map, never an object segmentation or a GT mask.
2. Route using original-image features: `0.04 < brightness < 0.90`, `noise <= 0.02`,
   and `0.001 < edge_density < 0.20`. Brightness/noise use the existing max-side-512
   `quality()` convention; density is the fraction of full-resolution Canny pixels
   **before dilation**. Otherwise Skip returns the original pixels exactly.
3. Dilate edges by radius 1 (3×3), then soften with a finite 3×3 Gaussian (sigma 0.8)
   to form mask `M` in [0, 1]. Compute detail `D = Y - Gaussian(Y, sigma=1.0)`.
4. Protect dark/highlight pixels with `p = clip(min(Y-8, 247-Y)/32, 0, 1)`.
   Add `round(clip(0.5*D, -12, 12) * M * p)` equally to BGR, clipping to uint8.
   Outside the finite mask neighbourhood, pixels remain unchanged. Per-channel
   changes are bounded by 12 **before JPEG encoding**; clipping near channel
   saturation can alter color. Binary edges are only displayed as diagnostics.

All coordinates, IDs, dimensions and labels stay unchanged. Both training and
evaluation use the same offline policy, followed by JPEG95 once. Skip has identical
export bytes to Raw under the same OpenCV build. The full-resolution operator is
resolution-dependent; a new processing scale is a new scientific configuration.

OpenCV references: [Canny tutorial](https://docs.opencv.org/4.10.0/da/d22/tutorial_py_canny.html)
and [Canny API](https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html).
The masking, bounded detail and route above are this experiment's hypothesis,
not an OpenCV claim about detection performance.

## Lifecycle

Each notebook independently supports **Restart & Run All** from saved prerequisites:

1. [01_experiments.ipynb](01_experiments.ipynb): fixed seed-2026 sample of at most
   64 calibration originals, read-only Original / Canny / Apply-or-Skip / difference
   panels; routing, changed-pixel fraction and CPU mean/p95. This is a pixel preview,
   not independent confirmation. It may include previously consumed calibration
   groups; do not tune thresholds or claim held-out evidence from these displays.
2. [02_train_prepare.ipynb](02_train_prepare.ipynb): `ALLOW_PREPARATION=True`
   exports `runs/v4-<YYYY-MM-DD>/data`. Review/freeze the research decision first.
3. [03_train.ipynb](03_train.ipynb): `ALLOW_TRAINING=True` fully fine-tunes from the
   same pinned COCO YOLO26n bytes as v0–v3: seed 2026, SGD, 50 epochs, batch 16,
   imgsz 640, freeze 0, geometry-only augmentation. Its own processed calibration
   split selects `best.pt`. These are the **current main** settings, not historical
   frozen-backbone S20 settings from #45/#47.

   **Local hardware adjustment (2026-11-10):** training/evaluation batch is 16 and
   training workers is 0 on the GTX 1660 Ti 6 GiB, versus batch 32/workers 8 in
   v0–v3's canonical batch32/workers8 configs. A synthetic AMP batch32 forward/backward
   used 4.94 GiB with only 5.01 GiB GPU memory free; available system RAM was
   approximately 3 GiB. Epochs, nbs=64, optimizer, initialization and policy stay
   fixed. The completed matched Raw control uses this hardware-adjusted
   recipe; it is recorded separately from the historical batch32 notebook outputs. This adjustment was made before v4 data preparation/training.
4. [04_eval.ipynb](04_eval.ipynb): `ALLOW_INFERENCE=True` evaluates only the matching
   v4 model/input on development; saves native metrics, cohorts and per-image matches.

All three expensive switches default to **False** in v4. Cross-version work remains
in [compare.ipynb](../compare.ipynb): 13 cells, including Raw/v4, v4/Raw and v4/v4,
with Raw/Raw shared across four 2×2 designs. Missing v4 is blocked without blocking
existing versions. No historical measurements or v0–v3 notebook/config bytes change.

## Research decision before execution

Proposed bounded follow-up (not executed): compare Raw, the configured Canny-guided
policy, ungated Canny-guided detail, and a matched **bounded global-detail** control
(same `D`, protection and offset bound but `M=1`). The last control is not the existing
unbounded BGR `sharpen()` operator. These four arms isolate edge support and routing;
do not add a grid search. A future multiscale/automatic-threshold Canny proposal
requires its own declared budget and config before looking at detector outcomes.

Before pilot inference, reconcile consumed source groups from #30/#45/#47. Declare
up to 512 unused, group-disjoint calibration-exploration images and a disjoint
confirmation reserve, with exact IDs, support counts, checkpoint and config hashes.
If unused groups or adequate motorcycle/bicycle support cannot be established,
stop; do not label the 64-image preview as new evidence.

Proposed gates, to freeze before measuring: primary native five-class mAP50:95
gain >= +0.2 percentage points over Raw; no class regression worse than -0.5 pp;
minimum 30 GT instances in at least 20 source groups for each protected two-wheeler
class; matched CPU policy p95 <= 25 ms/image at original BDD resolution with one
OpenCV thread. Report full support, mean/p95, routing, false positives and halos.
These are decision criteria, not measured results or a promised speed.

At most one unchanged candidate gets one independent confirmation with the same
gates. No development tuning, confirmation resampling or forced positive result.
Freeze the decision/config before any matched S20 train/eval; an explicitly labeled
negative exploratory training experiment requires a separate scientific decision.
Final-test pixels remain sealed. Conditional bootstrap CIs and a single training
seed do not prove generalization.

## Tracking

Issue bodies prepared in [docs/v4/issues](../../../docs/v4/issues): parent research
issue, bounded exploration, matched preparation/training, and development comparison.
The issue bodies retain the original proposals and have dated status notes; prospective pilot/confirmation gates are not claimed as completed.
