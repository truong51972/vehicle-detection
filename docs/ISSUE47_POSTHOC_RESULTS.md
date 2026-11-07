# Issue #47 / PR #53 — post-hoc Adaptive V3 S20/960 fine-tuning

**Completed single-seed post-hoc exploration; not independent confirmation.**
The separately requested Adaptive-trained model does not improve aggregate native
AP over either Raw-trained reference on the corresponding evaluation inputs, and
loses motorcycle and bicycle AP. No policy is promoted or released. The original
negative pilot, frozen policy, thresholds, gates and 01–04 lifecycle evidence stay
unchanged. This is a bounded result for this operator, recipe and seed.

## Fixed design and acquisition

The [post-hoc plan](ISSUE47_POSTHOC_TRAINING_PLAN.md), acquisition module, notebook
and safeguards were committed at `a94c0c896a9ad7aa099599f5e0cad98d3ca58ee1`
before data preparation. The experiment ran once in a fresh notebook kernel; no
retry, resume, recovery, AMP fallback, batch reduction or shorter schedule occurred.
The acquisition code, plan, tests and source notebook have not changed since.

- Independently initialize YOLO26n from the same pinned COCO weights as Raw PR #52;
  do not continue from the trained Raw checkpoint.
- Train **all 9,781 S20 images for 20 epochs**, square 960, batch 16, seed 2026,
  deterministic FP32, AdamW, frozen backbone boundary 11 and the unchanged PR #52
  recipe. Every resolved argument matches Raw except data/output paths, and
  initialized trainable parameter names match.
- Use the same full **13,975-image Raw development selector**, actual 640/pad 0/batch 32
  at every epoch and terminal validation. All 21 passes observed **384×640** FP32
  tensors. Native best selection chose **epoch 12**; Raw selected epoch 10.
- Evaluate exactly four cells on identical development IDs and original five-class
  GT, nominal 960/native pad 0.5/batch 32/FP32. Every cell observed **576×992** tensors,
  437 batches, confidence 0.001, IoU 0.7, max_det 300 and NMS-free mode. Native AP agrees with
  saved native matching statistics. Checkpoints are fixed across input domains.

Development is reused for checkpoint selection and this comparison. These are
one-seed descriptive point estimates, without new bootstrap intervals, significance
claims, confirmation access, policy selection or final-test evaluation.

## Native development matrix

AP values below are percentages; differences are **percentage points (pp)**.
A = Raw-trained/Raw inputs, B = Raw-trained/Adaptive inputs,
C = Adaptive-trained/Raw inputs, D = Adaptive-trained/Adaptive inputs.

| Training | Evaluation inputs | Cell | mAP50:95 (%) | mAP50 (%) |
| --- | --- | --- | --- | --- |
| raw | raw | A | 24.568446 | 41.600479 |
| raw | adaptive | B | 24.537852 | 41.522696 |
| adaptive | raw | C | 24.491937 | 41.387553 |
| adaptive | adaptive | D | 24.525520 | 41.430396 |

A reproduces PR #52’s saved 960/960 class/cohort AP within absolute 1e-6 and
all fixed-threshold TP/FP/FN counts exactly; aggregate AP is also identical.

| Fixed contrast | Δ mAP50:95 (pp) |
| --- | --- |
| D−B: training effect at Adaptive inputs | -0.012332 |
| C−A: training effect at Raw inputs | -0.076509 |
| B−A: preprocessing with Raw-trained model | -0.030593 |
| D−C: preprocessing with Adaptive-trained model | +0.033584 |
| D−A: joint pipeline change | -0.042925 |
| (D−B)−(C−A): interaction | +0.064177 |

The primary descriptive training contrast **D−B is −0.012332pp**. The joint
change **D−A is −0.042925pp** and changes both training and evaluation inputs.
The positive interaction is a difference between two negative training effects;
it does not establish an aggregate accuracy benefit.

### Per-class AP50:95

| Class | A | B | C | D | D−B (pp) | D−A (pp) |
| --- | --- | --- | --- | --- | --- | --- |
| car | 41.037974 | 41.024447 | 41.011143 | 41.080488 | +0.056042 | +0.042514 |
| truck | 28.803535 | 28.718528 | 28.588975 | 28.550538 | -0.167991 | -0.252997 |
| bus | 27.735028 | 27.762289 | 29.925944 | 30.038436 | +2.276148 | +2.303409 |
| motorcycle | 11.090448 | 11.077007 | 9.632939 | 9.650877 | -1.426130 | -1.439571 |
| bicycle | 14.175244 | 14.106991 | 13.300681 | 13.307262 | -0.799729 | -0.867981 |

### Per-class AP50

| Class | A | B | C | D | D−B (pp) | D−A (pp) |
| --- | --- | --- | --- | --- | --- | --- |
| car | 69.830974 | 69.860238 | 70.090200 | 70.198084 | +0.337846 | +0.367110 |
| truck | 43.268986 | 43.119842 | 42.621654 | 42.578921 | -0.540921 | -0.690065 |
| bus | 38.379809 | 38.396732 | 41.039183 | 41.123265 | +2.726533 | +2.743456 |
| motorcycle | 23.429944 | 23.208225 | 21.160355 | 21.181432 | -2.026794 | -2.248513 |
| bicycle | 33.092680 | 33.028441 | 32.026371 | 32.070279 | -0.958162 | -1.022401 |

At fixed Adaptive inputs, motorcycle AP50:95 falls **1.426130pp** and bicycle
falls **0.799729pp** with Adaptive training (two-class mean **−1.112929pp**).
Bus gains **2.276148pp**, which largely masks these losses in five-class mean AP.
The joint pipeline loses **1.439571pp** for motorcycle and **0.867981pp** for bicycle.

### Original support and native fixed-threshold errors

Counts use confidence ≥0.25 and native one-to-one matching at IoU 0.50. Entries are
**TP / FP / FN**, with unchanged original GT. They are separate from native
AP’s confidence sweep and are not the precision/recall operating point used by AP.

| Class | GT | Source groups | A: TP / FP / FN | B: TP / FP / FN | C: TP / FP / FN | D: TP / FP / FN |
| --- | --- | --- | --- | --- | --- | --- |
| car | 142541 | 8199 | 80224 / 18546 / 62317 | 80640 / 19187 / 61901 | 84304 / 23384 / 58237 | 84782 / 23883 / 57759 |
| truck | 5965 | 3043 | 2215 / 1509 / 3750 | 2244 / 1574 / 3721 | 2539 / 2332 / 3426 | 2571 / 2422 / 3394 |
| bus | 2276 | 1549 | 737 / 467 / 1539 | 736 / 462 / 1540 | 830 / 535 / 1446 | 824 / 529 / 1452 |
| motorcycle | 562 | 408 | 112 / 76 / 450 | 109 / 84 / 453 | 101 / 87 / 461 | 101 / 89 / 461 |
| bicycle | 1430 | 794 | 332 / 145 / 1098 | 325 / 145 / 1105 | 371 / 233 / 1059 | 371 / 235 / 1059 |

### Preserved image cohorts

Cohorts use PR #52’s frozen Raw-JPEG feature cache and original GT. Dark means
brightness <0.12; bright is its complement. Small-two-wheeler images contain at
least one original motorcycle/bicycle box smaller than 32² pixels. Every original
GT in a cohort image is retained; this is neither AP-small nor per-image AP.

| Cohort | Images | A: mAP50:95 (%) | B: mAP50:95 (%) | C: mAP50:95 (%) | D: mAP50:95 (%) |
| --- | --- | --- | --- | --- | --- |
| dark | 3308 | 18.236810 | 18.018087 | 17.205083 | 17.553391 |
| bright | 10667 | 25.447283 | 25.431571 | 25.476058 | 25.480032 |
| small_two_wheeler | 412 | 22.626081 | 22.665806 | 22.884890 | 23.021572 |

| Cohort/class | A: AP50:95 (%) | B: AP50:95 (%) | C: AP50:95 (%) | D: AP50:95 (%) |
| --- | --- | --- | --- | --- |
| dark/motorcycle | 4.997001 | 4.598440 | 2.423478 | 2.783383 |
| dark/bicycle | 10.516474 | 9.884599 | 8.852377 | 8.923354 |
| small_two_wheeler/motorcycle | 6.644077 | 6.772766 | 5.611743 | 5.649851 |
| small_two_wheeler/bicycle | 6.085219 | 6.086861 | 6.118359 | 6.109967 |

Dark aggregate AP falls **0.464697pp** for D−B and **0.683420pp** for D−A.
The small-two-wheeler image-cohort mean rises, while its motorcycle AP falls;
the all-class cohort mean must not be read as two-wheeler improvement. Complete
class/cohort AP50/AP50:95, support and TP/FP/FN remain in the per-cell JSON files.

## Training and preprocessing cost

Both arms ran all 20 epochs. Wall time includes native epoch selection and
terminal validation; these are same-host observations, not a randomized speed
benchmark. Host/cache/order effects limit timing comparisons.

| Arm | Wall (s) | Wall (min) | Peak allocated (GiB) | Peak reserved (GiB) | Selected epoch |
| --- | --- | --- | --- | --- | --- |
| raw | 5470.556 | 91.176 | 5.238813 | 5.544922 | 10 |
| adaptive | 5205.985 | 86.766 | 5.239342 | 5.597656 | 12 |

Adaptive training used **264.571 s less wall time (−4.836%)** in this observation,
with essentially equal allocated VRAM. This does not offset the measured
two-wheeler AP losses.

The native 20-row history, losses and learning rates are saved as
[training_results.csv](issue47-posthoc-adaptive-960/training_results.csv).
The [read-only comparison notebook](../notebooks/experiments/compare.ipynb) displays
the two training curves using the same actual-640 selector, the 960 matrix, both
per-class AP measures, selected epochs and costs. Selection metrics at 640 must
not be mixed with the separately evaluated 960 matrix.

Preparation/export took **179.493 s**, including source/Raw byte verification,
decoding, the frozen transform, JPEG95 encoding and receipt work. It is separate
from training and is not a per-image online policy benchmark. The unchanged
historical pilot measured Adaptive CPU preprocessing at 5.811 ms mean / 11.913 ms
p95 on its own matched population; see the
[frozen pilot report](ISSUE47_V3_960_RESULTS.md). Native validation timing below
reads already exported inputs and excludes that offline policy work.

| Cell | Phase wall (s) | Native inference (ms/image) | Batch mean (ms) | Batch p95 (ms) |
| --- | --- | --- | --- | --- |
| A | 115.850 | 2.381177 | 147.371 | 165.107 |
| B | 125.379 | 2.376386 | 147.988 | 167.405 |
| C | 115.885 | 2.372818 | 147.566 | 164.883 |
| D | 115.738 | 2.379162 | 146.113 | 161.854 |

Batch p95 is synchronized batch latency, not per-image p95 or a deployment SLA.
Full stage timings and checkpoint identities are in the saved measurements.

## Data identity and preservation

Adaptive export starts from each original decoded source and applies unchanged
`v3_adaptive.apply(image, "adaptive_local")`, followed by one JPEG95 encode. A
fresh Raw JPEG95 encode must reproduce PR #52’s saved bytes. Labels are copied
byte-for-byte; geometry, original five-class GT and negatives are preserved.
Skipped images are Raw-identical. All new model-facing files/caches are owned
under the separate study root; historical exports and models are untouched.

| Population | Images | Negatives | Apply | Skip |
| --- | --- | --- | --- | --- |
| train | 9781 | 73 | 3916 | 5865 |
| val | 13975 | 121 | 5653 | 8322 |

- train sorted-ID SHA256: `ca35e8665e4b7961379b75bf96c697210f29f0fd17c6a3a172c9de6903bea1bc`
- val sorted-ID SHA256: `bf2957e6f237c3c2373f9913117933e6de0ae3c7d3bcf5aa700362efa0e71ff3`

Pinned COCO initialization SHA256:
`9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`.
Raw best SHA256:
`b13ec9339f7a58d55dd53c46f7e0e24330a889a761f3ee8cc4a3e9764573b2cb`.
Adaptive best SHA256:
`a1bfb001ebc70189ff01ed6d415a1b4683c8439edc7f4779fa9a07f0a4858266`.
Adaptive last SHA256:
`ff7b4b8fbe0ec6e7afd73ef6a291178b771f5bba1b4e7a196cace68f82ed1a78`.

The pinned environment is NumPy 2.4.6, OpenCV-Python 4.14.0.94, Torch 2.15.0 and
Ultralytics 8.4.198 on the RTX 5060 Ti. The acquired plan binds source/notebook/test
hashes. Saved preservation checks verify **491 prior tracked hashes** and
**223,780 historical local size/mtime records** unchanged. Local historical
stat inventories are not a claim that every old pixel file was rehashed.

New acquisition totals: **195,620 training image exposures**, **293,475 selector/
terminal image passes**, and **55,900 explicit matrix image passes**. Confirmation
and final-test pixels, hashes and predictions were not acquired. No S100, v4,
new seed, policy retuning or additional experiment beyond this declared study
was started.

## Interpretation and reproducibility

This run supplies the previously unmeasured transformed-domain training arm. It
does not improve aggregate AP in either matched-input training contrast and
reduces the priority motorcycle/bicycle AP. Retain the existing negative policy
promotion conclusion. Do not retrospectively apply or relax the failed pilot
gates, treat this reused development set as independent confirmation, or publish
a released `configs/v3.yaml`. Small point differences and one-seed costs do not
establish general behavior across seeds, detectors, datasets or adaptive methods.

The frozen [01–04 v3 lifecycle](../notebooks/experiments/v3/README.md) records the
earlier conditional stop correctly. This later, explicitly authorized extension
is independently runnable through
[05_posthoc_adaptive_training.ipynb](../notebooks/research/issue47/05_posthoc_adaptive_training.ipynb),
with all acquisition switches defaulting False. Saved-only reporting needs no
local images, checkpoints, inference or upstream notebook execution.

[Acquired result](issue47-posthoc-adaptive-960/result.json) ·
[Evidence file hashes](issue47-posthoc-adaptive-960/evidence.json) ·
[Native checkpoint/selection receipts](issue47-posthoc-adaptive-960/checkpoint.json) ·
[Resolved training settings](issue47-posthoc-adaptive-960/training_args.yaml) ·
[Post-execution validation](issue47-posthoc-validation.json).
Original pixels, owned exports, models, native matching arrays, full logs and
executed notebooks remain under ignored local runs; compact evidence is committed
inside PR #53. Earlier acceptance and all frozen historical evidence are retained.
