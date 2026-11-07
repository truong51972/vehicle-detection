# Issue #47 — adaptive preprocessing after PR #52

Prospective bounded study, 2026-11-07. The user's continuation authorizes this
new lifecycle; it does not reopen or retune any historical failed gate. Commit
this protocol, exact membership, implementation and notebook before new pixels
or native inference. Use new `docs/issue47-v3-960` and `runs/issue47-v3-960` roots.
An existing acquisition attempt blocks reruns, including incomplete attempts.

## Review and research decision

PR #48's Gamma/Sharpen route changed 246/1,027 images, gained only 0.0063 AP
points, and failed truck/support/uncertainty checks. Most motorcycle/bicycle
frames were bright and skipped. PR #49's diagnosis cannot attribute individual
missed GT boxes from saved matches. PR #50's short resolution-training contrast
failed uncertainty and dark-bus gates. PR #52 completed S20/20 epochs and found
a +2.170-point incremental resolution-training effect, but its dark-bus FN
contradiction made the assessment inconclusive. These conclusions stay intact.

The PR #52 **Raw train960 checkpoint**, selected at epoch 10 on fixed actual-640
Raw development, is the frozen detector for this study. Its SHA256 is
`b13ec9339f7a58d55dd53c46f7e0e24330a889a761f3ee8cc4a3e9764573b2cb`.
Its 960 development mAP50:95 is 24.568%; that is a development result, never
the expected calibration control AP. No baseline training or development
inference is repeated for candidate search.

Research reviewed before measurement:

- [ERUP-YOLO, WACV 2025](https://openaccess.thecvf.com/content/WACV2025/papers/Ogino_ERUP-YOLO_Enhancing_Object_Detection_Robustness_for_Adverse_Weather_Condition_by_WACV_2025_paper.pdf)
  learns image-adaptive pixel/local filters for detection under adverse weather.
  Its YOLOv3 results motivate evaluating local contrast with a trained detector;
  they do not predict gains here. We do not reproduce its learned model.
- [DAI-Net, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Du_Boosting_Object_Detection_with_Zero-Shot_Day-Night_Domain_Adaptation_CVPR_2024_paper.html)
  learns illumination-invariant representations. Joint training is promising but
  would change the required matched recipe and add a new network. Defer it.
- [SCI](https://arxiv.org/abs/2204.10137) studies efficient learned low-light
  enhancement, including face detection/segmentation. It does not establish
  BDD five-class gains. No new checkpoint selection/download is justified here.
- [OpenCV's CLAHE documentation](https://docs.opencv.org/4.13.0/d5/daf/tutorial_py_histogram_equalization.html)
  explains local equalization, contrast limiting and noise amplification. This
  motivates a single conservative contrast/noise-gated comparator, not a grid.
  Earlier Issue #30 used unrestricted LAB CLAHE on a different frozen detector;
  this tests bounded, highlight-protected achromatic changes at trained 960.

These are external mechanisms, not experimental evidence for our policy. The
previously linked PubMed/AAAI pages were not readable in this review; no new
claim depends on them. The main hypothesis is **recoverable local structure**,
not that a brighter or visually preferred image must improve detection.

## Fixed candidates and falsification

Exactly three arms: `raw`, `always_local` (mechanism/control), `adaptive_local`
(the only candidate eligible to advance). No Gamma, Sharpen, ordered compositions,
threshold search, runner-up promotion, synthetic grid or extra resolution.

Compute existing `quality()` features from original pixels at maximum side 512.
The adaptive action is local contrast only when contrast <0.20,
0.04 < brightness <0.80, noise <=0.01, black fraction <0.30 and white fraction
<0.10; otherwise Skip. These constants are hypotheses fixed now, not fitted
from the new pixels. Neither scene labels, boxes, detector outputs nor class
presence participates in routing.

The local operator computes grayscale luminance, OpenCV CLAHE clip 1.5 / 8x8
tiles, then half the equalization difference, clipped to [-12,+12] code levels.
Multiply by `clip(min(Y-8,247-Y)/32,0,1)` to protect black/highlight extremes.
Round this offset once and add the same signed offset to B/G/R, with uint8 gamut
clipping. Geometry and source arrays stay unchanged. This permits darkening as
well as lifting and differs from v2's multiplicative illumination-map gain.
Both transformed arms use exactly this operator. Raw and Skip share JPEG95 bytes.

The always-on control tests whether routing predicts *detection* benefit rather
than merely selecting low contrast. The method is detection-evaluated, not a
jointly optimized enhancer. A failed frozen-detector gate cannot establish that
consistent fine-tuning would fail; it stops this conservative lifecycle only.

## Population and leakage audit

Use unchanged five-class `configs/data.yaml` / `configs/splits.csv.gz` and the
round2 evaluation partition. Exclude whole ID/leakage/source/SHA256 groups from
both historical calibration policies (`issue6`, `issue6-v2`), Issue #11's pilot,
round2 preprocessing (all sample roles), round3, Issue #45 and Issue #47. Validate
the historical membership against canonical metadata. Earlier pilots' narrower
exclusions are historical, not a justification for reusing exposed groups.

Use **all remaining 1,479 exploration images / 926 original source groups**;
sorted-ID digest `877b13d5f6a31fdd6af8d55f6af3b938dfafe99774a326277050d147ef8f5b12`.
Support: car/truck/bus/motorcycle/bicycle = 15,088/626/259/52/156 GT, in
918/307/178/42/91 source groups. No outcome-based enrichment or sample expansion.
Commit exact compact membership before acquisition; original GT comes only from
annotation metadata. Training/development/confirmation/final source images are
not opened by exploration. Canonical cross-role group/hash validation must pass.

If the one candidate qualifies, freeze its code/config and receipts in Git
**before** choosing/opening confirmation pixels. Select a whole-group prefix
of at least 1,024 and at most 1,040 previously unexposed confirmation images,
ordered by `digest([2026,"v3-960-confirmation",group])`, excluding every historical
exposure and exploration token. There are 2,807 eligible confirmation images
before selection. Audit >=20 source groups per primary class; insufficient
support stops inconclusive. Commit exact confirmation membership before opening
its pixels. This single confirmation supersedes older unused calibration
reservations only for these fixed groups; old reserved plans remain unread.
Final-test pixels/headers/hashes/predictions always remain sealed.

## Native measurements and gates (AP fractions internally)

Native YOLO26n FP32, nominal 960, batch 32, seed 2026, workers 0, conf .001,
IoU .7, max_det 300, rectangular native pad .5, NMS-free, no augmentation.
Use `ExperimentValidator` and native matching/AP; verify IDs, exact original
GT/label bytes, architecture/taxonomy, actual tensor shapes, checkpoint hash and
native/saved-match agreement. Export original sources once per arm at JPEG95
under owned directories; native caches must not touch historical files.

Report all/dark (<.12)/bright/low-contrast/low-sharpness/small-two-wheeler image
cohorts and apply/skip routes, retaining all GT in each image. Cohort AP is not
AP-small. Record all five per-class AP50/AP50:95, GT/TP/FP/FN at conf .25 / IoU .50,
supported and absent class effects, deterministic example IDs and route coverage.

For all/dark/bright, use **1,000** paired original-source-group bootstrap draws,
seed 2026, fixed supported classes, minimum five source groups, no imputation or
redraw. Preserve complete draws in gzip and summaries in JSON. Primary requires
all five classes and >=20 source groups each. Missing classes are reported;
>=950 valid draws are required. Exploratory intervals are conditional and
unadjusted; confirmation is one preselected policy, still one training seed.

Every advancement/confirmation must pass:

1. Adaptive minus Raw five-class mAP50:95 >=.002 (+0.2 points), mAP50 >=-.001,
   positive primary 95% interval lower bound, >=950 valid primary draws.
2. Motorcycle/bicycle mean AP50:95 gain >=.005 (+0.5 points); neither individual
   point change <0 and neither individual primary CI lower < -.01.
3. No all-class AP loss >.02; no supported dark/bright/route class AP loss >.01.
   Unsupported cohorts/classes remain explicitly inconclusive, not zero or proof
   of safety. A supported dark/bright class with >5% more FN (minimum +2 FN)
   requires a positive class CI lower bound; otherwise stop inconclusive.
4. On at least three common supported classes in apply and skip populations,
   always-on minus Raw macro gain in apply must exceed that in skip by >=.002.
   Apply macro gain >=.002; adaptive full-population AP >= always-on minus .001.
   This is the prospective prediction-of-benefit check, not per-image AP.
5. Direct matched CPU preprocessing, eight threads, one warmup, three repeats,
   rotated arm order on every selected source: adaptive mean <=10ms/p95 <=25ms.
   All arms include feature extraction; exclude decode/JPEG/inference/verification
   equally. Record native stage means, synchronized validation-batch mean/p95 and
   wall time separately. Batch p95 is not per-image p95 or a deployment SLA.

At most 4,437 exploration inference image passes (three arms) and 3,120
confirmation passes (three arms needed to repeat routing controls). No retry,
threshold change, additional method, extra seed or extra population after failure.
CPU-only reporting may continue from hash-verified saved matches without any
new pixel/inference pass; publish an interruption and continuation receipt first.

## Conditional lifecycle and presentation

Only after confirmation: freeze `configs/v3.yaml`; prepare exactly 9,781 S20
training and 13,975 development images, original five-class GT/negatives and
JPEG95. Independently initialize pinned COCO YOLO26n (do not continue Raw best),
20 epochs, square 960, batch16, AdamW lr .001/lrf .01, cosine, warmup3, nbs64,
weight decay .0005, freeze11, FP32, seed2026, workers0 and disabled augmentation.
Reuse PR #52's fixed actual-640 Raw-development selector (pad0, batch32) and
native fitness/best receipt. Validate every pass/tensor; no native recovery,
batch reduction, resume or schedule retuning. Evaluate one selected v3 checkpoint
on its matched processed development at nominal960/pad.5; compare with verified
PR #52 Raw960 saved development evidence, never a replay. Record paired source
uncertainty, per-class/cohort errors and costs without claiming independent test.

If exploration or confirmation fails, no policy config/export/training exists.
Complete 02/03/04 with explicit hash-verified skipped status and enforced gates,
not fabricated metrics. Keep cross-version presentation exclusively in
`compare.ipynb`, separating historical S20/640, PR #52 Raw960 development and
new trained-detector calibration evidence. Preserve the byte-exact old v3
notebook in the research directory; never alter historical evidence manifests.

Default fresh-kernel execution reads committed compact evidence only. Validate
operators, gate failures, group exclusions, native synthetic integration and
read-only notebook behavior with focused tests, Ruff and preservation receipts.
Hash historical tracked evidence/configs; inventory local historical size/mtime
without rereading their pixels. No S100, v4, new dependency, registry or trainer.
