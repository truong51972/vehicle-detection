# Issue #47 — adaptive local contrast with trained Raw S20/960

**Stop before confirmation and training.** The fixed adaptive candidate gains
**0.147479 AP points**, below the declared +0.2-point minimum, with a conditional
95% source-group interval **[-0.050963, +0.415904] points**. Its two-wheeler mean
gain also misses the +0.5-point gate, and supported dark-bus AP exceeds the
allowed loss. No v3 policy is frozen or promoted.

The [prospective protocol](ISSUE47_V3_960_PLAN.md), exact membership, implementation,
tests and notebook were committed at **`55d5b04` before acquisition**. This is a
new hypothesis after merged PR #52, not a retuning of the historical negative
Gamma/Sharpen or local-shadow policies. All old conclusions remain unchanged.

## Research review and distinguishing experiment

The prior exposure route skipped most two-wheeler scenes. PR #52 established a
trained 960 reference, but its dark-bus finding remained inconclusive. Neither
result establishes that visually stronger images improve detection.

The literature review in the plan distinguishes detection-trained image-adaptive
filters ([ERUP-YOLO](https://openaccess.thecvf.com/content/WACV2025/papers/Ogino_ERUP-YOLO_Enhancing_Object_Detection_Robustness_for_Adverse_Weather_Condition_by_WACV_2025_paper.pdf)),
illumination-invariant representation learning ([DAI-Net](https://openaccess.thecvf.com/content/CVPR2024/html/Du_Boosting_Object_Detection_with_Zero-Shot_Day-Night_Domain_Adaptation_CVPR_2024_paper.html)),
and efficient learned enhancement ([SCI](https://arxiv.org/abs/2204.10137)). These
are mechanisms from other settings, not evidence of BDD/YOLO26n gains. Joint
networks/extra augmentation would change the requested matched recipe, so this
bounded study tests one conservative, detection-evaluated local-contrast policy.

The operator uses contrast-limited local equalization, half-strength achromatic
offsets bounded to ±12 code levels, black/highlight protection, and original-pixel
contrast/brightness/noise/clipping gates. It can darken as well as lift. It does
not repeat Gamma/Sharpen or v2's multiplicative shadow gain. An always-on control
uses the identical operator; only the adaptive candidate can advance.

## Frozen population and baseline

Use all **1,479 remaining exploration images / 926 source groups**, including
12 negatives and 16,181 original five-class boxes. Exclusions cover both Issue #6
calibration policies, Issue #11, round2 preprocessing, round3 and Issues #45/#47,
using whole leakage/source/content groups. This corrects the narrower exposure
accounting used in recent historical pilots without rewriting their evidence.
These groups are unconsumed by the recorded policy/detection experiments, not
untouched by initial dataset construction or annotation auditing.

Class support (GT / original groups): car 15,088/918; truck 626/307; bus 259/178;
motorcycle 52/42; bicycle 156/91. Membership was not enriched after outcomes.
All 1,479 allowed original paths were audited; their full read list and original
features are compressed beside the summary. The entire remaining exploration
pool is now consumed and cannot become a future independent confirmation.

The unchanged **PR #52 Raw S20/960 best checkpoint**, selected at epoch 10, has
SHA256 `b13ec9339f7a58d55dd53c46f7e0e24330a889a761f3ee8cc4a3e9764573b2cb`.
It was independently trained on all 9,781 S20 images for 20 epochs. This study
does not retrain that baseline, reopen development for policy search, or use the
COCO-pretrained pilot detector. The baseline's 24.568% development AP is a
different population from the calibration control below.

## Native paired detection

Same detector, source IDs, original GT/negative labels, geometry and JPEG95;
FP32, nominal960, batch32, workers0, seed2026, native rectangular pad.5, conf.001,
IoU.7, max_det300, NMS-free, no augmentation. All observed tensors are 576×992,
as expected for native rectangular padding. Saved-match and native AP agree.

- Raw: **26.011103% AP50:95 / 43.738080% AP50**.
- Always-on: **26.034995% / 43.778730%**, only +0.023892 primary points.
- Adaptive: **26.158582% / 43.942740%**, +0.147479 primary / +0.204660 AP50 points.

Adaptive minus Raw class AP50:95 changes and conditional 95% intervals, in points:

- car: **-0.043305**, interval [-0.103815, +0.021350].
- truck: **+0.080937**, interval [-0.136180, +0.318671].
- bus: **+0.147976**, interval [-0.160207, +0.512532].
- motorcycle: **+0.484384**, interval [-0.365908, +1.752600].
- bicycle: **+0.067401**, interval [-0.245173, +0.352285].

The motorcycle/bicycle mean gain is **+0.275893 points**, below the +0.5 gate.
At conf≥.25 / matching IoU.50, motorcycle TP/FP/FN changes **9/8/43 → 10/7/42**;
bicycle **46/18/110 → 47/17/109**. Each gains one TP; this does not establish a
reliable solution to missed two-wheelers. Across all classes, TP/FP/FN changes
**8,975/2,197/7,206 → 9,013/2,263/7,168**, so the extra TP coexist with extra FP.
Native F1-selected precision/recall use a separate operating point.

## Routes, cohorts and uncertainty

The adaptive route processes **584 images (39.49%)** and skips **895 (60.51%)**.
Processed images contain 17/52 motorcycle and 26/156 bicycle GT. Controls match
adaptive JPEG bytes exactly in their respective apply/skip populations; paired
label bytes and original geometry were checked for every arm/image.

The prospective route-control test passes: across five common supported classes,
always-on minus Raw AP change is **+0.766622 points in apply** versus **-0.177045
in skip**, a **+0.943667-point separation**. This is a population AP association,
not per-image AP, a causal proof or independent confirmation. The route avoids
some blanket-filter damage, but this signal cannot override the failed gates.

**Dark bus** has 20 GT / 16 source groups and loses **1.616524 AP points**, beyond
the allowed 1-point loss. Its interval [-7.552324, +1.118486] is wide; this is a
failed point-tolerance gate, not proof of statistically established harm.
TP/FP/FN remains 3/2/17. Dark motorcycle/bicycle fixed-confidence TP remain 1/2
despite their AP point gains. Their tiny 5/6-group support limits interpretation.

Each all/dark/bright cohort uses 1,000 seed-2026 paired original-source-group
draws, fixed supported classes, no imputation/redraw. Primary has 1,000 valid /
0 missing-class draws; dark has 995 valid / 5 excluded draws; bright has 1,000
valid / 0 excluded draws. All counts,
class/cohort intervals and complete compressed draws remain in the evidence.
Intervals condition on this detector/population and omit training-seed and model
selection uncertainty; they have no multiplicity adjustment. Small-two-wheeler
image-cohort AP retains every GT in those images and is not AP-small. Saved native
matches do not identify individual missed-box indices or FP geometry.

## Cost and decision

Direct eight-thread CPU preprocessing mean / p95:

- Raw diagnostic control: **2.515 / 3.281 ms**.
- Always-on: **10.401 / 12.510 ms**.
- Adaptive: **5.811 / 11.913 ms** — passes the 10/25ms gate.

Each arm has 4,437 measured calls: three repeats on every original image, rotated
order after warmup, including features and excluding reads/JPEG/inference equally.
Raw timing deliberately includes the same quality diagnostics; it is not the cost
of a deployment Skip path. Feature extraction and the median-residual noise proxy
do not establish which individual objects are recoverable.

Native inference means are **2.979 / 2.872 / 2.873 ms/image** for Raw/always/adaptive.
Synchronized validation-batch mean/p95 are **195.423/251.468**, **183.838/241.868**,
and **184.616/253.009 ms**, 47 batches each. Native validation-call walls are
21.025/17.274/17.323 seconds. These sequential one-host setup/cache-sensitive
measurements are descriptive; batch p95 is not per-image p95 or a deployment SLA.

Preserve every failed gate: primary minimum gain, positive primary interval,
two-wheeler mean gain, and supported dark-bus regression. Cost and route-control
success do not authorize candidate freezing, runner-up promotion or training.
There is no threshold change, additional candidate, retry or sample expansion.

## Completed conditional lifecycle and preserved evidence

The scientific acquisition uses exactly **4,437 inference image passes** and
**zero new training, confirmation or development passes**. Original confirmation
and final-test pixels/headers/hashes/predictions stay sealed. The 9,781 S20 and
13,975 development memberships, completed v0/v1/v2 and PR #48–#52 artifacts remain
unchanged. This fixed frozen-detector policy failed; consistent transformed-domain
fine-tuning was not tested and its result must not be inferred from this failure.

All four [v3 notebooks](../notebooks/experiments/v3/README.md) run independently
from saved prerequisites. 01 contains research and native measurements; 02/03/04
verify the stopped study and record skipped preparation/training/evaluation with
no override. No v3 config, S20 pixels, checkpoint or development AP is fabricated.
The [comparison](../notebooks/experiments/compare.ipynb) alone presents historical
S20/640 v0/v1/v2, PR #52 Raw960 development and current calibration evidence, with
clear population/selector attribution. It triggers no acquisition or resampling.

The byte-exact [old Gamma/Sharpen notebook](../notebooks/research/issue47/04_gamma_sharpen_pilot.ipynb)
is preserved alongside unchanged historical plans/results. The current
[result](issue47-v3-960/result.json), [evidence index](issue47-v3-960/evidence.json),
native metrics/measurement receipts and deterministic class-count example IDs
are committed. Full original membership/features/read lists and every bootstrap
draw are losslessly compressed and included in that index. Inspect them with
`gzip -dc docs/issue47-v3-960/all_interval.json.gz`; no resampling is necessary.
Original images, weights, exports, per-image native matching arrays, executed
acquisition notebook and log stay under ignored local roots.

Historical preservation checks verify **383** unchanged tracked evidence/config/
notebook hashes and **214,854** unchanged local run/weight size/mtime records.
This is not a cryptographic rehash
of all historical pixels. Source grouping is inferred and residual near-duplicates
may remain; these are single-seed remaining-pool exploration results, not a final
test or independent generalization claim. Issue #47 may be closed as a bounded negative study once this PR merges;
new mechanisms require independent justification. No S100 or v4 is started.

Validation: **214 tests passed**, Ruff lint/format and whitespace checks passed.
Fresh-kernel checks execute all four v3 defaults and the updated comparison from
committed prerequisites with models, inference, original-source acquisition and
preprocessing replay blocked. Original PR #52 archives and exact summary
equivalence also verify. Synthetic native checks are software evidence only.

The actual opted-in acquisition notebook completed in **688.760 seconds**;
the acquisition function took **661.603 seconds**, excluding earlier baseline/
membership verification and original annotation loading. Both include their
respective CPU reporting work; neither is GPU busy-time or training cost. There
was one real acquisition, no retry or interrupted reporting continuation.
The separate [post-execution validation receipt](issue47-v3-960/validation.json)
binds the unchanged acquisition evidence index, local saved-match/export hashes,
executed notebook and software checks; it does not rewrite the scientific index.

## Issue #47 closure assessment across completed studies

The broader #47 acceptance criterion allows a documented negative conclusion instead of
forcing S20 preparation and training without a defensible policy. The new v3/960
study is one bounded hypothesis, **not** an exhaustive search of adaptive methods.
To close the issue responsibly, interpret it together with the prior committed
experimental evidence rather than describing the single CLAHE treatment as the
only adaptive route considered:

| Evidence | Detector / population / objective | Observed finding and decision |
| --- | --- | --- |
| [Issue #30, round 2](ISSUE30_RESULTS.md) | Pretrained, controlled clean/corruption scenarios, independent exploration/confirmation | Denoise-5 looked +0.6736 AP points over Skip in exploration but was **-0.4623** in confirmation; reject. Ordered Gamma/CLAHE and Sharpen/Bilateral pairs did not establish robust superiority. |
| [Issue #30, round 3](ISSUE30_RESEARCH_ROUND3.md) | Pretrained, new 256-image slice / five-scenario sentinel | Previous route's +0.59575-point old-slice advantage over v2 became **-0.01621** on new groups under matched sentinels; fixed noise-range bilateral also failed. |
| [Issue #45, local shadow](ISSUE45_RESULTS.md) | Pretrained, 256 natural calibration images | +0.1145 points over Raw, but below +0.2; dark-car regression and CPU mean-cost gates failed. |
| [Earlier Issue #47 Gamma/Sharpen](ISSUE47_RESULTS.md) | Pretrained, 1,027 natural calibration images | +0.006313 points over Skip, failed primary/uncertainty and subgroup checks; ordered pair not supported as a replacement. |
| **Current PR #53 local contrast** | **Trained Raw S20/960**, 1,479 different natural calibration images | **+0.147479** points over Raw; primary minimum and interval, two-wheeler mean, and dark-bus regression gates fail. |

These rows have different detectors, groups, conditions and estimands. Only
within-study paired deltas are interpretable; do **not** rank the absolute AP
levels or pool them into a synthetic meta-score. They collectively establish
repeatedly weak or non-transferable gains **for the tested policies and budgets**.
They do not establish that enhancement is impossible or that a jointly trained
enhancer/detector would fail.

### Failure diagnosis and why no more exploration is authorized

- The current image-quality router applies local contrast to **584/1,479** images,
  but only **17/52 motorcycle** and **26/156 bicycle** ground-truth instances
  occur in processed images. Global low contrast is therefore a poor coverage
  proxy for the rare-object objective; the feature-to-detection-response link
  remains insufficient even though the route-control association passes.
- Motorcycle and bicycle each gain only **one TP at confidence 0.25**, while
  total false positives increase **2,197 to 2,263**. Improved confidence ranking
  or localization can affect AP without changing the fixed-threshold TP count;
  the saved matching evidence does **not** identify every missed ground-truth box.
- Dark-bus AP loses **1.6165 points** on just **20 GT / 16 source groups**.
  The wide interval overlaps zero, so this is a predeclared point-tolerance
  failure, **not** evidence of reliably harmful local contrast.
- Earlier fixed routes also skipped most small-two-wheeler scenes; the low-sharpness
  examples were sparsely supported. Overall quality proxies cannot reliably
  identify recoverable details at the object/box scale. That is a mechanistic
  limitation to investigate separately, not permission to choose a new threshold.
- All **1,479** remaining eligible exploration images are now consumed. The
  historical exposure audit and group-disjoint membership prevent quietly
  reusing them for a new independent experiment. The reserved confirmation
  and sealed final test are not fallback exploration sets.

### Acceptance and final disposition

The prospective design, native paired metrics, class/cohort error accounting,
source-group uncertainty, matched CPU cost, routing control, immutable negative
decision, complete standalone `01`-`04` notebook lifecycle, read-only comparison,
hash-based preservation and CI validation are documented within this PR.
**No policy passes the declared advancement gates**, so the conditional
confirmation, S20 preparation, independently initialized v3 training and
trained-v3 development evaluation are correctly **not performed**.

**Recommendation: close Issue #47 as completed with a negative, bounded research
result upon merging PR #53.** Keep v0/v1/v2 and PR #52 as attributed baselines;
do not promote v3, consume confirmation/final, or create another PR merely to
force a train/eval metric. Further research into object-aware preprocessing,
learned enhancement, or transformed-domain co-training would require a separately
justified question, a legitimately fresh population, and a new prospective
protocol; those untested directions are outside this issue's closure claim.
