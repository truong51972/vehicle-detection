# PR #53 post-hoc review addendum — frozen Apply/Skip cohorts

This addendum addresses the [final review findings](https://github.com/truong51972/vehicle-detection/pull/53#issuecomment-6704455614)
without changing the original acquisition, policy, models, result files or
validation receipts. It supplements the [original post-hoc report](ISSUE47_POSTHOC_RESULTS.md).
All results here are **descriptive saved-match aggregation**, not independent
confirmation, a policy promotion gate or another training/inference experiment.

## Calibrated scientific conclusion and selection limit

There is **no demonstrated overall benefit under this frozen recipe/protocol**.
The full-development D−B = −0.012332 pp and D−A = −0.042925 pp are tiny single-seed
point differences on development already reused for selection. They do not show
that adaptive fine-tuning is universally worse. The larger observed motorcycle
(−1.426130 pp) and bicycle (−0.799729 pp) AP50:95 changes at fixed Adaptive inputs
are the stronger practical concern. These are still descriptive one-seed findings,
not causal or independently confirmed effects. The original prospective pilot's
failed gates remain separate and unchanged.

Adaptive **epoch 12** was selected with PR #52's matched **Raw development,
actual 640 / pad 0** selector, while the 2×2 matrix uses **nominal 960 / pad 0.5**
(actual 576×992 tensors). This matches selection protocols between training arms;
it does **not establish that epoch 12 is optimal for Adaptive/960**. Checkpoints
and routes are unchanged, with no reselection or rerun.

## Saved-only method and support

The four original `paired_stats.npz` files are verified against their acquired
SHA256 identities. Hash-bound `prepared.json`, `exports.json` and `rows.json`
supply the frozen image-level route decisions and original GT. Every development
ID belongs to exactly one route. The saved full-development native AP/counts
reproduce the original result before cohort reporting. No pixels, models,
confirmation/final-test data, training or detector inference are accessed.

Pool all original GT and native confidence/matching arrays within each cohort,
then run the unchanged native AP aggregation. This is **dataset-level five-class
macro AP**, neither AP per image nor an average of per-image AP. AP is nonlinear:
the overall AP or its delta cannot be recovered as an image-weighted average of
Apply and Skip AP. Whole-image routes retain every GT, including objects that
might not themselves need preprocessing. These aggregates do not establish
object-level causality.

Missing-class AP and five-class macro AP are **unestimated**, never zero-imputed.
Classes with fewer than five original source groups are labelled weak; the flag
is an engineering minimum, not evidence of statistical adequacy. All five classes
in both observed cohorts exceed that minimum, although motorcycle/bicycle have
far less support than car. Source groups can occur in both image-route cohorts;
the cohorts are not independent group samples.

A = Raw-trained/Raw inputs; B = Raw-trained/Adaptive inputs;
C = Adaptive-trained/Raw inputs; D = Adaptive-trained/Adaptive inputs.
AP is shown in percent; paired changes are percentage points (pp).

## Apply development cohort — 5,653 images

Frozen membership sorted-ID SHA256: `bc89b057d7c5bcd3801346152129a2d0c7cefa808c1d266f610ea69c96de36df`.
Original source groups: 3,629; canonical leakage groups: 3,629.

| Original class | Images with GT | GT instances | Source groups | Support |
| --- | --- | --- | --- | --- |
| car | 5585 | 53330 | 3598 | ≥5 source groups |
| truck | 999 | 1414 | 899 | ≥5 source groups |
| bus | 434 | 545 | 411 | ≥5 source groups |
| motorcycle | 115 | 137 | 113 | ≥5 source groups |
| bicycle | 238 | 377 | 226 | ≥5 source groups |

| Cell | mAP50:95 (%) | mAP50 (%) |
| --- | --- | --- |
| A | 19.779407 | 34.818052 |
| B | 19.709724 | 34.596920 |
| C | 19.656529 | 34.352723 |
| D | 19.846649 | 34.655317 |

| Paired contrast | Δ mAP50:95 (pp) | Δ mAP50 (pp) |
| --- | --- | --- |
| D-A | +0.067242 | -0.162735 |
| D-B | +0.136925 | +0.058397 |

### Per-class AP50:95

| Class | A (%) | B (%) | C (%) | D (%) | D−B (pp) | D−A (pp) |
| --- | --- | --- | --- | --- | --- | --- |
| car | 36.273539 | 36.254097 | 36.346262 | 36.539657 | +0.285559 | +0.266118 |
| truck | 24.696094 | 24.552083 | 24.225699 | 24.126827 | -0.425256 | -0.569267 |
| bus | 20.854607 | 20.859149 | 23.196619 | 23.730799 | +2.871650 | +2.876191 |
| motorcycle | 5.580821 | 5.631794 | 4.239740 | 4.509190 | -1.122604 | -1.071632 |
| bicycle | 11.491972 | 11.251497 | 10.274323 | 10.326773 | -0.924724 | -1.165199 |

### Per-class AP50

| Class | A (%) | B (%) | C (%) | D (%) | D−B (pp) | D−A (pp) |
| --- | --- | --- | --- | --- | --- | --- |
| car | 65.753039 | 65.685482 | 66.045790 | 66.369375 | +0.683893 | +0.616336 |
| truck | 37.717394 | 37.490841 | 36.920558 | 36.863075 | -0.627766 | -0.854319 |
| bus | 29.387662 | 29.341194 | 32.505613 | 32.948698 | +3.607504 | +3.561036 |
| motorcycle | 14.395234 | 13.706030 | 11.480303 | 12.152927 | -1.553102 | -2.242307 |
| bicycle | 26.836931 | 26.761053 | 24.811351 | 24.942507 | -1.818546 | -1.894424 |

## Skip development cohort — 8,322 images

Frozen membership sorted-ID SHA256: `a098643a5dff1ba5a27cfb28a993f9a86608401fed007449f195671a28eec98f`.
Original source groups: 5,473; canonical leakage groups: 5,472.

| Original class | Images with GT | GT instances | Source groups | Support |
| --- | --- | --- | --- | --- |
| car | 8235 | 89211 | 5434 | ≥5 source groups |
| truck | 2774 | 4551 | 2244 | ≥5 source groups |
| bus | 1325 | 1731 | 1173 | ≥5 source groups |
| motorcycle | 320 | 425 | 296 | ≥5 source groups |
| bicycle | 634 | 1053 | 582 | ≥5 source groups |

| Cell | mAP50:95 (%) | mAP50 (%) |
| --- | --- | --- |
| A | 26.416611 | 44.053838 |
| B | 26.416611 | 44.053838 |
| C | 26.384174 | 43.966936 |
| D | 26.384174 | 43.966936 |

| Paired contrast | Δ mAP50:95 (pp) | Δ mAP50 (pp) |
| --- | --- | --- |
| D-A | -0.032437 | -0.086902 |
| D-B | -0.032437 | -0.086902 |

### Per-class AP50:95

| Class | A (%) | B (%) | C (%) | D (%) | D−B (pp) | D−A (pp) |
| --- | --- | --- | --- | --- | --- | --- |
| car | 43.813808 | 43.813808 | 43.750112 | 43.750112 | -0.063697 | -0.063697 |
| truck | 30.146023 | 30.146023 | 29.998186 | 29.998186 | -0.147837 | -0.147837 |
| bus | 29.935447 | 29.935447 | 32.060920 | 32.060920 | +2.125472 | +2.125472 |
| motorcycle | 12.987143 | 12.987143 | 11.634830 | 11.634830 | -1.352313 | -1.352313 |
| bicycle | 15.200635 | 15.200635 | 14.476823 | 14.476823 | -0.723812 | -0.723812 |

### Per-class AP50

| Class | A (%) | B (%) | C (%) | D (%) | D−B (pp) | D−A (pp) |
| --- | --- | --- | --- | --- | --- | --- |
| car | 72.186156 | 72.186156 | 72.417723 | 72.417723 | +0.231568 | +0.231568 |
| truck | 45.013929 | 45.013929 | 44.420288 | 44.420288 | -0.593642 | -0.593642 |
| bus | 41.253620 | 41.253620 | 43.741781 | 43.741781 | +2.488161 | +2.488161 |
| motorcycle | 26.458598 | 26.458598 | 24.542427 | 24.542427 | -1.916171 | -1.916171 |
| bicycle | 35.356887 | 35.356887 | 34.712460 | 34.712460 | -0.644427 | -0.644427 |

## Interpretation and preservation

Apply's five-class D−B point change is **+0.136925 pp**, while Skip's is
**−0.032437 pp**. Motorcycle/bicycle AP50:95 changes are **−1.122604 / −0.924724 pp**
in Apply and **−1.352313 / −0.723812 pp** in Skip. Bus gains help the cohort means;
the Apply aggregate gain does not show improvement in the priority classes.
The same frozen trained model is used on every image, including Skip, so
training-arm differences can occur in Skip even though its input bytes are Raw.
These within-cohort contrasts do not isolate per-object causes.

For each fixed checkpoint, Skip's Raw-input and Adaptive-input AP/counts are
identical. D−B and D−A therefore coincide in Skip. This is a useful consistency
check of the frozen skip route, not independent scientific confirmation.

The [compact derived result](issue47-posthoc-routes/result.json) includes both
native AP measures, every class's original GT/source-group support, TP/FP/FN,
paired contrasts, input hashes and zero-new-acquisition counters. Its
[separate evidence index](issue47-posthoc-routes/evidence.json) leaves the original
16-file acquisition index untouched. The [comparison notebook](../notebooks/experiments/compare.ipynb)
reads this compact evidence only; it does not replay native matching aggregation.

The original report and validation receipts remain byte-exact snapshots of
commit `b133ad06553c290e4f1249455a4eed1e2e6a9ac3`. The comparison notebook is the
only pre-existing tracked file updated for this review, as explicitly requested.
A new [review validation receipt](issue47-posthoc-routes-validation.json) records
its new hash and verifies the earlier comparison hash against that commit;
it does not rewrite or invalidate the historical snapshot.

No policy or checkpoint is changed, no additional seed is run, and no S100/v4
or new PR is created. No claim of universal harm or statistically demonstrated
aggregate deterioration follows from this post-hoc single-seed study.
