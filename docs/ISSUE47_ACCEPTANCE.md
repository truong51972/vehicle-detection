# Issue #47 — final acceptance of PR #53

**Verdict: complete the issue as a bounded negative study when PR #53 merges.**
The current adaptive policy fails four frozen gates. The tested historical
policies do not establish a confirmed replacement either. This permits closure
under the issue's negative-result branch; it does not establish universal failure
of image adaptation or predict untested transformed-domain joint training.

This review uses committed reports and already-saved native matching arrays.
It acquires **zero new source pixels, inference passes or bootstrap draws**.
Confirmation and final-test pixels remain sealed. All 1,479 current exploration
images are consumed and cannot become independent validation. No S100, v4,
new policy search, additional PR or forced training is part of acceptance.

## Acceptance mapping

| Issue #47 requirement | Final assessment and evidence |
| --- | --- |
| Review failures and propose bounded mechanisms | Complete. The [cross-study synthesis](ISSUE47_V3_960_RESULTS.md#issue-47-closure-assessment-across-completed-studies) includes #30/PRs #37–#39, local-shadow pilot and S20 extension, Gamma/Sharpen, and current trained-Raw960 local contrast. The [prospective plan](ISSUE47_V3_960_PLAN.md) distinguishes task-trained/learned alternatives and fixes one candidate plus controls. |
| Test operators, ordered pairs, quality routing and benefit prediction | Complete within the declared budgets. Historical ordered pairs and quality routes retain negative and unconfirmed findings. Current Raw/always-on/adaptive controls isolate the pixel route's association with benefit. No post-hoc runner-up is promoted. |
| Freeze population, compute/latency/regression/support/stop gates before inference | Complete at `55d5b04`; exact 1,479-image / 926-group membership excludes documented historical exposures and all confirmation groups. The declaration, operator, tests and 01 notebook remain unchanged. |
| Native paired metrics, class/cohort support, errors, uncertainty and CPU cost | Complete. Original five-class GT, negatives and JPEG95 are retained; native and saved-match metrics agree. [Frozen result](issue47-v3-960/result.json), complete draw archives and [saved-only review](issue47-v3-960-acceptance/saved_audit.json) retain all failed gates. |
| Select/freeze at most one policy and independently confirm only if warranted | Negative branch complete. No current candidate meets the frozen advancement gates, so no policy is frozen and confirmation is not consumed. PR #38's former exploration leader remained unconfirmed; its later fixed transfer check failed. |
| Conditional config, S20 preparation and independent matched training | Correctly skipped. [02](../notebooks/experiments/v3/02_train_prepare.ipynb) and [03](../notebooks/experiments/v3/03_train.ipynb) verify this terminal decision. No v3 config, export, checkpoint or training history is fabricated. Existing 9,781 S20 / 13,975 development IDs and Raw960 recipe remain intact. |
| Conditional own-policy development evaluation and separate comparison | Correctly skipped in [04](../notebooks/experiments/v3/04_eval.ipynb). [compare.ipynb](../notebooks/experiments/compare.ipynb) reads separately attributed historical S20/640, Raw960 development and current calibration evidence; no v3-trained row or acquisition is introduced. |
| Independent notebook defaults, integrity, focused/full tests and review closure | All four defaults and comparison are checked in fresh kernels from committed prerequisites with source/model/acquisition paths blocked. Original evidence/receipt hashes and historical files are verified; new tests cover the remaining clipping and saved-data review findings. Final software/CI status is recorded in the separate [acceptance receipt](issue47-v3-960-acceptance/validation.json) and PR discussion. |

Only within-study paired deltas are used in the synthesis. Ten-scenario
family-balanced objectives, five-scenario sentinels, natural calibration and
trained development have different estimands/populations; there is no pooled
score, matched cross-study AP claim or cross-run latency ranking. Focused checks
verify the quoted deltas against their saved sample/candidate keys.

## Why most two-wheeler GT was skipped

The deterministic [saved-match audit](issue47-v3-960-acceptance/saved_audit.json)
binds the unchanged feature/membership archives, original receipt/index and all
three local matching-file hashes. Native target-class arrays agree across arms.
It recomputes existing all/dark/low-sharpness class AP and fixed-confidence
TP/FP/FN once from those arrays, with exact support/count agreement and AP agreement
within 1e-12. It does not load images, model weights, a detector or new metadata.

- Motorcycle: 17/52 GT occur in 14 processed images; the other **35 GT in 30
  images all fail contrast <0.20**, with no other failed gate.
- Bicycle: 26/156 GT occur in 22 processed images; the other **130 GT in 76
  images all fail contrast <0.20**. Of those, 129 fail contrast alone; one also
  fails white-fraction <0.10. No remaining guard rejects a bicycle image.

These counts weight each image by its GT count. Overlapping gate counts are not
independent and must not be added. They identify which frozen *image decision*
skipped the GT-containing frame, not why any individual object was missed or
whether changing the threshold would help. The median-residual noise proxy and
global contrast cannot identify object-local recoverability from these records.
No threshold is revised from this descriptive audit.

## Dark-bus and low-sharpness limits

Dark bus has **20 GT in 16 source groups**. The AP50:95 point change is -1.616524
points, exceeding the fixed 1-point tolerance; its conditional interval is wide
[-7.552324, +1.118486]. TP/FP/FN stays 3/2/17 at conf .25 / IoU .50. This remains
a failed point-tolerance gate, not statistically established harm or a claimed
cause. The overall dark macro cannot override that class gate. Dark motorcycle
and bicycle support is only five/six groups; their AP point gains do not create
additional fixed-confidence TP.

The low-sharpness cohort contains **15 images**, with car 71 GT / 12 groups,
truck three GT / two groups, motorcycle one GT / one group, and no bus/bicycle
GT. Only car crosses the five-group engineering floor. Its macro covers three
observed classes, not five; absent AP is not zero. The lone motorcycle has a
Raw AP50:95 of 59.7% but zero TP at conf .25, illustrating how a sparse ranking
statistic can differ from fixed-threshold coverage. This is a Laplacian-defined
image subset, not a verified physical-blur diagnosis. It cannot establish rare-
class blur robustness, object-local cause or size-specific AP-small.

## Operator bounds and saturated channels

The requested shared signed offset is rounded after black/highlight protection,
then B/G/R are individually clipped to [0,255]. Synthetic checks verify unchanged
shape/dtype/source arrays and **absolute per-channel changes <=12** across full
uint8 ranges. Gray luminance <=8 or >=247 receives zero offset; this protection
uses grayscale luminance, not the maximum individual color channel.

Channel differences are preserved only where the shared offset avoids gamut
clipping. A saturated colored highlight can have grayscale luminance well below
247. For example, uniform BGR (0,160,255) becomes (2,162,255) in the pinned
environment: the red channel clips while the other channels increase. Negative
offsets similarly clip a zero-valued channel while reducing others. Thus the
operator is **not unconditionally chroma-preserving**, although the ±12 bound
and grayscale-extreme protection hold. Tests cover both clipping signs. This
clarifies an existing limit; no operator, gate, result or frozen test is changed.

## Reproduction and preservation

Run the saved-only audit on the acquisition host, then the software checks:

```sh
.venv/bin/python -m vehicle_detection.v3_acceptance
.venv/bin/pytest -q
.venv/bin/ruff check .
.venv/bin/ruff format --check .
git diff --check
```

The audit command requires the already-ignored native match files; it never fills
missing inputs. Its committed compact output is readable without them. The full
test suite also includes synthetic CPU model fixtures; these are software checks,
not new study inference/training. A dedicated local test blocks pixel reads,
weights, model construction and inference while reproducing the saved audit.

The original acquisition index and validation receipt remain byte-identical;
new explanatory analysis and final validation use `issue47-v3-960-acceptance/`.
All original tracked files except the explicitly revised non-frozen closure prose
remain unchanged. Historical local image/model files are checked against the
original size/mtime inventory without opening their pixels. That inventory is
not a cryptographic rehash of every historical image, and inferred source grouping
cannot exclude every residual near-duplicate. No seed/selection/generalization
claim is added by this acceptance review.
