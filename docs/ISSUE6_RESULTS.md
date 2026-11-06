> Historical workflow record. Current entrypoints and replay paths: [issue 41 migration](HISTORICAL_REPLAY.md).

# Issue #6: nested Raw / Adaptive pilots

This implements [issue #6](https://github.com/truong51972/vehicle-detection/issues/6)
on the merged issue #5 protocol. The original v1 execution is preserved in
[the evidence notebook](https://github.com/truong51972/vehicle-detection/blob/51687b273e7e87b5315d84f5b67202d4cc746992/notebooks/archive/phase2/evidence/02_adaptive_preprocessing_v1.ipynb).
The [review follow-up](ISSUE6_REVIEW.md) documents v2; figures and measurements in
this report continue to describe v1 at revision `7380eed`.
No detector inference, optimizer steps or development/final-test image inspection
are part of this issue. Historical YOLO11 notebooks and issue #5 exports remain
independent; the owner's existing historical notebook outputs were preserved.

## Immutable nested populations

The 48,903-image locked training pool supplies **S10: 4,890 images in 2,974 groups**
and **S20: 9,781 images in 5,915 groups**. S10 is strictly contained in S20 by
image ID. Both retain complete issue #5 groups, including inferred sequence and
duplicate-candidate grouping. Nominal fractions are 9.9994% and 20.0008%; groups
are never divided to force an exact percentage. S40 is supported but not selected.

Selection searches 128 seeded group orders (seed 2026), minimizing combined
nested-subset drift in class image presence and instances, box sizes, negatives,
weather, time of day, scene, condition intersections and class × condition counts.
Only training-pool evidence enters selection. Evaluation roles are unchanged.
This finite search approximates natural distributions, not perfect balance.

S10 has 53,696 vehicle instances and 34 negatives; S20 has 107,439 instances and
73 negatives. The largest class-presence/condition/negative marginal drift is
**1.870 percentage points** for S10 and **0.699 pp** for S20 (clear weather).
No audited feature with expected support at least five has zero observed support.
There are nevertheless 126/105 feature counts below five in S10/S20, including
60/36 zero counts. Feature counts include image presence, box instances and
intersections; they are not independent statistical populations.

The train class remains fragile: S10 has **5 images / 5 objects**, S20 has
**11 images / 13 objects**. Sparse intersections require parent-marginal reporting;
no oversampling or claim of reliable rare-stratum inference follows.

The local immutable `runs/issue6-v1/subsets.json` contains IDs, groups, source-row
fingerprints, summaries, diagnostics and algorithm/source hash. The compact
[membership file](issue6/subset_membership.csv.gz) covers the full training pool
with S10/S20 flags; [support CSV](issue6/subset_support.csv) records expected and
observed counts, drift and rare/zero support. Full-parent zero combinations remain
available in the issue #5 class × condition audit.

Parent protocol: `680090b8eef098fbe10c77e632aacbfd3d6854d8489d7518e20f062e032b6e44`.
Nested-subsets identity: `8a9ba46aa120f0e1d113d4bd907bc8ef5e82ab9bb3bd533d863812deb3f30b4f`.

## Calibration-only frozen policy

A deterministic 256-image sample comes exclusively from the 6,985-image calibration
role. Pixel-derived brightness, contrast, noise and Laplacian sharpness use the
existing maximum-512-pixel metric resolution. Thresholds are calibration quantiles
clamped to conservative bounds. Two mild strengths per operator are compared with
skip on at most 12 actually routed calibration images, after matched JPEG95 encoding.
The recipe is frozen before any pilot outcome is inspected.

The [frozen policy](issue6/policy.json) records sample IDs, hashes, measurements,
candidate supports/losses, recipe, code/library identities and limitations:

- Brightness below **0.12**: Gamma **0.85**, supported by 12 candidate images.
- Otherwise contrast below **0.12**: CLAHE clip **1.5**, grid **8**, supported by
  only five candidate images.
- Denoise and Sharpen: **disabled**, because no images reached those routes in
  the calibration sample under the initial bounded thresholds and fixed priority.
- Otherwise skip. At most one operator runs; combinations are deferred.

Policy identity: `2e487f9a9ec7c5b2589ded366df7e1bcadb06f217b81295222e3d57f9a6cdb9f`.
No environmental ground truth enters routing. Labels are joined only afterward
for descriptive subgroup plots. Unsupported operators are not claimed calibrated.

The proxy loss penalizes darkness, flat contrast, estimated noise, very low
sharpness and black/white saturation. Its fixed weights are heuristic. Candidate
loss reduction is not detection improvement, and aggregate changes can conceal
washed-out night scenes, noise amplification, lost texture or halos. The small
calibration sample and five-image CLAHE support limit generalization. Under the
new v2 minimum of 12, the historical CLAHE fit is **uncalibrated/insufficiently
supported**; historical Denoise/Sharpen are also uncalibrated. V1 nevertheless
applied its recorded CLAHE recipe; this disclosure does not change those artifacts.

## Paired exports and diagnostics

Only **S10** is materialized, under `runs/issue6-v1/datasets/S10/{raw,adaptive}`.
Both branches decode the original source and encode JPEG **quality 95** with the
same OpenCV version. Raw has no enhancement. Native-converted labels are copied
byte-for-byte; image IDs, source hashes, dimensions, boxes and split assignments
are shared. Original source files remain unchanged. S20 is a frozen manifest only.

The official BDD→Scalabel→COCO→YOLO converter verifies selected source boxes within
0.01 pixel. Deep verification hashes/decodes both exports and checks exact pairing.
Pair/export identities bind the parent, subset, recipe, source/label content,
encoder and relevant implementation. Changed identities require a fresh root;
partial exports can safely reuse verified completed files.

The notebook plots paired decoded-JPEG distributions/deltas, decisions by weather,
time, scene and quality strata, filter/skip counts, and offline mean/p50/p95 costs.
Best/worst proxy changes, the largest saturation change and a skipped control are
shown side-by-side. Those rankings identify visual inspection candidates, not
actual detector successes/failures. Source-image examples and per-image large
tables stay gitignored; compact diagnostic charts and provenance live in docs.

The [machine-readable execution summary](issue6/summary.json) records measured
routing, adverse proxy counts, timing, image storage, environment and file hashes.
The notebook completed on **2026-11-03**, using Python 3.11.14, NumPy 2.4.6 and
OpenCV 4.14.0 with one OpenCV thread on the 20-logical-CPU WSL host:

- **3,690 skipped (75.46%)**, 1,107 Gamma, 93 CLAHE; no multi-filter outputs.
- All 1,200 transformed images improved the chosen aggregate proxy loss, while
  skipped images had zero paired change. This is not evidence of detection gain.
- Individual adverse proxies remain: white saturation increased in **1,056**
  transformed images (maximum **0.0326 percentage points**), estimated noise rose
  in **5**, and sharpness decreased in **541**. A lower aggregate loss can hide
  those changes. The best-proxy example still contains very dark road detail.
- Adaptive metric-extraction/filter cost: **p50 2.980 ms, p95 6.255 ms** per image;
  Raw metric extraction: **p50 2.813 ms, p95 3.254 ms**. Full measured preparation
  p50/p95: Adaptive **8.189/10.768 ms**, Raw **7.496/8.953 ms**. These per-image
  intervals exclude conversion, preflight and whole-workflow verification.
- Encoded images occupy **0.568 GiB Raw + 0.584 GiB Adaptive**, plus labels,
  journals, conversion and reports. These are offline CPU costs, not GPU/model,
  online end-to-end or deployment-throughput measurements.

[Storage estimates](issue6/storage_estimate.csv) use 16 selected-training images,
p90 encoded size plus 25% headroom and 1 GiB for labels/conversion. This is a
sampled estimate, not a worst-case bound. It gates export without materializing
the full training pool or additional fixed-filter branches.

## Checks and boundaries

Local fixture checks: **38 tests passed**, including all existing native-converter
and pretrained-adapter checks. New tests cover nested whole groups, deterministic
multi-label selection, calibration-only access, routing/no-op/shape/dtype/source
invariance, byte-identical no-op JPEGs, exact branch pairing, export reuse and
rejection of changed policy, membership, annotations and tampered output labels.
Ruff lint/format, source/wheel build and whitespace checks pass. CI/review have
not been run remotely for this branch.

No S20/S40 exports, model training or detection-benefit measurement are claimed.
No development-validation or final-test pixels were opened by this workflow.
Issue #7 supplies partial fine-tuning and detection comparisons; issue #8 releases
the final test after protocol/model/recipe lock. Inferred video groups and equal
dHash safeguards still do not prove absence of all near duplicates, and final-test
content leakage checks remain deferred to controlled release.
