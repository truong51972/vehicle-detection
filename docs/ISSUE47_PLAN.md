# Issue #47 — fixed adaptive exposure/detail pilot

This declaration and its implementation must be committed before opening new
exploration pixels or running inference. Starting revision: `07dd9cd`. This is
an independent v3 research direction, not the historical “Adaptive V3” policy.

## Evidence and mechanism

Read [round-2 rejection](ISSUE30_RESULTS.md), [quality review](ISSUE30_QUALITY_REVIEW.md),
[failed replication](ISSUE30_RESEARCH_ROUND3.md) and the [v2 pilot](ISSUE45_RESULTS.md).
The historical GT/error examples show missed small vehicles, low-sharpness images
and adverse clean TP/FP changes. A night street lost a TP and gained an FP under
local gain; the small bicycle remained missed. Those counts do not identify the
lost box or establish that brightness caused a miss. No new development search
or diagnostic replay is authorized.

Quality alone did not predict detection benefit. The new test therefore compares
components on the **same original-pixel routes**, instead of ranking hundreds of
thresholds or fitting a proxy loss. Two competing mechanisms are declared:

1. In moderately dark, low-contrast images with little black clipping/noise,
   mild Gamma 0.85 might expose retained vehicle edges without local gain's
   stronger amplification. Brightness/contrast are eligibility signals, not
   proof of benefit. Gamma previously lost in aggregate; this tests a restricted
   natural-image population, not a presumed winner.
2. In moderately low-sharpness, low-noise images, mild unsharp masking (amount
   0.15, sigma 1) might strengthen retained edges. Severe blur is skipped because
   filtering cannot restore missing spatial information. Texture can inflate
   sharpness; small targets can remain capacity-limited. Denoising risks deleting
   small detail and the previous bilateral/noise-range direction failed, so it
   is not retried. CLAHE/order grids are also excluded.

The null is that neither signal predicts useful detector response. Stop rather
than add thresholds if it survives. Noise is the existing median-residual proxy,
which can collapse to zero; report this limitation rather than treat it as a
validated sensor-noise estimator.

## Fixed routes and treatments

All features use existing `quality` units (max side 512) on original BGR pixels.
The constants are mechanism-based engineering hypotheses, not estimated cutoffs:

- E: brightness <0.18, contrast <0.20, black_fraction <0.10, noise <=0.01.
- D: 50 <= sharpness <150 and noise <=0.01. The lower boundary preserves the
  historical severe-low-sharpness diagnostic; the upper bound limits enhancement
  to a moderate range. Neither threshold is a calibrated physical blur scale.
- Four mutually exclusive strata: E only, D only, both, neither.
- Exactly four arms: Skip; Gamma on E; Sharpen on D; full routed candidate.
  The candidate uses Gamma on E only, Sharpen on D only, **Gamma then Sharpen**
  on both, and Skip on neither. Components use the same original route, so the
  intersection isolates this one ordered composition. No reverse order or
  additional parameters are searched. Runtime receives pixels only, no labels,
  detector outputs, GT, source identity or scenario metadata.

## Population, compute and integrity

Use only the saved calibration-exploration partition. Exclude the union of
`docs/round2-research-v3/membership.csv.gz` (512 consumed images) and
`docs/issue45/membership.csv` (256). Continue the historical seed-2026 whole-group
order using four successive 256-image prefixes, each allowing at most 16 overshoot:
**1,024–1,088 images; four native evaluations; <=4,352 image passes**.
The metadata audit finds 2,753 available images / 1,673 leakage groups before
this selection. Save exact IDs, original GT/source hashes and group membership.
The selected metadata prefix contains **1,027 images / 649 source/leakage groups**;
sorted IDs hash to
`d925d907e4ea2da0ff1301a418d7a5a194ff0f4c114f73aba2adb0b0e8c45084`.
Do not select by labels, features or outcomes, replace a failed slice, or reuse
newly consumed groups for independent confirmation. These are unconsumed within
the documented Round-2/v2 searches; historical Gamma calibration used the broader
calibration role, so they are not a historically untouched holdout.

Reject before inference unless each primary class has >=5 original source groups.
Audit disjointness by ID, leakage group, original source group and source hash
against consumed groups and the **entire** calibration-confirmation partition.
The reserved 1,025-image confirmation plan stays unread. No development or final
pixels, image headers, source hashes or predictions are accessed. Annotations
remain metadata, and training/development IDs/GT remain unchanged.

Reuse pinned COCO-80 YOLO26n, SHA256
`9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`,
existing five-class native validator/AP, FP32 640, batch 32, seed 2026, workers 0,
NMS-free settings. Decode originals, apply treatment, encode JPEG95 once; preserve
geometry, negatives and exact paired label bytes. No synthetic scenarios are needed.
Existing/incomplete output directories block reruns. Default notebook execution
only verifies/reads committed compact evidence, without models or local data.

## Metrics, support, advancement and stopping

Primary: natural-data native five-class mAP50:95; descriptive mAP50, precision,
recall and per-class AP. Report all four route strata and historical diagnostic
subgroups: dark <0.12; bright >=0.12; low sharpness <50; images with original
motorcycle/bicycle GT area <32². Include all GT in subgroup images. Subgroup macro
AP covers represented classes only; absent classes stay absent, never zero.
No per-image AP. Illustrative TP/FP at confidence .25 / IoU .50 are counts, with
examples selected by largest TP loss then FP rise, ID tie-breaking.

Only the **complete fixed routed candidate** may advance; components are
diagnostic, not runner-ups. Require all of:

- Primary gain >=0.2 AP percentage points against Skip.
- Each E-only/D-only stratum has all five classes at >=5 original source groups
  and >=0.2-point macro gain over Skip. The intersection has the same support
  requirement and >=0.2-point gain over **each matched component**. This tests
  the signal/combination's benefit, not merely its response to image quality.
  Insufficient route support blocks advancement, without relabeling it as harm.
- No primary class regression >2 points or subgroup/class regression >1 point
  wherever that class has >=5 source groups. Unsupported rare-class effects are
  inconclusive. Skipped images must remain byte-identical across arms.
- A positive lower bound in 100 paired seed-2026 **original-source-group**
  bootstrap draws for primary gain; require all five classes in the estimand.
  Report invalid/missing-class draws, no imputation/redraw. This exploratory
  interval has no multiplicity correction and does not establish confirmation.
- Direct matched CPU preprocessing mean <=10 ms / p95 <=25 ms: one process,
  eight OpenCV threads, warm up all four arms, three repeats, rotate arm order,
  <=13,056 timed calls. Include original quality/routing diagnostics for every
  arm; exclude reads, JPEG, inference and verification equally.

If any gate fails, record a negative or insufficient-evidence decision and stop.
No retuning, extra candidates, sample expansion, runner-up confirmation or
post-pilot training override. If all pass, freeze at most one candidate and a
separate single independent group-disjoint confirmation design before reading
confirmation pixels. There is no confirmation inference authorization here.

Only a defensible independently confirmed policy warrants `configs/v3.yaml` and
02/03/04 lifecycle notebooks. Otherwise document why they do not exist. A later
justified S20 arm must preserve the frozen 9,781 training and 13,975 development
IDs/GT, independent pinned initialization, seed 2026, 20 epochs, 640, batch 16,
AdamW/freeze/augmentation settings and shared v0 Raw checkpoint selection.
Completed v0/v1/v2 policies, runs, evidence and notebooks stay immutable.
Cross-version S20 presentation remains exclusively in `compare.ipynb`; never
fabricate a v3 metric row or chain a new S20 arm through its older snapshot.
