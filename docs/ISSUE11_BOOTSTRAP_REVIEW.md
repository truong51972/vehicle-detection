# Issue #11 review: fixed estimand and bounded final specification

This supersedes the variable-class v1 bootstrap intervals and shortlist gate in
[the original findings](ISSUE11_RESULTS.md). Native predictions and six-class
descriptive AP are unchanged. The [executed analysis v2 notebook](../notebooks/archive/phase2/02b_synthetic_robustness.ipynb)
verifies and reuses original saved statistics **without inference**. The complete
[original notebook](../notebooks/archive/phase2/evidence/02b_synthetic_robustness_v1.ipynb),
all files under `docs/issue11-v1/` and archived Issue #6 v1/v2 evidence remain
byte-identical. No inference identity is migrated or rewritten.

## Fixed statistical target

The original bootstrap omits the train-class group in **43/100 draws**, mixing
six- and five-class means. Those intervals do not estimate a fixed six-class target.

V2 freezes the target from original GT support before recomputing intervals:
**at least five distinct source groups per class**. This is an engineering
minimum, not a statistical adequacy guarantee. Car has 254 images / 156 groups;
truck 75 / 64; bus 40 / 35; motorcycle 10 / 10; bicycle 12 / 12. Train has **1 / 1**
and is descriptive-only. The fixed macro target equally weights the five supported
classes. Point recovery, CI, clean-safety comparison and shortlist gate all use
this same target. Six-class native metrics remain separate descriptive columns.
No source images, ground-truth boxes, predictions or split memberships are removed;
only the statistical macro target excludes the unsupported class.

Draw complete source groups with replacement, paired identically pre/post. A
draw missing any required class is invalid, excluded without AP imputation and
never replaced. The percentile interval is conditional on required-class presence.
Per-class intervals separately report valid/excluded denominators. Unsupported
train retains descriptive AP/recovery but has no CI, even when its group is drawn.

The [draw ledger](issue11-v2/bootstrap_draw_support.csv.gz) records all 26
comparisons ×100 draws ×six classes: source-image multiplicities, instances,
distinct original groups, fixed denominator, observed class count, missing required
classes and validity. This pilot has **100 valid / 0 invalid macro draws per
comparison**, **2,600 valid / 0 invalid total**. The 43 train omissions remain
visible without changing the denominator. [Per-class recovery](issue11-v2/per_class_recovery.csv)
includes explicit support, exclusions, present-draw counts and nullable CIs.

This does not fix limited pilot power, inferred grouping, a single corruption
seed or selection bias. The 100-draw nominal 95% intervals are exploratory,
without multiplicity adjustment or independent confirmation.

## Corrected evidence

The [v2 matrix](issue11-v2/experimental_matrix.csv) retains original native metrics
and explicitly names supported-class columns. Unchanged thresholds (recovery
≥0.005, CI lower bound >0, clean loss ≤0.01, restore p95 ≤25 ms) apply to the
fixed supported target throughout.

CLAHE on moderate uniform veil has:

- Six-class native Raw **0.147666 → 0.159629**, descriptive recovery **+0.011963**.
- Supported-class Raw **0.177199 → 0.191555**, recovery **+0.014356** (1.436 AP
  percentage points), conditional CI **[+0.003985, +0.029012]** (+0.398 to +2.901 points).
- Supported clean loss **-0.004608**; original measured restore p95 **8.898 ms**.

CLAHE/veil remains the **only exploratory operator shortlist**. Underexposure
CLAHE now crosses the supported point threshold but its CI still crosses zero.
Null/adverse findings remain in the matrix. No new detections, image preparation
or timings were measured. The [analysis identity](issue11-v2/analysis.json) records
original artifact hashes, policy, corrected code, seed, resamples and libraries.

## Operator screen and router confirmation are separate

**No v3 deployment.** One naturally foggy image and an operator selected on these
same 256 calibration images do not validate natural-fog improvement or a pixel-only
route. The high-pass indicator's cross-condition false positives and Gamma priority
masking remain limitations. Frozen v2 is **Gamma-or-Skip**, not a four-operator router.

[The confirmation gate](issue11-v2/confirmation_gate.json) allows only future
group-disjoint calibration confirmation. Excluding all 158 screening groups leaves
**6,729 images / 4,078 groups** eligible. None is decoded, chosen from outcomes or
evaluated here. Before confirmation, freeze the pixel-only route, parameters and
cohort. Measure clean safety, negative cases, estimator precision/recall and support,
paired fixed-target AP recovery/uncertainty and CPU p50/p95. Do not repeatedly tune
on confirmation outcomes without a new declared protocol/version. Development and
final-test data cannot tune this route. The [no-v3 decision](issue11-v2/v3_decision.json)
records the gate and unchanged v2 identity. Confirmation is a tracked follow-up;
an unsupported v3 is not required to merge this PR.

## Practical #8 matrix and cost guard

[The final specification](issue11-v2/final_robustness_spec.json) is **unexecuted**:
natural clean, plus shot/read noise, motion and veil at levels 2 and 3; only Raw
and Gamma-or-Skip; pretrained, future Raw-trained and future Gamma-or-Skip-trained
YOLO26n models. This spans distinct mechanisms with **7 conditions ×2 inputs ×3
models =42 evaluations**. All standalone screened operators are excluded;
defocus/exposure stay calibration-only. Natural and synthetic results are separate.

For 9,730 reserved IDs: **58,380 natural +350,280 synthetic =408,660 image passes**,
reduced from the old implied 1,109,220. There are 14 input datasets /136,220 files
if all retained. Prepare one condition's pair, evaluate all three models, then
release image copies while retaining identities, statistics and result caches.

Calibration extrapolations with headroom:

- Image storage: **132.03 GiB** retained, or **18.86 GiB** active-pair plus **5 GiB
  scratch**, using the largest measured mean encoded size ×1.5.
- Model stage: **1,447 s (24.1 min)**, using the largest observed batch-32
  per-image inference mean ×408,660 ×2 headroom.
- Serial corruption: **7,381 s (2.05 h)**; router **516 s (8.6 min)**, based on
  original per-image p95 costs.

These are hardware/input extrapolations, not worst-case or end-to-end guarantees.
I/O, codecs, native AP/plots, startup, peak memory and broader uncertainty cost are
excluded. Freeze exact trained checkpoint hashes and semantic mappings, complete
final pixel/geometry/content leakage checks, preserve IDs/GT and enforce this
exact matrix. Expanded scope requires a new version before release. There is one
final release after model/policy freeze; unchanged identity replay is allowed,
test-driven retuning is not.

Final uncertainty freezes its class set from final-source GT support before
model/condition comparisons, following the same rule. It does not blindly reuse
the calibration pilot's five-class list. No final-test pixels are accessed here.

## Validation

Regressions cover the one-group rare class, omission without a changing target,
descriptive-only exclusions, audited invalid required-class draws, verified
recorded-statistic loading without model construction, and bounded specification
generation without image access. Saved prediction statistics are sufficient;
no GPU reevaluation is needed for this correction.
**79 tests passed** locally, including the rare/missing-class and no-inference
provenance regressions. Ruff lint/format, source/wheel build and whitespace checks pass.
