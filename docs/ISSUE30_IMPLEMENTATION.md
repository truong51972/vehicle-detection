# Issue 30 — Round 2 preprocessing experiment implementation

[Issue 30](https://github.com/truong51972/vehicle-detection/issues/30) depends on the
completed [issue 28 data preparation](ISSUE28_IMPLEMENTATION.md). The new entrypoint
is `notebooks/03_preprocessing_experiments.ipynb`. It consumes those saved artifacts
independently; it never runs an upstream notebook or training.

The implemented workflow has completed 200 exploration, 30 confirmation and 30
fixed development evaluations. **Denoise 5 led exploration but failed the unchanged
confirmation gates; no new v3 policy is released.** Frozen v2 remains unchanged.
See [the measured results and limits](ISSUE30_RESULTS.md). Issue #30 remains open
for a justified v3 selection; #33 must not consume the rejected candidate.

## Experiment contract

The notebook defaults to saved-result reuse (`ALLOW_INFERENCE=False`). A fresh
Restart & Run All validates prerequisites, freezes a plan and reports missing
measurements without creating a policy. Opt into inference in section 1 to run
missing comparisons with the pinned pretrained YOLO26n and existing native
calibration evaluator. Native matches, AP, plots and checkpoint verification reuse
existing helpers; no AP implementation, detector wrapper or dependency was added.

The execution plan targets 256 images **per calibration half**, selected by
seeded hash order of whole source groups. Group boundaries can increase the image count.
The samples stay inside notebook 02's exploration/confirmation populations. Every
candidate uses identical IDs, annotations, geometry, checkpoint, FP32 NMS-free
settings and final JPEG95 encoding. Synthetic corruptions start from original
source pixels; all treatment steps precede that one encoding.

The bounded search has 20 candidates:

- Skip and unchanged historical Gamma-or-Skip v2.
- Gamma 0.85/0.70, CLAHE 1.2/2.0, colored nonlocal-means strengths 2/5,
  unsharp amounts 0.15/0.40, plus existing bilateral and percentile operators.
- Both orders of Gamma 0.70 / CLAHE 1.2, Denoise 2 / Gamma 0.70, and
  Bilateral / Sharpen 0.15; no exhaustive product search.
- Two first-match quality rules: brightness <0.25 → Gamma 0.70; or
  high-pass noise >0.025 → Bilateral, otherwise the brightness rule. Otherwise
  Skip. Thresholds are preregistered hypotheses, not fitted using confirmation or
  corruption labels. The existing high-pass diagnostic is still only a proxy.

Bilateral filtering follows the hypothesis of edge-preserving smoothing in
[Tomasi and Manduchi](https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/MANDUCHI1/Bilateral_Filtering.html).
The existing LAB-luminance percentile stretch is inspired by
[Limare et al., Simplest Color Balance](https://www.ipol.im/pub/art/2011/llmps-scb/);
it is an adaptation, not the paper's separate RGB-channel method. These references
justify screening alternatives, not an assumption of detection improvement.

When inference is enabled, selected calibration labels are materialized into a new
run directory through the existing official/native converter and six→five
remapper. Every resulting label is checked against frozen boxes. No historical
conversion or source annotation is replaced. Negative frames remain present.

## Selection and artifacts

Use `runs/round2-preprocessing-v2/` for new conversions, JPEG exports and native
metrics/matching caches; `docs/round2-preprocessing-v2/` holds the plan, compact
measurements, plots, shortlist and confirmation decision. A policy is written
only if the locked candidate passes confirmation; this attempt wrote no policy.
The 200 exploration comparisons at the default sample target represent about
51,200 image passes before confirmation. Exports are bounded to those samples,
not the entire BDD dataset. Set a different target/output root before a larger run.

Rank by mean native mAP50:95 with equal weight for clean, underexposure, shot/read
noise and motion families, then equal weight per severity within each family.
This is an explicit research objective, not an estimate of natural deployment
prevalence. Skip wins exact ties. Missing classes or fewer than five source groups
for any class block selection; five groups is an engineering floor, not evidence
of statistical adequacy.

Freeze the winning exploration candidate and evidence hash before confirmation.
Confirmation measures only that candidate and the two controls. Its family-balanced
AP must meet both controls; each clean scenario can regress at most 0.005 AP, each
synthetic scenario at most 0.01, and each class/scenario at most 0.02 (absolute 0–1
units). Failed confirmation records rejection; it never searches runners-up on the
same confirmation data. These point-estimate tolerances are preregistered engineering
choices, not statistical significance or a best-policy guarantee.

Reports retain native mAP50/mAP50:95, class AP/support, weather/time/scene slices,
actual decision frequencies, processing mean/p95, and representative IDs losing
native IoU50 true positives at confidence ≥0.25. Cost includes quality routing and
filters, excluding I/O, corruption and JPEG. Native detector speed is recorded
separately. Unsupported slices remain labeled; the error examples do not replace AP.

A successful `policy.json` binds the candidate config, v2 recipes, plan, input/code/
numeric-library versions, exploration shortlist and confirmation evidence hashes.
`preprocessing_experiments.replay_policy(image, policy)` validates the frozen hash,
implementation and versions before replaying the recipe. New settings, code or
inputs require new output roots; existing measurements reject mismatches. This
policy remains a calibrated pretrained candidate, not proof of best fine-tuned AP.

## Validation

Tests cover ordered compositions, quality routing/Skip, deterministic complete-
group samples, manifest tampering, exact paired labels, one JPEG write per branch,
source preservation, changed-cache rejection, fixed-class selection, class
regression vetoes, frozen-policy replay and development/final-role isolation.
Fresh-kernel fixtures run from both repository and notebook directories with
upstream regeneration and unintended inference blocked. The implementation passed
130 tests, Ruff lint/format, whitespace checks and `uv build`.

The earlier four-image [integration smoke](issue30-smoke-v1/smoke.json) is preserved
separately and never counts as selection evidence. The v2 study's
[validation receipt](round2-preprocessing-v2/validation.json) verifies all 260 native
results, plots and matching files. Fixed-class AP reaggregation agrees with native
point differences to 1e-8. The last development case resumed after its notebook
parent exited; [the resume receipt](round2-preprocessing-v2/resume.json) records
unchanged protected inputs and zero guard rejections. Completed inference was not
repeated. A final cache-only fresh-kernel Run All passed all eight code cells with model
construction blocked, zero error outputs and zero source/holdout guard rejections.
Its [preservation receipt](round2-preprocessing-v2/preservation.json) binds the
protected inputs. Deep source/export verification took 63.84 minutes even with
inference disabled; use compact reports for lightweight results inspection.

The report retains negative findings, source-group uncertainty, decisions,
costs and illustrative errors. The immutable rejection and
[handoff](round2-preprocessing-v2/handoff.json) block a new v3 handoff to #33.
No model training, S20 expansion or final-test access occurred. Historical v2
already used calibration; the Round 2 halves are disjoint within this round,
not an untouched historical holdout.

## Expanded execution plan

The initial 128-image plan was checked using annotations before new detection
measurements. Exploration had only four motorcycle groups and three bicycle
groups; confirmation had four motorcycle groups. This failed the preregistered
five-group floor. The new v2 plan expands the seeded whole-group prefix to target
256 before searching AP; no balancing, oversampling or changed source role occurs.
The original v1 plan and four-image smoke evidence remain preserved.

The v2 samples contain 256 exploration images / 155 groups, 260 confirmation
images / 156 groups, and 256 development images / 146 groups. Motorcycle group
counts are 9 / 9 / 8; bicycle group counts are 12 / 14 / 17. Complete IDs, frozen
rows and supports are saved in `samples.json`, `membership.csv.gz`,
`sample_support.csv` and `support_audit.json` under the v2 report root.

The same notebook now performs the development transfer check after evaluating
the locked candidate on confirmation. It evaluates exactly the candidate and Skip/v2 controls on
the one predeclared development sample and ten scenarios, using the same native
pretrained evaluator. Transfer is descriptive and cannot change the shortlist,
thresholds, parameters or confirmation decision. Native cache identities retain
`development_validation` as the actual role; development is never disguised as
calibration. Final-test sample materialization is rejected.

OpenCV uses eight threads, recorded in the plan, after an execution-only latency
check of nonlocal means (no AP). No candidate, strength or numerical library
version changes. All 130 tests, lint/format, whitespace checks and the package
build pass after adding the fixed transfer path.
