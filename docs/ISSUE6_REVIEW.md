> Historical workflow record. Current entrypoints and replay paths: [issue 41 migration](HISTORICAL_REPLAY.md).

# Issue #6 PR review follow-up: verification and v2 support gate

This addresses the review on [PR #10](https://github.com/truong51972/vehicle-detection/pull/10).
The original results in [ISSUE6_RESULTS.md](ISSUE6_RESULTS.md) remain v1 evidence
from revision `7380eed`. The original executed notebook is copied byte-for-byte
to [the v1 evidence archive](https://github.com/truong51972/vehicle-detection/blob/51687b273e7e87b5315d84f5b67202d4cc746992/notebooks/archive/phase2/evidence/02_adaptive_preprocessing_v1.ipynb),
including its embedded charts and selected example images. Every original file
under `docs/issue6/`, and the local v1 subset/policy manifests, remains unchanged.
The [preservation inventory](issue6-v2/v1_preservation.json) records their hashes.

The current [notebook 02](https://github.com/truong51972/vehicle-detection/blob/51687b273e7e87b5315d84f5b67202d4cc746992/notebooks/archive/phase2/02_adaptive_preprocessing.ipynb)
uses fresh `runs/issue6-v2/` and `docs/issue6-v2/` roots. Current code explicitly
rejects v1 policy reuse: replay v1 using its recorded revision, rather than silently
relabeling old artifacts. Subset IDs and the parent protocol are unchanged.

## Verification always runs

`export_pair()` now calls deep Raw/Adaptive verification unconditionally and
raises `ValueError` if its image count differs from the selected subset. It no
longer puts the verifier inside `assert`, whose entire expression disappears
under `python -O`. The verifier still hashes/decodes both exports, checks exact
pairing and writes both `verified.json` manifests.

The regression launches an actual optimized Python subprocess with a small paired
export and checks that both verification manifests exist. A second subprocess
forces an incorrect verifier count and must fail with the explicit count error.
The test would fail against the original assertion-based implementation.

## Calibration support is explicit

`calibrate_policy()` now emits **issue6-policy-v2**. For each route it enumerates
all eligible images in the seeded calibration sample, then uses seeded uniform
sampling across that full list rather than taking the first IDs. This removes
the ID-prefix selection bias; it does not claim optimal stratification.

The default gate is **12 selected candidate images**, configurable with
`min_support`. It is an engineering minimum, not a statistical adequacy guarantee.
Routes below it do not evaluate candidate strengths and remain disabled with
status `uncalibrated_insufficient_support`. Supported routes record either
`proxy_calibrated_filter` or `proxy_calibrated_skip`. Validation rejects inconsistent
support/status records and enabled uncalibrated routes.

The real 256-image calibration sample produces:

- **Gamma:** 65 eligible, 12 seeded candidates; `proxy_calibrated_filter`, Gamma 0.85.
- **CLAHE:** 5 eligible/candidates; uncalibrated and disabled, rather than fitting
  a strength from five images as v1 did.
- **Denoise / Sharpen:** 0 eligible/candidates; uncalibrated and disabled.

The [v2 policy](issue6-v2/policy.json) records every candidate ID, counts, minimum,
status, candidate outcomes and source/code identities. Its identity is
`1ac3fb9bb94b132f70d3e168b5ef15a54b616444e535b9a3bb265682a6fce0ef`.
No development-validation or final-test data enters this fitting step.

## Noise estimator and performance limits

The median-residual noise proxy remains **unvalidated**: 229 of 256 sampled
calibration images have zero noise, with maximum **0.0116281**, below the initial
**0.035** routing floor. V2 explicitly records those diagnostics and does not claim
that Denoise is calibrated. Lowering the floor merely to activate a route would
not validate the estimator. A replacement needs separate noise-reference/synthetic
validation and another versioned calibration decision; it is deferred here.

The optional redundant-quality-extraction optimization is also deferred.
`preprocess.py` and its metric/operator semantics remain unchanged; the new run
does not claim an implementation speedup. V1 timing is not reused as v2 timing.
Image-quality proxy changes still do not establish detection benefit.

## Independent v2 execution

Notebook 02 completed on **2026-11-03** with all code cells executed and no errors.
Both S10 branches contain **4,890** deeply verified images with matching labels,
dimensions and IDs. The complete nested-subset manifest is identical to v1, and
the Raw branch retains its identical content identity. Calibration reproduced the
exact v2 policy, eligible counts and seeded candidate IDs on an independent rerun.

V2 applies Gamma to **1,107** images and skips **3,783 (77.36%)**. It no longer
applies CLAHE to the 93 S10 cases processed by v1. All 1,107 transformed images
improve the aggregate proxy loss; this is not a detector result. Images occupy
**0.568 GiB Raw + 0.580 GiB Adaptive**, excluding labels/conversion/reports.

Measured CPU extraction/filter cost is **p50 2.915 ms / p95 5.980 ms** for Adaptive
and **2.780 / 3.224 ms** for Raw metric extraction. Full per-image preparation is
**7.974 / 10.304 ms** for Adaptive and **7.432 / 8.825 ms** for Raw. These intervals
exclude whole-workflow conversion and verification and are not deployment latency.
They are independently measured v2 costs, with no optimization claim.

## Checks

**43 tests pass**, including optimized-Python verification/count failure,
all-eligible seeded sampling, minimum-support rejection, calibrated-filter and
calibrated-skip cases, noise collapse diagnostics, invalid support/status and
v1-policy reuse rejection, plus the existing converter and dataset tests.
Ruff lint/format, source/wheel builds and whitespace checks pass.
The [v2 execution summary](issue6-v2/summary.json) records the
independent paired export, quality, timing and storage evidence.
