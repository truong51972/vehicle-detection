# Issue 28 — Round 2 data preparation

Issue [#28](https://github.com/truong51972/vehicle-detection/issues/28) covers
the first two phases of [Round 2](https://github.com/truong51972/vehicle-detection/issues/27).
The implementation reuses the existing five-class #14 data and frozen v2
references. It makes no new taxonomy, allocation or policy decision.

## Entry points and saved inputs

Run `notebooks/01_data_init.ipynb`, then
`notebooks/02_evaluation_data_preparation.ipynb` using the project study kernel.
Each supports **Restart & Run All** from the repository or `notebooks/` directory.
Notebook 02 consumes notebook 01's saved audit; it never reruns the audit or an
upstream phase. Restore these existing prerequisites or configure their paths:

- `runs/issue5-v1/protocol.json` and original `data/bdd100k/` sources.
- `runs/issue6-v2/subsets.json` and `docs/issue6-v2/policy.json`.
- Historical six-class conversions: `runs/issue6-v2/conversion/S10` and
  `runs/issue5-v1/conversion`.
- Existing five-class S10 and development pairs: `runs/preprocessing-five-v1/`.

Missing or stale inputs fail before publishing a successful audit. Notebook 01
reads original annotation hashes and frozen IDs, groups, boxes and conditions;
hashes all non-final source images and checks their headers; checks six-class
label geometry, exact surviving five-class label bytes, negative images and both
Raw/v2 pairs. It reuses the shared deep verifier with receipt writing disabled.
It never repairs inputs or regenerates labels, splits, policies or images.

The former split-building notebook 01 is archived byte-for-byte under
`notebooks/archive/round1/`, with its source revision and checksum in the archive
manifest. Earlier preprocessing/training/evaluation notebooks retain their paths.

## Outputs and the issue 30 handoff

`docs/round2-data-v1/` contains the immutable `audit.json`, compressed
source/role/group membership, five-class role support and deterministic
visual-quality samples (up to 16 images per non-final role). Final support is
annotation-only. Quality summaries describe data; they do not select a policy.

`docs/round2-evaluation-v1/` contains `evaluation_sets.json`, compressed membership,
`set_support.csv`, sample diagnostics, visual examples and `summary.json`.
The manifest binds protocol/audit identities, five-class mapping, IDs, seed,
corruption code, preparation code, NumPy/OpenCV versions and JPEG95 encoding.
Change inputs, recipes, seeds or sample settings using a new version/output root.
Existing reports reject mismatched settings before replacing their evidence.

Calibration groups are ordered by SHA256 of `[2026, group]` and divided into
equal group halves. All calibration images remain present, with no group crossing
exploration and confirmation. All development IDs remain present. This subdivides
the calibration role only; the four original data roles are unchanged.

There are ten paired conditions: clean and explicitly **synthetic** underexposure,
shot/read noise and motion blur at severities 1, 2 and 3. Parameters reuse
`stress.CORRUPTIONS`; randomness reuses `stress.corrupt` with recorded IDs and seed.
Every condition starts from original source pixels and receives one JPEG95
encoding. Clean uses the same encoding. No bulk image export is required here.

For #30, validate the saved audit with `validate_audit` and verify the saved
manifest before use (the notebook adds audit and sampling provenance to the
base manifest):

```python
assert manifest["manifest_id"] == digest({k: v for k, v in manifest.items() if k != "manifest_id"})
expected = evaluation_sets(protocol, seed=manifest["seed"])
assert all(manifest[k] == v for k, v in expected.items() if k != "manifest_id")
assert manifest["data_audit_id"] == audit["audit_id"]
```

Resolve selected IDs through the saved protocol rows.
Use `scenario_image(row, DATA_ROOT, scenario, seed=manifest["seed"])` for checked
in-memory images; it verifies source hashes and geometry before returning decoded
JPEG95 pixels. Retain the existing five-class mapping and source box order for
native labels, including empty labels for negative images. Do not re-encode these
decoded arrays when comparing conditions. Materialized native datasets, if needed
by #30, must retain the same single-encoding contract.

Use exploration for candidate search, confirmation for shortlisted choices and
development for later evaluation. Historical v2 already used the calibration
role: Round 2 confirmation is disjoint from Round 2 exploration, not an untouched
historical holdout. Rare or absent class/condition support remains descriptive;
no balancing, oversampling or unsupported subgroup inference is introduced.

## Validation and limits

The local real-data run retains the original roles: 48,903 training-pool,
6,985 calibration, 13,975 development and 9,730 reserved-final images. The
calibration halves contain 3,521 exploration and 3,464 confirmation images, with
2,118 groups each. Existing S10 pairs contain 4,890 images per branch;
development pairs contain 13,975. Negative frames remain 34 S10 and 121
development images per branch.

Both notebooks were executed in fresh kernels against real saved BDD artifacts
with guards rejecting image opens/decodes under official val and writes into
historical data, runs and docs. Executed notebooks are local receipts under
`runs/issue28-validation/`; compact evidence lives in the two report directories.
The audit's final-test statement describes the execution contract; the guarded
runs and regression tests supply the access checks.

Fresh-kernel fixture tests also reject upstream regeneration, check saved-input
reuse from both launch directories, compare historical contents/mtimes, reproduce
report contents, and reject changed sampling settings without modifying evidence.
Additional tests cover deterministic/group-atomic membership, stale originals,
removed-railway coordinate changes, derived labels, exports and invalid scenarios.
Synthetic fixtures and pixel diagnostics do not count as detection evidence.
Validation passed: **120 tests**, Ruff lint/format, whitespace checks and `uv build`.

```sh
uv run --extra study pytest -q
uv run --extra study ruff check .
uv run --extra study ruff format --check .
uv build
```

No policy screening, training, inference, AP measurement or final-test evaluation
is performed. Final-test geometry still uses the historical annotation canvas;
cross-official content duplicates remain unverified because final pixels stay
sealed. Synthetic degradations are controlled approximations, not calibrated
camera physics. The reserved final evaluation remains issue #8.
