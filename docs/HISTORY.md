> Historical workflow record. Current entrypoints and replay paths: [issue 41 migration](HISTORICAL_REPLAY.md).

# Historical design and execution evidence

This narrative describes the original study. Use the four canonical notebooks in
[the current guide](../README.md) for new work; paths below identify historical artifacts.

## Data contract and holdout boundary

Official train supplies approximately **70% training pool, 10% calibration,
20% development validation**, keeping groups intact. A deterministic search over
64 seeded group permutations minimizes drift in class image presence/instances,
negative frames, box sizes, weather, lighting, scene and their intersections.
This preserves natural proportions; it does not balance classes or weather.
Counts below five are flagged for parent-marginal reporting rather than separate
subgroup inference. Allocation uses official-train evidence only.

Explicit video IDs take precedence; BDD filename prefixes are a best-effort
sequence fallback. Exact byte duplicates and equal 64-bit dHash candidates are
co-grouped within official train. These safeguards **do not prove absence of all
near duplicates**. The protocol records the limitations and source-group counts.

Final test comes from official val after excluding every previously exposed ID in
[the exposure ledger](https://github.com/truong51972/vehicle-detection/blob/51687b273e7e87b5315d84f5b67202d4cc746992/notebooks/test_exposure.json), related groups and any groups
shared with official train. Missing files and invalid boxes are disclosed.
Official-val annotations/metadata and file existence are audited; **its images
are not opened, decoded, hashed, exported, previewed or evaluated**. Its box-size
summary uses the BDD2018 1280×720 annotation canvas; pixel geometry and content
leakage checks remain pending until final release. Final-test distributions never
enter the split-selection score or Adaptive/model tuning.

The immutable local `runs/issue5-v1/protocol.json` stores IDs, relative source
paths, group keys, labels, metadata, image/annotation hashes, exclusions, seed,
algorithm and summaries. Compact membership and provenance are saved under
`docs/issue5/`. Changing the protocol requires a new version/root and an explicit
membership diff. `validate_protocol()` is the downstream integrity check.
The historical train/tune reassignment is an optional separate
`historical_comparison.json` report; legacy artifacts never enter canonical identity.

Post-allocation class × weather/time-of-day audits record image presence, instance
counts, condition denominators, rare/zero support, and both joint and conditional
drift from official train in `class_condition_support.csv` and notebook heatmaps.
Final-test metadata is descriptive only. These audits never rebalance or reassign
the frozen IDs. The review provenance record distinguishes the corrected identity
metadata from the original measured baseline; no new inference is implied.

## Conversion and baseline

Reuse the pinned BDD100K 1.0.1 / Scalabel 0.3.1 / Pydantic 1.x converter stack,
then native Ultralytics 8.4.198 COCO→YOLO conversion with 0.01-pixel box roundtrip
checks. Fixed YOLO IDs are `0 car, 1 truck, 2 bus, 3 train, 4 motorcycle, 5 bicycle`.
BDD `motor`/`bike` aliases are mapped; crowd/ignored and duplicate boxes follow
native exclusion rules. Negative images keep empty labels. Continuous extents
explicitly correct Scalabel's added endpoint pixel. Export preserves dimensions.

**Raw is no enhancement, JPEG quality 95**, establishing the common Raw/Adaptive
encoding contract for the next issue. Only development images are materialized
here; native YAML `val` points to `images/tune`. Other image directories are empty.

The untouched YOLO26n release checkpoint is downloaded under gitignored
`weights/` and pinned by SHA256. It keeps all 80 COCO output classes. A small
`DetectionValidator` adapter filters/remaps postprocessed predictions:
COCO `2,7,5,6,3,1` → BDD `0,1,2,3,4,5`; actual semantic names are checked at runtime.
Native matching/AP and plots remain unchanged. Missing-class AP stays missing.

Validation fixes imgsz 640, batch 32, CUDA 0, FP32, confidence 0.001, IoU 0.7,
max detections 300, rectangular batches, no augmentation, and **`nms=False`**
(YOLO26 one-to-one end-to-end head). Native defaults have changed over time;
this setting is explicit and must remain fixed for later comparisons.
Identities include checkpoint/data/adapter hashes, settings, versions, hardware
and plot hashes. Changed inputs require a new experiment root. Native average
exported-file stage timing is not application end-to-end or batch-1 latency.

## Nested pilots and Adaptive calibration

Notebook 02 reads the immutable issue #5 protocol without changing data roles.
S10/S20 (optional S40) are nested by image ID, retaining whole locked groups.
Selection minimizes drift across 128 seeded group orders using class presence and
instances, conditions and their intersections, class × condition support, object
sizes and negative frames. Percentages approximate image counts when needed;
rare and missing combinations are reported rather than oversampled.

The default materialization is **S10 only**, with exactly **Raw and Adaptive**.
Both use JPEG quality 95, source dimensions and byte-identical native YOLO labels.
S20 is a frozen manifest for a later scaling decision, not another automatic export.
No development or final-test images are read by notebook 02. Existing issue #5
development exports and historical YOLO11 artifacts remain independent.

Adaptive thresholds use a seeded 256-image calibration sample. V2 enumerates all
eligible cases per route and samples up to 12 with the seeded RNG, instead of
taking the first IDs. Two mild strengths are compared with skip only when at least
12 candidates are available. Smaller populations are explicitly **uncalibrated
and disabled**. The policy records eligible/candidate counts, minimum support and
filter/skip/uncalibrated status. This gate does not establish statistical adequacy.
The actual frozen v2 treatment is **Gamma-or-Skip**. CLAHE, Denoise and Sharpen
remain disabled, not validated routes. Weather/time/scene labels are used only in subgroup reports.
Quality-loss weights are heuristic, and better quality scores do not prove better
detection. Adverse cases and sparse calibration support remain limitations.

`runs/issue6-v2/` holds immutable subsets/policy, conversion, paired dataset roots,
per-image decisions/costs and decoded-JPEG quality deltas. `docs/issue6-v2/` holds
compressed full-pool/subset membership, recipe, provenance and compact figures.
Changed source IDs, parent protocol, recipe or implementation require a fresh
version/root. A sampled storage estimate gates export; retain additional headroom
because this is not a worst-case bound. No source images are overwritten.

The original v1 reports/datasets remain under `docs/issue6/` and
`runs/issue6-v1/`. Its [executed notebook](https://github.com/truong51972/vehicle-detection/blob/51687b273e7e87b5315d84f5b67202d4cc746992/notebooks/archive/phase2/evidence/02_adaptive_preprocessing_v1.ipynb)
retains every embedded chart and example image byte-for-byte. Replay v1 with source
revision `7380eed`; the current implementation explicitly rejects v1 policies.
Do not relabel old timing or quality evidence as v2. The noise estimator remains
unvalidated and often reports zero; the new diagnostics disclose this limitation.

## Synthetic robustness research

The original executed v1 study froze a 256-image whole-group calibration pilot and a bounded
45-dataset matrix: six corruption families at three severities, independent
moderate-severity operator screens, clean safety controls and a frozen-v2 baseline.
It includes bilateral filtering and percentile luminance stretch alongside the
existing candidate operators. Nothing is trained, no roles/boxes move, and no
development-validation or final-test pixels are accessed. All exports share a
single final JPEG95 stage. Natural clean and synthetic AP remain separate.

`runs/issue11-v1/` holds the pilot, labels, datasets, per-image native matching
statistics and safe identity-checked metric caches. `docs/issue11-v1/` contains
the original descriptive study, whose variable-class intervals are superseded.
Current notebook 02b verifies those original statistics under their recorded identities
and writes the corrected analysis to `runs/issue11-v2/` and `docs/issue11-v2/`.
It fixes the inferential target before resampling (five supported classes in this
pilot), records every draw's support and missing-class exclusions, and uses the same
target for recovery, CI, clean safety and shortlist gates. Six-class native AP stays descriptive.
Frozen Issue #6 artifacts/helpers remain unchanged. Selected notebook image
embeds are intentional review evidence; exported image roots/weights stay ignored.

The pilot's AP/clean-safety/latency gates can shortlist operators, but independent
calibration-group router confirmation is required before deploying Adaptive v3.
Until then downstream work retains the exact v2 **Gamma-or-Skip** baseline. Synthetic corruptions
are approximations, and a better synthetic quality score is not detector recovery
or evidence of real-world camera robustness.

The current notebook requires the original saved pilot/statistics/figures, not a GPU,
raw BDD pixels or fresh model predictions. The original complete
[v1 notebook](https://github.com/truong51972/vehicle-detection/blob/51687b273e7e87b5315d84f5b67202d4cc746992/notebooks/archive/phase2/evidence/02b_synthetic_robustness_v1.ipynb) is preserved
byte-for-byte; replay its image/model experiment with revision `98d7a1f` and its
original prerequisites. Do not rewrite old inference metadata to match new code.

The unexecuted #8 spec explicitly limits final evaluation to clean plus moderate/
severe shot/read noise, motion and veil, with Raw/Gamma-or-Skip and three model
states: 42 evaluations / 408,660 image passes. Prepare one condition pair at a time;
keep natural and synthetic reporting separate. See the versioned specification for
calibration-based disk/inference estimates and the one-time release/cache gates.

## WSL migration and history

The move from `/mnt/d/Projects/vehicle-detection` required repairing copied
virtual-environment launchers/editable paths and absolute raw-image symlinks.
The local repair inventory is `runs/wsl-migration.json`. On another move, prefer
recreating `.venv` with `uv sync`; audit links before opening an old notebook.
Never automatically move source data or rewrite checkpoint/provenance identities.

The original [preparation](https://github.com/truong51972/vehicle-detection/blob/51687b273e7e87b5315d84f5b67202d4cc746992/notebooks/archive/01_data_preparation.ipynb) and
[training](https://github.com/truong51972/vehicle-detection/blob/51687b273e7e87b5315d84f5b67202d4cc746992/notebooks/archive/02_training_benchmark.ipynb) notebooks are **historical YOLO11**.
Their original manifests, YAML and checkpoint arguments retain old absolute
paths; they are not the new execution entrypoints. Do not resume their optimizer
state as YOLO26. [The old execution snapshot](PHASE2_RESULTS.md) records the
then-observed epoch-1 result, not this new pretrained baseline.

## Checks

```sh
uv run --extra study pytest -q
uv run --extra study ruff check .
uv run --extra study ruff format --check .
uv build
git diff --check
```

Fixture tests require no dataset/model downloads. Native converter/adapter tests
run with study extras; notebook checks compile sources and inspect saved evidence
without running GPU experiments. Unit tests alone do not establish real baseline
metrics. Images, caches, runs and weights stay gitignored.

References: [BDD toolkit](https://github.com/bdd100k/bdd100k),
[native validation](https://docs.ultralytics.com/modes/val/),
[validator API](https://docs.ultralytics.com/reference/models/yolo/detect/val/),
[YOLO26 inference modes](https://docs.ultralytics.com/guides/end2end-detection/).
