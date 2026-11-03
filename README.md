# BDD100K Raw / Adaptive vehicle-detection study

[Phase 2 parent #4](https://github.com/truong51972/vehicle-detection/issues/4)
uses **pretrained YOLO26n** and two treatments: **Raw** and **Adaptive**.
[Issue #5](https://github.com/truong51972/vehicle-detection/issues/5) establishes
four data roles and a pretrained Raw development-validation baseline before any
new fine-tuning. Historical YOLO11 experiments remain separate evidence.

## Run

Python 3.11+, `uv`, OpenCV libraries (`libgl1`, `libglib2.0-0` on Ubuntu) and a
compatible PyTorch/CUDA installation are required. Keep data and artifacts on
the WSL-native filesystem. Extract detection images and annotations under
`data/bdd100k/`; nested Kaggle layouts are supported:

```text
data/bdd100k/.../images/100k/train/**/*.jpg
data/bdd100k/.../images/100k/val/**/*.jpg
data/bdd100k/.../bdd100k_labels_images_train.json
data/bdd100k/.../bdd100k_labels_images_val.json
```

```sh
uv sync --locked --extra study
uv run --extra study jupyter lab notebooks/phase2
```

Select the project `.venv` kernel and run:

1. [01_data_protocol.ipynb](notebooks/phase2/01_data_protocol.ipynb): discover/audit
   sources, freeze training/calibration/development-validation/final-test roles,
   show distributions and drift, and export only development Raw.
2. [02_adaptive_preprocessing.ipynb](notebooks/phase2/02_adaptive_preprocessing.ipynb):
   freeze nested S10/S20 training subsets, calibrate a bounded v2 pixel-only policy on
   calibration data, export paired Raw/Adaptive S10 JPEG95 images, and inspect
   support, quality proxies, adverse examples, offline latency and storage.
3. [02b_synthetic_robustness.ipynb](notebooks/phase2/02b_synthetic_robustness.ipynb):
   issue #11 calibration-only corruption/operator screen with real frozen YOLO26n
   metrics, matched-condition recovery, estimator errors and paired uncertainty.
4. [03_finetuning_benchmark.ipynb](notebooks/phase2/03_finetuning_benchmark.ipynb):
   currently **pretrained development validation only**, including six-class
   mapping, native AP/precision/recall/plots and verified result-cache reuse.

Issue #6 is documented in [the Adaptive execution record](docs/ISSUE6_RESULTS.md).
The [v2 review follow-up](docs/ISSUE6_REVIEW.md) explains the minimum-support gate
and preserves the original v1 results independently.
The [issue #11 research record](docs/ISSUE11_RESULTS.md) adds a separately versioned
synthetic robustness study. Its experiments never replace clean BDD evaluation.
Issue #7 will extend notebook 03 with progressive partial fine-tuning.
Issue #8 releases the final test only after protocol/model/policy freeze.
See [the issue #5 execution record](docs/ISSUE5_RESULTS.md).

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
[the exposure ledger](notebooks/test_exposure.json), related groups and any groups
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
The frozen policy runs at most one of Gamma, CLAHE, Denoise or Sharpen; otherwise
it skips. Weather/time/scene labels are used only in subsequent subgroup reports.
Quality-loss weights are heuristic, and better quality scores do not prove better
detection. Adverse cases and sparse calibration support remain limitations.

`runs/issue6-v2/` holds immutable subsets/policy, conversion, paired dataset roots,
per-image decisions/costs and decoded-JPEG quality deltas. `docs/issue6-v2/` holds
compressed full-pool/subset membership, recipe, provenance and compact figures.
Changed source IDs, parent protocol, recipe or implementation require a fresh
version/root. A sampled storage estimate gates export; retain additional headroom
because this is not a worst-case bound. No source images are overwritten.

The original v1 reports/datasets remain under `docs/issue6/` and
`runs/issue6-v1/`. Its [executed notebook](notebooks/phase2/evidence/02_adaptive_preprocessing_v1.ipynb)
retains every embedded chart and example image byte-for-byte. Replay v1 with source
revision `7380eed`; the current implementation explicitly rejects v1 policies.
Do not relabel old timing or quality evidence as v2. The noise estimator remains
unvalidated and often reports zero; the new diagnostics disclose this limitation.

## Synthetic robustness research

Notebook 02b freezes a 256-image whole-group calibration pilot and a bounded
45-dataset matrix: six corruption families at three severities, independent
moderate-severity operator screens, clean safety controls and a frozen-v2 baseline.
It includes bilateral filtering and percentile luminance stretch alongside the
existing candidate operators. Nothing is trained, no roles/boxes move, and no
development-validation or final-test pixels are accessed. All exports share a
single final JPEG95 stage. Natural clean and synthetic AP remain separate.

`runs/issue11-v1/` holds the pilot, labels, datasets, per-image native matching
statistics and safe identity-checked metric caches. `docs/issue11-v1/` contains
compact configuration, membership, result/support/uncertainty tables, diagnostics,
the versioned policy decision and an unexecuted final robustness specification.
Frozen v1/v2 artifacts and helpers remain unchanged. Selected notebook image
embeds are intentional review evidence; exported image roots/weights stay ignored.

The pilot's AP/clean-safety/latency gates can shortlist operators, but independent
calibration-group router confirmation is required before deploying Adaptive v3.
Until then downstream work retains the exact v2 baseline. Synthetic corruptions
are approximations, and a better synthetic quality score is not detector recovery
or evidence of real-world camera robustness.

Notebook 02b requires the completed issue #5 protocol and pinned local
`weights/yolo26n.pt`. On a fresh checkout, run notebook 03's existing pretrained-only
baseline after notebook 01 to obtain/check that checkpoint before notebook 02b.

## WSL migration and history

The move from `/mnt/d/Projects/vehicle-detection` required repairing copied
virtual-environment launchers/editable paths and absolute raw-image symlinks.
The local repair inventory is `runs/wsl-migration.json`. On another move, prefer
recreating `.venv` with `uv sync`; audit links before opening an old notebook.
Never automatically move source data or rewrite checkpoint/provenance identities.

The original [preparation](notebooks/01_data_preparation.ipynb) and
[training](notebooks/02_training_benchmark.ipynb) notebooks are **historical YOLO11**.
Their original manifests, YAML and checkpoint arguments retain old absolute
paths; they are not the new execution entrypoints. Do not resume their optimizer
state as YOLO26. [The old execution snapshot](docs/PHASE2_RESULTS.md) records the
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
