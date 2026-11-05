# BDD100K Raw / Adaptive vehicle-detection study

Round 2 has independent data-preparation and preprocessing-experiment notebooks. Configure the
saved input paths in section 1, select the project kernel, then choose
**Restart & Run All**. Missing or stale prerequisites fail with an error; no
notebook launches another phase.

1. [01_data_init.ipynb](notebooks/01_data_init.ipynb) audits the existing **five-class**
   #14 artifacts and original BDD sources. It reuses frozen roles/groups, checks
   original/derived labels and geometry, deep-verifies the Raw/v2 S10 and
   development pairs **without writing their receipts**, and saves compact support,
   membership and sampled visual-quality evidence under `docs/round2-data-v2/`.
   It does not split, convert labels or export images.
2. [02_evaluation_data_preparation.ipynb](notebooks/02_evaluation_data_preparation.ipynb)
   reads the saved audit and partitions the existing calibration role into
   group-disjoint exploration/confirmation halves. It retains all development IDs
   and freezes clean plus explicitly synthetic low-light, noise and motion-blur
   recipes at severities 1–3, with fixed seeds and JPEG95 encoding. It saves
   `docs/round2-evaluation-v2/` manifests, diagnostics and visual examples. Images
   are prepared in memory from original sources; no bulk export is needed.

3. [03_preprocessing_experiments.ipynb](notebooks/03_preprocessing_experiments.ipynb)
   screens 20 bounded controls/operators/sequences/routing candidates with native
   five-class calibration metrics, then locks one candidate for disjoint
   confirmation. It defaults to saved-result reuse; `ALLOW_INFERENCE=True` runs
   missing comparisons. Plans and compact evidence use
   `docs/round2-preprocessing-v2/`; native caches use `runs/round2-preprocessing-v2/`.
   **Study measured: leading Denoise 5 candidate failed confirmation; v2 retained.** See
   [issue 30 results](docs/ISSUE30_RESULTS.md) and
   [experiment contracts and gates](docs/ISSUE30_IMPLEMENTATION.md).
   Section 6 adds an opt-in quality/threshold search, three measured operator
   refinements, diverse routing and local threshold refinement. New evidence uses
   `round2-quality-v2` roots; `round2-quality-v1` remains preserved.
   Set `RUN_QUALITY_SEARCH=True`; inference and direct runtime each have separate opt-ins.
   See [review follow-ups and replay](docs/ISSUE30_QUALITY_REVIEW.md) and the
   [first exploration record](docs/ISSUE30_QUALITY_SEARCH.md).
   [Autonomous research round 3](docs/ISSUE30_RESEARCH_ROUND3.md) adds a preregistered
   expansion to 512 exploration images, fixed competing hypotheses, early stopping
   and source-group uncertainty; existing notebook/evidence bytes stay preserved.

These use the existing five-class order: **car, truck, bus, motorcycle, bicycle**.
Original six-class annotations/conversions and frozen Gamma-or-Skip v2 results
remain unchanged. Final-test checks use metadata only: no final image opens,
hashes, headers, previews, label materialization or evaluation.

Default prerequisites reflect the preserved local study: `runs/issue5-v1/protocol.json`,
`runs/issue6-v2/subsets.json`, `docs/issue6-v2/policy.json`,
`runs/issue6-v2/conversion/S10`, `runs/issue5-v1/conversion` and
`runs/preprocessing-five-v1/`. Restore those saved inputs or configure their paths;
Round 2 never regenerates them. See [issue 28 implementation and replay](docs/ISSUE28_IMPLEMENTATION.md).

Issue #30 has a completed measured attempt and a rejected candidate; a new v3
policy selection remains unresolved. Later phases
belong to #33 (training preparation/fine-tuning) and #35 (development evaluation).
No new policy, model training or final-test release is claimed by this implementation.

The earlier workflow remains available for historical replay:

- [Original split-building notebook 01](notebooks/archive/round1/01_data_init.ipynb)
  is preserved byte-for-byte with its source revision and SHA256 in the archive
  manifest. It creates six-class `runs/data-v1/` artifacts and is not a Round 2 step.
- [02_data_preprocessing.ipynb](notebooks/02_data_preprocessing.ipynb) defaults to
  `FIVE_CLASS_ONLY=True`, deriving the earlier five-class S10/development pairs
  under `runs/preprocessing-five-v1/` from saved six-class conversions. Its
  six-class preparation path is explicitly opt-in.
- [03_training.ipynb](notebooks/03_training.ipynb) defaults to the existing
  five-class S10 exports from the earlier preprocessing notebook and writes native
  YOLO26n checkpoints, histories and paired settings under `runs/training-five-s10-v1/`.
  Both branches share initialization/budgets and Raw development validation.
  `RESUME` explicitly opts in to native `last.pt` continuation.
- [04_eval_and_report.ipynb](notebooks/04_eval_and_report.ipynb) defaults to the
  completed five-class S10 checkpoints/development exports and writes
  `runs/development-five-v1/`; historical six-class weights remain supported.
  It reports pretrained/Raw-trained/Adaptive-trained × Raw/Adaptive metrics,
  per-class AP, latency, plots and representative errors. Saved-result reuse is
  the default; inference requires `ALLOW_INFERENCE=True`. Set
  `BOOTSTRAP_RESAMPLES=100` for exploratory source-group uncertainty; the default
  is zero. Object-size image-presence slices overlap and retain all GT/predictions;
  they are not area-restricted AP. Gamma/Skip route deltas are descriptive comparisons
  of cached matches. Checkpoint selection and final-test release remain separate;
  changed inputs require a new output root. This notebook never imports the trainer.

These are earlier workflow entrypoints, not the new Round 2 phases. See the
[evaluation plan](docs/development-five-v1/PLAN.md),
[handoff checklist](docs/EPIC14_FIVE_CLASS_HANDOFF.md) and
[five-class development report](docs/development-five-v1/RESULTS.md) for the completed
six-cell matrix, support, paired group uncertainty, costs and holdout scope.

## Prerequisites

Python 3.11+, `uv`, OpenCV system libraries (`libgl1`, `libglib2.0-0` on Ubuntu),
and compatible PyTorch/CUDA for the recorded GPU protocol:

```sh
uv sync --locked --extra study
uv run --extra study jupyter lab notebooks/
```

Use the `.venv` kernel. Repository-root and `notebooks/` launch directories both
work. Every notebook starts with a purpose, artifact contract and linked numbered
outline. The historical six-class taxonomy, including railway `train`, stays unchanged;
Round 2 reuses its already derived five-class mapping. Extract
BDD detection images/annotations under `data/bdd100k/`; nested Kaggle
layouts are supported. Keep data/runs on the WSL-native filesystem.

```text
data/bdd100k/.../images/100k/train/**/*.jpg
data/bdd100k/.../images/100k/val/**/*.jpg
data/bdd100k/.../bdd100k_labels_images_train.json
data/bdd100k/.../bdd100k_labels_images_val.json
```

The local pretrained checkpoint must match SHA256
`9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`.
Notebooks fail if it is absent; they never silently download a model or the dataset.
Large images, model weights and run caches remain gitignored.

Notebook 03 defaults to the preserved local protocol at `runs/issue5-v1/protocol.json`,
subsets at `runs/issue6-v2/subsets.json`, frozen `docs/issue6-v2/policy.json`, and
five-class S10/development exports under `runs/preprocessing-five-v1/`.
It runs both branches sequentially for 20 epochs with identical settings and writes
native histories plus `summary.json`. Use a fresh `TRAIN_ROOT` for a new study;
never initialize from a historical six-class fine-tuned checkpoint.
The completed [five-class S10 report](docs/ISSUE21_RESULTS.md) includes saved
convergence plots, native histories, checkpoint hashes and the Raw interruption/
resume disclosure. Notebook 03 reuses completed runs and writes `convergence.png`;
its native CSV timing totals sum visible resume sessions, excluding partial work.

## Artifact contracts and existing experiments

A protocol binds four disjoint roles, groups/duplicate safeguards, source hashes,
boxes and classes. A policy binds calibration-only IDs and the frozen operator code.
An export binds IDs/labels/geometry, recipes and JPEG95, with a deep-verification
receipt. New training binds verified exports, initial checkpoint, native freeze and
shared settings in a short study manifest; native `args.yaml` is checked against
the declared settings on reuse. Ultralytics owns optimization, weights, history
and resume. Native resume is epoch-boundary continuation, not bitwise
reproducibility or an atomic last-checkpoint guarantee. Evaluation binds the
explicit checkpoint, verified development export, native settings and metric plots.
Raw and Adaptive always retain the same IDs and GT. COCO-80 pretrained predictions
map to the explicit five/six vehicle classes; fine-tuned heads use native IDs/AP.
Five-class mapping drops railway predictions and maps motorcycle/bicycle to 3/4.
Five-vs-six macro-mAP cannot establish an algorithmic gain because the denominator changes.

Defaults use new output roots so recorded results are preserved. To inspect the
completed S10 experiments without rerunning earlier phases, configure notebook 04:

- `PROTOCOL_PATH`: `runs/issue5-v1/protocol.json`
- `POLICY_PATH`: `runs/issue6-v2/policy.json`
- `TRAIN_ROOT`: `runs/issue7-s10-v2`
- Raw input: `runs/issue5-v1/datasets/raw`
- Adaptive input: `runs/development-v1/inputs/adaptive` for the completed six-class
  matrix, or a separately prepared export from the same original sources, labels
  and frozen policy. Phase 02 exposes `RAW_DEVELOPMENT` / `DEV_CONVERSION`.
- `OUTPUT`: `runs/development-v1` for the completed native evaluation caches.
- `PLAN`: `docs/development-v1/PLAN.md` for the historical six-class analysis.

Original training receipts remain available as historical evidence, but the new
evaluator does not require or parse them.
New runs must use distinct roots; do not resume the old six-class S10 checkpoints
with this simplified training implementation. Native `best.pt` can be evaluated
without a custom `complete.json` receipt.
No full-data training/holdout runs are part of CI. Fresh-kernel smoke fixtures have
synthetic images/checkpoints and never count as scientific evidence.

## Evidence and checks

[Archive and replay instructions](notebooks/archive/README.md) identify every
historical notebook by its original path, source revision and preserved SHA256.
[Historical design](docs/HISTORY.md), [S10 results](docs/ISSUE7_RESULTS.md) and the
[module audit](docs/REFACTOR.md) and [cleanup decisions](docs/EPIC14_CLEANUP.md) preserve study context and remaining limitations.
The [development matrix record](docs/DEVELOPMENT_RESULTS.md) preserves the six
completed evaluations and identifies unfinished #7 analysis after benchmarking stopped.
[Issue #13](https://github.com/truong51972/vehicle-detection/issues/13) restructures
the workflow; [#7](https://github.com/truong51972/vehicle-detection/issues/7)
retains the development analysis/selection deliverables. Neither implies a final-test
release or new scientific results from a refactor.

```sh
uv run --extra study pytest -q
uv run --extra study ruff check .
uv run --extra study ruff format --check .
uv build
```
