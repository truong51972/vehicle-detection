# BDD100K Raw / Adaptive vehicle-detection study

Use these **four independent notebooks**, directly under `notebooks/`. Set the
configuration/artifact paths at the top, select the project kernel, and choose
**Restart & Run All**. Each runs only its own phase. Earlier phases supply saved
artifacts; missing or mismatched artifacts fail rather than launching upstream work.

1. [01_data_init.ipynb](notebooks/01_data_init.ipynb) audits BDD sources, classes,
   boxes, checksums and leakage; freezes training/calibration/development/final
   roles; converts labels and exports verified Raw development references.
   Writes `runs/data-v1/` and compact audits under `docs/data-v1/`.
2. [02_data_preprocessing.ipynb](notebooks/02_data_preprocessing.ipynb)
   consumes the frozen protocol and development references. Inspects the original
   calibration/operator evidence and frozen **v2 Gamma-or-Skip**, freezes nested
   S10/S20 membership, and exports paired S10 and development inputs with equal
   IDs, native labels, geometry and JPEG95 encoding. Writes
   `runs/preprocessing-v1/` and `docs/preprocessing-v1/`. S20 is a manifest only.
3. [03_training.ipynb](notebooks/03_training.ipynb) consumes prepared inputs and
   a local pinned `weights/yolo26n.pt`; runs both native YOLO26n partial fine-tuning
   branches with equal initialization/budgets and shared Raw development validation.
   Writes `runs/training-native-s10-v1/`: native `best.pt`, `last.pt`,
   `results.csv`, `args.yaml` and a small paired dataset manifest.
   `RESUME` explicitly opts in to native `last.pt` continuation.
4. [04_eval_and_report.ipynb](notebooks/04_eval_and_report.ipynb) consumes existing
   explicit native checkpoints (including historical six-class weights) and frozen
   Raw/Adaptive development inputs. Reports native
   pretrained / Raw-trained / Adaptive-trained × Raw / Adaptive metrics, per-class
   AP, latency, plots and representative errors under `runs/development-native-v1/`.
   **Default: saved-result reuse.** Set `ALLOW_INFERENCE=True` to fill missing
   evaluations. Natural condition slices are descriptive; two matched
   Raw-trained vs Adaptive-trained contrasts use native saved matches.
   `BOOTSTRAP_RESAMPLES=0` reports points by default; set 100 for
   exploratory source-group uncertainty.
   Valid caches are reused with either execution opt-in. Checkpoint/scale selection
   and final-test release remain separate. Changed inputs
   require a new output root. This notebook never imports the training runner.

No notebook invokes another notebook. You can run phase 04 while training is idle,
or edit its chart cells and Run All against valid saved results. Final-test execution
remains locked pending the separate explicit model/protocol release for evaluation.
Do not use final-test results to choose policies, checkpoints or training size.

## Prerequisites

Python 3.11+, `uv`, OpenCV system libraries (`libgl1`, `libglib2.0-0` on Ubuntu),
and compatible PyTorch/CUDA for the recorded GPU protocol:

```sh
uv sync --locked --extra study
uv run --extra study jupyter lab notebooks/
```

Use the `.venv` kernel. Repository-root and `notebooks/` launch directories both
work. Every notebook starts with a purpose, artifact contract and linked numbered
outline. The six-class taxonomy, including railway `train`, is unchanged. Extract
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
map to the six vehicle classes; fine-tuned six-class heads use native IDs/AP directly.

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
