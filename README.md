# BDD100K offline preprocessing + YOLO study

Two notebooks reuse the BDD100K toolkit and native Ultralytics training/evaluation.
Phase 1 is merged in [PR #2](https://github.com/truong51972/vehicle-detection/pull/2).
Phase 2 starts with **full raw YOLO11n fine-tuning**; preprocessing experiments follow
that baseline. [Issue #1](https://github.com/truong51972/vehicle-detection/issues/1)
stays open until the research requirements are complete.

## Run the notebooks

Python 3.11+, `uv`, OpenCV system libraries (`libgl1`, `libglib2.0-0` on Debian/Ubuntu)
and a compatible PyTorch/CUDA installation are required for GPU training.

```sh
uv sync --locked --extra study
uv run --extra study jupyter lab notebooks
```

Select the project `.venv` kernel and execute:

1. [01_data_preparation.ipynb](notebooks/01_data_preparation.ipynb) discovers real
   detection annotations, audits training data, creates deterministic train/tune
   partitions, calls the official converters, and verifies original-image links.
2. [02_training_benchmark.ipynb](notebooks/02_training_benchmark.ipynb) runs native
   `YOLO.train()` with per-epoch notebook curves and checkpoints, then evaluates
   the best checkpoint on **tune**, with native AP, per-class AP, confusion/PR plots,
   exported-file model timing, separate offline costs and example predictions.

Configuration cells default to `MODE="full"`, `runs/phase2-full`, and only
`raw_original`. All available annotated official-train images are used, split 80/20
by seeded video group (sample ID when no video name exists). Default training is
YOLO11n, **100 epochs maximum**, batch 32, image size 640, seed 2026, CUDA device 0,
patience 20, workers 8, checkpoint every epoch. These are configurable in notebook 2.
No fractional training or pilot limit is used in full mode.

`RESUME=True` uses native `YOLO(last.pt).train(resume=True)` after interruption.
Completed runs, including native early stopping, verify/reuse best and last hashes.
Changes to settings, versions or dataset identity require a fresh experiment root.
Resume also verifies saved checkpoint arguments, output/data paths and epoch history;
a mismatched checkpoint/CSV pair fails before any training is overwritten.
Native resume restores optimizer/scheduler state from the last saved epoch; an
interrupted partial epoch can be repeated. Avoid running two notebook writers or
trainers against the same root. A completed run is not resumed for extra epochs;
use a separately identified experiment for that decision.

For unattended execution without a web server:

```sh
uv run --extra study python - <<'PY'
from pathlib import Path
import nbformat
from nbclient import NotebookClient
root = Path.cwd()
for path in sorted((root / 'notebooks').glob('0*.ipynb')):
    notebook = nbformat.read(path, as_version=4)
    notebook.metadata.phase2_status = 'running'
    def save(**kwargs):
        nbformat.write(notebook, path)
    try:
        NotebookClient(notebook, timeout=None, kernel_name='python3',
                       resources={'metadata': {'path': str(root)}},
                       on_cell_executed=save).execute()
        notebook.metadata.phase2_status = 'complete'
    finally:
        nbformat.write(notebook, path)
PY
```

The committed notebooks record their actual execution state. Unexecuted cells have
no results; a pending/running notebook does not establish completed training.
See [Phase 2 execution evidence](docs/PHASE2_RESULTS.md) for the measured snapshot.

## Protect the final holdout

Official BDD val supplies final holdout IDs. Exclude **all** exposed IDs in
[notebooks/test_exposure.json](notebooks/test_exposure.json), including both Phase 1
pilots. Full preparation defaults to `PREPARE_HOLDOUT=False`: label discovery
reserves the remaining IDs, but their image pixels are not decoded, hashed,
exported, previewed or evaluated. EDA, image-quality samples, preflight and recipe
previews use official train only. The test directories are empty.

Notebook 2 defaults to `EVAL_SPLIT="tune"` (native YAML `val`). Best checkpoints and
all recipe/threshold choices use train/tune. Final test evaluation requires explicit
`FROZEN_FOR_TEST=True`, a `FINAL_PROTOCOL` file, and a separate fresh export root
with holdout prepared. The file-existence gate is an explicit acknowledgement;
it does not automatically prove scientific finalization. Freeze the model hash,
recipe parameters, split identities and evaluation settings before release. Do not
retune after viewing final holdout results. Final preparation must also complete
ID/video/content leakage checks against train/tune.

After the full raw baseline completes, calibrate **Gamma, CLAHE, Denoise, Sharpen
and Adaptive** on train/tune. Their existing defaults are uncalibrated heuristics.
Export them with the **JPEG quality 95-only raw control**, same selected IDs,
byte-identical labels, dimensions and encoder. Compare the same frozen raw-trained
checkpoint across input variants using identical native evaluation settings.
Original-to-JPEG95 differences measure recompression; filtering effects must be
compared with JPEG95-only control. Separate model retraining is a different axis,
not required to measure the frozen model's preprocessing effect. Notebook 2 gates
additional variants on completion of the raw baseline. Changing recipes or adding
holdout requires a fresh export identity/root; keep the completed baseline intact.

## Dataset and conversion

Extract BDD100K under `data/bdd100k/` (gitignored). Nested Kaggle layouts are supported;
there must be one canonical `images/100k` tree and one detection JSON per split:

```text
data/bdd100k/.../images/100k/train/**/*.jpg
data/bdd100k/.../images/100k/val/**/*.jpg
data/bdd100k/.../bdd100k_labels_images_train.json
data/bdd100k/.../bdd100k_labels_images_val.json
```

Images or segmentation masks alone are insufficient. The 2018 JSON list is streamed;
modern Scalabel `{frames: [...]}` / `det_train.json`, `det_val.json` is fixture-tested.
A separate real 2020 dataset has not been exercised. Missing detection labels,
ambiguous paths, duplicate IDs and invalid selected boxes fail clearly. Full mode
fails on missing train images. Frame counts can differ from nominal BDD100K counts;
all available eligible annotated data, rather than a fabricated 70k count, is used.

Conversion stack: BDD100K **1.0.1**, Scalabel **0.3.1**, Pydantic **1.x**, Ultralytics
**8.4.198**. The toolkit maps `motor`/`bike` aliases, converts BDD→Scalabel→COCO;
Ultralytics converts COCO→YOLO. Six fixed YOLO IDs are `0 car`, `1 truck`, `2 bus`,
`3 train`, `4 motorcycle`, `5 bicycle`. Nonvehicle/polygon labels are excluded;
occluded/truncated vehicles remain included. Native crowd/ignored labels are
excluded; negatives have empty labels; duplicate boxes follow native conversion.

Scalabel 0.3.1 adds one pixel to extents. The explicit `continuous` adapter subtracts
that added pixel and recomputes area for this 2018 study; `inclusive` is available
for inputs defined that way. Actual decoded dimensions replace fixed assumptions.
Every selected YOLO box is checked against its source within **0.01 pixel**. No
clipping, image resizing or coordinate-changing preprocessing occurs during export.

## Reuse, artifacts and timing limits

Artifacts stay under `runs/phase2-full/`, outside Git:

- `selection.json`: source annotation hashes, seed, selected train/tune IDs, reserved
  final test IDs, prior exposures, selection identity and box convention.
- `conversion/`, `preflight.json`, `disk_estimate.csv`: official conversion evidence,
  header/box audit and sampled disk budget.
- `datasets/raw_original/`: original source links, labels, YAML, per-image metadata,
  export identity and deep verified manifest. Original images stay unchanged.
- `metadata.csv`, `preparation_cost.csv`: offline costs and inventory.
- `experiments/yolo11n-seed2026-v1/raw_original/`: `study.json`, native arguments,
  epoch CSV, best/last/per-epoch checkpoints, plots and completion hashes.
- `experiments/yolo11n-seed2026-v1/evaluation/`: native evaluation artifacts and
  identity-checked summaries. Split is part of each pair's path and cache identity.

Preparation verifies source hashes and deeply decodes/hashes selected exports.
Training startup uses manifest fingerprints, class/split/file inventories and a
verified identity. Raw JPEGs must have valid end markers so native validation cannot silently
re-encode them. Training asserts that native loaders retain every selected image.
Treat exports as immutable. `DEEP_VERIFY_EXPORTS=True` detects
arbitrary existing-image/label edits; cheap inventory checks detect missing files.
Rerunning preparation verifies/reuses outputs and repairs generated files. Unexpected
symlinks and changed settings are rejected. Use only one exporter per directory.

Evaluation caches validate splits, datasets, model hashes, settings, software,
hardware and saved plot hashes. `FORCE_REEVALUATE=True` explicitly reruns native
validation; cached speed remains the original measurement.

Offline read/decode/quality/filter/encode/write cost is separate from native
validation speed. Preparation reports mean/median/p95 per-image costs after training.
Native average preprocessing/inference/postprocess timing on exported files is
**not** batch-1 p50/p95 latency, measured online end-to-end cost or application
throughput. Warm-up, synchronized CUDA timing and representative latency measurements
remain Phase 2 work. No handwritten AP engine, fake detector, web UI or CLI framework
is added. Rare-class AP is reported only for evaluated classes; missing classes
remain missing.

## Checks

```sh
uv run --extra study pytest -q
uv run --extra study ruff check .
uv run --extra study ruff format --check .
uv build
git diff --check
```

Tests use small fixtures, require no dataset/model download and make no detector
quality claims. With study extras, the actual official converter runs on fixtures.
The holdout isolation test fails on any attempt to decode official-val images in
development preparation. Notebook smoke checks distinguish pending/running execution
from completed evidence; they do not run full GPU experiments in CI.

References: [BDD toolkit](https://github.com/bdd100k/bdd100k),
[coordinate conventions](https://github.com/bdd100k/bdd100k/blob/master/doc/source/format.rst),
[native conversion](https://docs.ultralytics.com/reference/data/converter/),
[native training](https://docs.ultralytics.com/modes/train/),
[native validation](https://docs.ultralytics.com/modes/val/).
