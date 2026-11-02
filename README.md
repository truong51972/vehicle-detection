# BDD100K offline preprocessing + YOLO study

Two inspectable notebooks investigate whether offline preprocessing improves vehicle
detection and what it costs. This branch replaces the earlier CLI/custom evaluator;
there is no bespoke model/metric/report engine or online filtering during YOLO runs.

## Install and execute

Python 3.11+ and [uv](https://docs.astral.sh/uv/) are required. Linux/WSL is recommended
for raw-image symlinks. Minimal Linux containers need OpenCV's system libraries
(`libgl1`, `libglib2.0-0` on Debian/Ubuntu). The study extra includes PyTorch/CUDA and
Jupyter and is large; use a compatible PyTorch build for your machine.

```sh
uv sync --locked --extra study
uv run --extra study jupyter lab notebooks
```

Select this project's `.venv` Python kernel and execute in order:

1. [01_data_preparation.ipynb](notebooks/01_data_preparation.ipynb): real dataset/schema
   inspection, class/box/metadata EDA, image-quality diagnostics, official conversion,
   tune-only before/after previews, offline exports and verified output reuse.
2. [02_training_benchmark.ipynb](notebooks/02_training_benchmark.ipynb): direct native
   `YOLO.train()`/checkpoint resume and `YOLO.val()` comparisons, inline loss curves,
   confusion matrix/PR plots, native metrics, timing and diagnostic failure examples.

Both notebooks have a configuration cell near the top. The default pilot selects
64 train / 16 tune / 16 test images and materializes **raw and gamma**. Recipes for
CLAHE, denoise, sharpen and a bounded adaptive filter/skip policy are also available.
Choose enabled variants before export. `MODE='full'` uses all eligible images;
processed PNG variants can consume substantial disk space. Full-study execution is
not implied by successful pilot cells.

For unattended execution, without a web server:

```sh
uv run --extra study python - <<'PY'
from pathlib import Path
import nbformat
from nbclient import NotebookClient
root = Path.cwd()
for path in sorted((root / 'notebooks').glob('0*.ipynb')):
    notebook = nbformat.read(path, as_version=4)
    NotebookClient(notebook, timeout=1800, kernel_name='python3',
                   resources={'metadata': {'path': str(root)}}).execute()
    nbformat.write(notebook, path)
PY
```

## Dataset and conversion compatibility

Extract detection images and annotations under `data/bdd100k/` (gitignored).
Nested Kaggle directories are supported; there must be one canonical `images/100k`
tree and one official label JSON per split:

```text
data/bdd100k/.../images/100k/train/**/*.jpg
data/bdd100k/.../images/100k/val/**/*.jpg
data/bdd100k/.../bdd100k_labels_images_train.json
data/bdd100k/.../bdd100k_labels_images_val.json
```

The local **2018 list-of-frames** files are the real-data validation target. Modern
Scalabel `{frames: [...]}` / `det_train.json`, `det_val.json` containers are supported
and covered by a small fixture; a separate real 2020 dataset has not been validated.
Images, segmentation masks or image-level metadata alone are insufficient. The
1.45 GB training JSON is streamed so the toolkit receives only the selected frames.
Missing images and ignored categories are counted; ambiguous paths/duplicate IDs,
missing detection labels and invalid selected geometry fail clearly.

Pinned conversion stack: **BDD100K 1.0.1 + Scalabel 0.3.1 + Pydantic 1.x +
Ultralytics 8.4.198**. Pydantic 1.x is required by this Scalabel release. The notebook
shows installed versions, real schema examples, the installed extent conversion
function, and structural/overlay checks. It calls the official BDD category mapper
and detection converter, then Ultralytics `convert_coco(cls91to80=False)`.

The six fixed YOLO IDs are `0 car`, `1 truck`, `2 bus`, `3 train`, `4 motorcycle`,
`5 bicycle`. Official BDD mapping handles 2018 `motor`/`bike` aliases; nonvehicle
classes/polygons are excluded. Occluded/truncated vehicles remain included.
Native crowd/ignored labels are excluded; duplicate boxes follow native converter
behavior. Negatives receive empty label files.

**Documented coordinate adapter:** Scalabel 0.3.1 adds 1 to `x2-x1` / `y2-y1`.
The default 2018 study explicitly uses continuous extents, so the adapter removes
only that added pixel from official COCO width/height and updates area. `inclusive`
can be selected explicitly for data using that convention. Actual decoded image
sizes replace the toolkit's fixed-size assumption. Every selected converted YOLO
box is checked against its source within 0.01 pixel; nothing is silently clipped,
resized, or reclassified. Native category IDs are remapped by name to COCO 1–6
before the native utility creates contiguous YOLO 0–5.

## Splits, exports and reuse

Train/tune derive reproducibly from official train with video grouping where
available. Official val supplies test only. Selected ID/group/content-hash overlap
is rejected. All variants share image IDs, label bytes, dimensions and split
assignments. `data.yaml` explicitly uses `val: images/tune`, `test: images/test`.

The earlier CLI pilot's test IDs and this notebook pilot's exposed test IDs are
recorded in `notebooks/test_exposure.json`. A future full study excludes both sets.
Pilot reruns intentionally reuse the same integration subset, so it must not be
presented later as untouched research holdout. Calibrate recipes and select models
on train-derived **tune**, freeze choices, then evaluate a fresh final test set.
Never retune on test results.

Artifacts stay under `runs/notebook-pilot/` (or the configured root):

- `selection.json`, `conversion/`: selected IDs/input hashes, source subsets, official
  COCO annotations, native YOLO labels and conversion/version checks.
- `datasets/<variant>/`: raw source links or lossless processed PNGs, identical labels,
  per-variant `data.yaml`, export identity, compact metadata JSONL/CSV with parameters,
  quality scores, decisions, source/output hashes and preparation cost.
- `preparation_cost.csv`, `metadata.csv`: separate offline cost and compact inventory.
- `experiments/<study>/<variant>/`: independent native YOLO best/last checkpoints,
  arguments, results/loss curves and a small config/hash record.
- `experiments/<study>/evaluation/`: native evaluation plots for each trained/evaluated
  pair; `results.csv`, `results.json`, `per_class_AP.csv`, `validation.json` summarize it.

Rerun notebook 1 to hash-verify/skip completed exports and repair missing/corrupt
**generated** files. Changed recipes/inputs are rejected; choose a fresh output root.
Unexpected symlinks are rejected before repairs, and source images stay untouched.
Use one exporter per output directory. A partial final metadata append is discarded
and that image is recomputed. All earlier complete entries are retained.

Rerun notebook 2 with the same settings to verify/reuse completed checkpoints;
interrupted training uses **native `YOLO(last.pt).train(resume=True)`**. Changed
settings/dataset identity require a new experiment root. Checkpoint hashes prevent
silently substituting another completed run. No fake detector is used in notebooks.

## Interpretation

Notebook 2 separates frozen raw-checkpoint evaluation on raw/processed images
from separately trained raw/processed checkpoints on a common raw test set. All
metrics and speed fields come directly from `model.val()`. Per-class AP includes
only classes actually evaluated (`ap_class_index`); absent-class values remain missing.
Rare-class support, confusion matrices, PR curves and example failures matter.

Offline preparation cost is reported **separately**. Native speed on exported files
covers YOLO preprocessing/inference/postprocessing, excludes offline filters, and is
not online end-to-end latency or application throughput. Count-gap failure ranking
is a qualitative diagnostic, not IoU matching or AP.

Default adaptive thresholds are uncalibrated heuristics using pixels, never BDD
metadata. Gamma/CLAHE/denoise/sharpen preserve geometry; sharpening cannot recover
lost blur detail. This minimal workflow omits optional mixed training, synthetic
stress exports, automated threshold search, repeated-seed automation and bootstrap
intervals. A full research study still needs calibration, frozen choices, an unexposed
test set, independent seeds and uncertainty analysis.

## Checks and execution evidence

```sh
uv sync --locked                 # Small helper/CI environment; no CUDA/model downloads
uv run pytest -q                 # Native converter integration is skipped without study extras
uv run ruff check .
uv run ruff format --check .
uv sync --locked --extra study
uv run --extra study pytest -q   # Also exercises actual library conversion on synthetic fixtures
uv build
```

Helper tests cover conversion geometry/category mapping, deterministic partitioning,
recipe invariants/skip, export reuse/repair, changed inputs/settings and source safety.
Notebook smoke checks validate syntax and saved execution/visual outputs without
running real data or GPU training in CI.

**Executed locally, 2 November 2026:** both notebooks completed on real BDD2018 data,
with Python 3.11.14 / PyTorch 2.15.0+cu132 / CUDA 13.2 on an RTX 5060 Ti. The pilot
exported raw and gamma datasets with 64 train / 16 tune / 16 test images per variant,
verified every selected converted box, and separately fine-tuned both models for
3 epochs (batch 8, imgsz 640, seed 2026) from the same pretrained checkpoint.
Four native evaluation pairs completed; inline tables/curves/matrices/examples are
saved in the notebooks. Both notebooks then completed a clean fresh-kernel rerun
(66.1 s and 41.4 s); all exports were verified/reused and all four best/last checkpoint
hashes **and modification times remained unchanged**. This verifies completed-output
reuse; an intentionally interrupted native training resume was not exercised.

The current study-extra checks pass **13 tests**. A fresh core-only environment passes
**12 tests with the optional native-converter integration test skipped**. Builds and
lint/format checks pass. Full-size export/training/evaluation, real 2020 annotations,
additional materialized recipes, mixed training, calibration and multi-seed uncertainty
studies have **not** run. These new native scores use a different 16-image pilot subset
and evaluator and must not be compared numerically with the superseded CLI scores.
Data, variants, weights and full training outputs remain gitignored.

References: [BDD toolkit](https://github.com/bdd100k/bdd100k),
[BDD coordinate conventions](https://github.com/bdd100k/bdd100k/blob/master/doc/source/format.rst),
[Ultralytics converter](https://docs.ultralytics.com/reference/data/converter/),
[native training](https://docs.ultralytics.com/modes/train/),
[native validation](https://docs.ultralytics.com/modes/val/).
