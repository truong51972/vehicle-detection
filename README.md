# BDD100K / YOLO26n versioned preprocessing experiments

Does an offline enhancement policy (brightness, contrast, local illumination) applied
before YOLO26n help detect vehicles, especially two-wheelers, on BDD100K? Each version
pairs one offline policy with a detector **fully fine-tuned** on that policy's output.

| | Policy | Config |
|---|---|---|
| [v0](notebooks/experiments/v0/README.md) | Raw (JPEG95 only) | [v0.yaml](notebooks/experiments/v0/v0.yaml) |
| [v1](notebooks/experiments/v1/README.md) | Gamma-or-Skip | [v1.yaml](notebooks/experiments/v1/v1.yaml) |
| [v2](notebooks/experiments/v2/README.md) | Local-shadow | [v2.yaml](notebooks/experiments/v2/v2.yaml) |
| [v3](notebooks/experiments/v3/README.md) | Adaptive local contrast | [v3.yaml](notebooks/experiments/v3/v3.yaml) |

## Protocol

- **Data:** one frozen five-class split ([data.yaml](configs/data.yaml),
  [splits.csv.gz](configs/splits.csv.gz)): car, truck, bus, motorcycle, bicycle; railway
  `train` is dropped but negative images are kept. Groups never cross roles.
- **Roles per run:** Ultralytics `train` = S20 (9,781 images), `val` = calibration
  (6,985; selects `best.pt` each epoch), `test` = development (13,975; reported once
  in 04). The 9,730-image final holdout stays sealed: no pixels, hashes or inference.
- **Policy:** applied offline to original pixels, then JPEG95 once. GT, IDs and
  geometry are identical for every version; each detector is trained, selected and
  evaluated on its own policy's images.
- **Version config:** each `notebooks/experiments/<version>/<version>.yaml` owns its name,
  policy recipes, pinned checkpoint (project-relative path + SHA256), export threads
  (`preparation.workers`) and every native Ultralytics training/evaluation argument.
  Shared protocol (splits, taxonomy, data root, seed, JPEG95) stays in `configs/`.
- **Training:** all versions currently share one recipe: full fine-tune (`freeze: 0`) from the pinned COCO
  `weights/yolo26n.pt`, SGD lr0 0.01, 50 epochs, batch 32, AMP, imgsz 640,
  geometry-only augmentation (mosaic/flip/scale/translate; HSV off so photometric
  augmentation does not dilute the policy). A version may change `imgsz` (e.g. 960) if
  its processing needs it — change `training.imgsz` and `evaluation.imgsz` together.
- **Evaluation:** native FP32 NMS-free validation, conf 0.001, max_det 300; AP50:95
  macro over five classes, plus dark (raw brightness < 0.12) / bright and Apply/Skip
  cohorts from saved per-image matches. Single seed; CIs from source-group bootstrap
  are conditional on the trained models.

## Run layout

Each run is dated and self-contained (gitignored):

```text
runs/<version>-<YYYY-MM-DD>/
  config.yaml             # snapshot of <version>.yaml at preparation
  data/                   # images|labels/{train,val,test}, data.yaml, metadata.csv
  train/                  # native args.yaml, results.csv, weights/, complete.json
  eval/                   # metrics.json, paired_stats.npz, native plots
```

Saved artifacts use project-relative paths (`data.yaml` has no `path:` key, so Ultralytics
resolves splits next to it). Only Ultralytics-native `args.yaml`/checkpoints record the
absolute runtime paths it requires. Reuse checks bytes and settings: a policy change needs a
new run date; a training-arg change after training also needs a new run date.

## Notebook workflow

Use the project `.venv` kernel. Every notebook is independent **Restart & Run All**,
inspects saved results by default and only does expensive work after an explicit switch:

1. [00_data_init.ipynb](notebooks/00_data_init.ipynb): frozen membership; `AUDIT_SOURCES=True`
   audits original non-final sources.
2. `v*/01_experiments.ipynb` (v1–v3): policy preview on calibration originals; read-only.
3. `v*/02_train_prepare.ipynb`: `ALLOW_PREPARATION=True` exports today's run
   (`RUN_DATE=None`) or a given `RUN_DATE="YYYY-MM-DD"`.
4. `v*/03_train.ipynb`: `ALLOW_TRAINING=True` trains the newest run (`RESUME=True` continues
   an interrupted one).
5. `v*/04_eval.ipynb`: `ALLOW_INFERENCE=True` evaluates the selected checkpoint on development.
6. [compare.ipynb](notebooks/experiments/compare.ipynb): newest evaluated run per version;
   `BOOTSTRAP_RESAMPLES=200` adds paired CIs against v0.

```sh
uv sync --locked --extra study
uv run --extra study jupyter lab notebooks/
```

Extract BDD100K under `data/bdd100k/` (see [data/README.md](data/README.md)). The local
`weights/yolo26n.pt` must have SHA256
`9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`. Nothing is downloaded
automatically.

## Checks

```sh
uv run --extra study pytest -q
uv run --extra study ruff check .
uv run --extra study ruff format --check .
```

Tests use tiny synthetic data (including fresh-kernel notebooks and native CPU training);
they are never scientific evidence.

## Historical study

[docs/report.html](docs/report.html) is a frozen, self-contained snapshot of the earlier
frozen-backbone study (`freeze=11`, no augmentation, 20 epochs). Its preprocessing gains
stayed within ±0.15 AP points of Raw while resolution 640→960 gave about +4 points. Its
evidence files were removed in this refactor; to re-render
[report.qmd](docs/report.qmd) or replay that study, check out commit `948e1d3`.
