# BDD100K / YOLO26n versioned experiments

One experiment pairs an offline preprocessing policy with the detector trained on
its output. Start with [00_data_init.ipynb](notebooks/00_data_init.ipynb), then open
an experiment README:

- [v0 — Raw baseline](notebooks/experiments/v0/README.md): Skip + JPEG95; completed
  historical S10 training and development evaluation.
- [v1 — Gamma-or-Skip](notebooks/experiments/v1/README.md): the historical algorithm
  called **Adaptive v2**; completed historical S10 training and development evaluation.
- [v2 — Later preprocessing exploration](notebooks/experiments/v2/README.md):
  measured exploration with failed gates; no replacement policy locked, no paired
  detector trained or evaluated.

Folder IDs are stable sequential experiment identities, independent of algorithm
names, maturity and performance. A changed policy/configuration that changes the
scientific comparison needs the next experiment ID. Each experiment README is its
human-readable source of truth; there is no separate experiment registry.

Each implemented lifecycle has independent **Restart & Run All** notebooks:
`01_experiments` where useful → `02_train_prepare` → `03_train` → `04_eval`.
They consume explicitly saved prerequisites and never execute upstream notebooks.
Configuration is in section 1. Launch from the repository root or any notebook's
directory. Raw has no dummy exploration phase; v2 has no fabricated downstream phases.

## Run or inspect

Install Python 3.11+, `uv`, OpenCV system libraries (`libgl1`, `libglib2.0-0` on
Ubuntu), and the study dependencies. Use the project `.venv` kernel:

```sh
uv sync --locked --extra study
uv run --extra study jupyter lab notebooks/
```

Defaults inspect historical evidence without new training/inference. Preparation
inspection requires the original verified local exports. To execute new work:

1. In notebook 00, optionally set `AUDIT_SOURCES=True` to audit original sources.
2. In each experiment's 02, set `ALLOW_PREPARATION=True`. This exports the same
   frozen 4,890 S10 and 13,975 development IDs/GT from originals using that policy
   and JPEG95. Prepare v0 before training v1 so shared Raw selection data exists.
3. In 03, set `READ_HISTORICAL=False`, `ALLOW_TRAINING=True`. Native Ultralytics
   owns optimization/checkpoints/history. Both detectors initialize independently
   from pinned COCO-80 YOLO26n and select checkpoints on shared v0 Raw development.
   Native continuation additionally requires `RESUME=True`.
4. In 04, set `READ_HISTORICAL=False`. Saved metrics are reused; new inference
   additionally requires `ALLOW_INFERENCE=True`. This evaluates each detector on
   its matching offline policy over the identical original development IDs/GT.

New outputs live under `runs/v0/` and `runs/v1/`, leaving original measurements
under their original paths. Prepared images are owned files; no dataloader filters
or custom training engine are used. An existing export/run with changed inputs
fails. Preparation publishes atomically; inspect a leftover `data.tmp` before
removing it and retrying. There is no partial-export resume engine.

## Shared data and small configs

[configs/data.yaml](configs/data.yaml) binds the single frozen
[split/group table](configs/splits.csv.gz), original dataset/GT version and seed.
It is projected from the preserved protocol and S10 membership, without resplitting.
All experiments use **car, truck, bus, motorcycle, bicycle**; railway `train` is
excluded by the existing six-to-five mapping, while negative images remain.
The final holdout is sealed: no image reads, headers, hashes, previews, label
materialization or inference. Final annotations are metadata only.

[configs/v0.yaml](configs/v0.yaml) and [configs/v1.yaml](configs/v1.yaml) hold only
policy parameters, JPEG quality, seed, initialization and training settings.
Native `data.yaml`/`train.yaml`, `args.yaml`, `results.csv`, checkpoints and one
`eval/metrics.json` replace the active multi-stage JSON receipt chain. Input/file
bindings still reject changed splits, policies, prepared bytes and foreign detectors.

Extract original BDD images/annotations under `data/bdd100k/`, or update the shared
`data_root`. The saved table has source-relative paths matching the preserved
nested local layout. A different extraction layout needs explicit path relocation
and a re-audit; do not change memberships or silently edit measured identities.
The local `weights/yolo26n.pt` must have SHA256
`9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`.
Nothing downloads datasets/checkpoints automatically. Keep data/runs on the native filesystem.

## Historical evidence and checks

[Historical replay and migration audit](docs/HISTORICAL_REPLAY.md) records the
baseline commit/tag, original notebook hashes and Git links. Archives are removed
from the active tree. Measured JSON/CSV/plots and local checkpoints stay preserved;
legacy research helpers remain for those evidence paths. The later
[#33 verified-reuse handoff](docs/ISSUE33_RESULTS.md) and its checkpoint identities
also remain unchanged; its original P4/P5 sources use the recorded upstream replay commit. Existing single-seed S10
results are exploratory; no new training, validated improvement, S20/full-data
expansion or final release is claimed by this refactor.

```sh
uv run --extra study pytest -q
uv run --extra study ruff check .
uv run --extra study ruff format --check .
uv build
```

Tests use tiny synthetic data, including fresh kernels and native CPU training;
they never count as scientific evidence or run the real study on a GPU.
