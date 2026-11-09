# v0 — Raw baseline

**Policy:** No enhancement: original pixels are only re-encoded as JPEG95, like every other version.

[v0.yaml](v0.yaml) holds everything this version owns: display name, policy recipes,
pinned initialization checkpoint, export threads and every native Ultralytics
training/evaluation argument. Shared protocol (splits, taxonomy, data root) stays in
[configs/data.yaml](../../../configs/data.yaml). Changing the policy needs a new run date.

**Lifecycle** (independent Restart & Run All; nothing expensive runs without its switch):

1. [02_train_prepare.ipynb](02_train_prepare.ipynb): `ALLOW_PREPARATION=True` exports `runs/v0-<YYYY-MM-DD>/data`.
2. [03_train.ipynb](03_train.ipynb): `ALLOW_TRAINING=True` fully fine-tunes YOLO26n; calibration selects `best.pt`.
3. [04_eval.ipynb](04_eval.ipynb): `ALLOW_INFERENCE=True` reports development metrics and cohorts.

Cross-version results: [compare.ipynb](../compare.ipynb).

**Prior evidence:** Historical frozen-backbone S20/640 development AP50:95: 20.61%. See the [historical report](../../../docs/report.html).
