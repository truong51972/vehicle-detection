# v1 — Gamma-or-Skip

**Policy:** Gamma 0.85 when the global brightness (grayscale mean / 255 at max side 512) is below 0.12; every other image is Skip (Raw-identical). This is the historical policy called “Adaptive v2”.

[v1.yaml](v1.yaml) holds everything this version owns: display name, policy recipes,
pinned initialization checkpoint, export threads and every native Ultralytics
training/evaluation argument. Shared protocol (splits, taxonomy, data root) stays in
[configs/data.yaml](../../../configs/data.yaml). Changing the policy needs a new run date.

**Lifecycle** (independent Restart & Run All; nothing expensive runs without its switch):

1. [01_experiments.ipynb](01_experiments.ipynb): read-only policy preview on calibration originals.
2. [02_train_prepare.ipynb](02_train_prepare.ipynb): `ALLOW_PREPARATION=True` exports `runs/v1-<YYYY-MM-DD>/data`.
3. [03_train.ipynb](03_train.ipynb): `ALLOW_TRAINING=True` fully fine-tunes YOLO26n; calibration selects `best.pt`.
4. [04_eval.ipynb](04_eval.ipynb): `ALLOW_INFERENCE=True` reports development metrics and cohorts.

Cross-version results: [compare.ipynb](../compare.ipynb).

**Prior evidence:** Historical frozen-backbone S20/640: 20.69% (+0.08 points over Raw). See the [historical report](../../../docs/report.html).
