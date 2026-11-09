# v3 — Adaptive local contrast

**Policy:** Route: contrast < 0.20, 0.04 < brightness < 0.80, noise ≤ 0.01, black fraction < 0.30 and white fraction < 0.10; otherwise Skip. Operator: CLAHE(clip 1.5, 8×8) on grayscale `Y`, offset `round(clip(0.5·(E−Y), −12, 12)·p)` with highlight/shadow protection `p = clip(min(Y−8, 247−Y)/32, 0, 1)`, added equally to BGR.

[v3.yaml](v3.yaml) holds everything this version owns: display name, policy recipes,
pinned initialization checkpoint, export threads and every native Ultralytics
training/evaluation argument. Shared protocol (splits, taxonomy, data root) stays in
[configs/data.yaml](../../../configs/data.yaml). Changing the policy needs a new run date.

**Lifecycle** (independent Restart & Run All; nothing expensive runs without its switch):

1. [01_experiments.ipynb](01_experiments.ipynb): read-only policy preview on calibration originals.
2. [02_train_prepare.ipynb](02_train_prepare.ipynb): `ALLOW_PREPARATION=True` exports `runs/v3-<YYYY-MM-DD>/data`.
3. [03_train.ipynb](03_train.ipynb): `ALLOW_TRAINING=True` fully fine-tunes YOLO26n; calibration selects `best.pt`.
4. [04_eval.ipynb](04_eval.ipynb): `ALLOW_INFERENCE=True` reports development metrics and cohorts.

Cross-version results: [compare.ipynb](../compare.ipynb).

**Prior evidence:** Historical: frozen Raw960 pilot +0.15 points (below its +0.2 gate); a post-hoc frozen-backbone S20/960 training was −0.01 points vs Raw-trained with two-wheeler regressions. This is its first full fine-tune. See the [historical report](../../../docs/report.html).
