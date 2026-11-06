# v1 — Historical Gamma-or-Skip

Hypothesis: selectively brighten very dark images while preserving other images.
The original algorithm name is **Adaptive v2**; `v1` is its stable experiment folder
ID. Quality uses the preserved maximum-512-pixel-side estimator. Brightness below
0.12 selects Gamma 0.85, otherwise Skip. Other filter gates are disabled; at most
one filter is applied before JPEG95. Exact parameters live in the
[experiment config](../../../configs/v1.yaml).

Status: **locked and evaluated historical S10 experiment**. The new versioned paths
have not been trained or evaluated. The existing detector is paired with this exact
policy, not a later denoising/routing candidate.

- [01_experiments](01_experiments.ipynb): read original calibration/policy evidence.
- [02_train_prepare](02_train_prepare.ipynb): inspect preserved Adaptive exports,
  or prepare `runs/v1/data/` from originals using this policy for train and evaluation.
- [03_train](03_train.ipynb): inspect original native history, or explicitly train
  under `runs/v1/train/`. Requires saved v0 Raw development for common selection.
- [04_eval](04_eval.ipynb): read the original Adaptive→Adaptive measurement, or
  evaluate the v1 detector with matching policy; new metrics use `runs/v1/eval/metrics.json`.

[Shared data](../../00_data_init.ipynb): identical 4,890 S10/13,975 development
IDs/GT as v0, seed 2026, five classes, independently initialized from the same pinned
COCO YOLO26n checkpoint. The shared training budget/settings and Raw development
checkpoint selection match v0. The final holdout remains sealed.

Original outputs: `runs/preprocessing-five-v1/{datasets/S10,development}/adaptive/`,
`runs/training-five-s10-v1/adaptive/`, `runs/development-five-v1/adaptive-adaptive/`.
Historical Adaptive→Adaptive development mAP50:95: **19.3631%**. These are single-seed,
exploratory development results; no validated improvement is claimed. Both original
detectors selected epoch 6; Raw's later interruption limits final-epoch comparisons.

Provenance: [frozen policy](../../../docs/issue6-v2/policy.json),
[policy review](../../../docs/ISSUE6_REVIEW.md),
[training](../../../docs/ISSUE21_RESULTS.md),
[development results](../../../docs/development-five-v1/RESULTS.md),
[exact historical replay](../../../docs/HISTORICAL_REPLAY.md).

[#33 verified reuse](../../../docs/ISSUE33_RESULTS.md) later authenticated these
same original checkpoints and the #35 handoff; it did not train another detector.
