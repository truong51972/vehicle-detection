# v0 — Raw baseline

Purpose: measure the native YOLO26n baseline on original pixels, reencoded once
as JPEG95. Policy: Skip, with unchanged IDs, five-class labels and geometry.

Status: **S20 training and development evaluation completed**, alongside the historical
S10 experiment. [Issue #43](../../../docs/ISSUE43_EXECUTION.md) records the paired
S20 comparison: Raw→Raw mAP50:95 **20.6124%**, selected epoch **9**. Historical results remain attributed to
`runs/training-five-s10-v1/raw/` and `runs/development-five-v1/raw-raw/`.

- [02_train_prepare](02_train_prepare.ipynb): inspect verified historical exports
  at `runs/preprocessing-five-v1/{datasets/S10,development}/raw/`, or explicitly
  prepare S20 at `runs/v0/data/` from original sources.
- [03_train](03_train.ipynb): inspect the original CSV, or explicitly run native
  training under `runs/v0/train/` with saved preparation and pinned initialization.
- [04_eval](04_eval.ipynb): inspect the original measured pair, or evaluate the v0
  detector on v0 Raw development; new metrics use `runs/v0/eval/metrics.json`.

The shared [data contract](../../00_data_init.ipynb) and
[config](../../../configs/v0.yaml) select 9,781 S20 and 13,975 development IDs,
seed 2026, five classes and the pinned COCO YOLO26n initialization. Both detectors
select checkpoints on this same Raw development population. Budget: 20 epochs,
640 pixels, batch 16, FP32, AdamW, frozen backbone 11, disabled image augmentations.
The historical 4,890 S10 IDs remain a strict whole-group subset of S20.
Standalone evaluation uses `EVAL_SETTINGS={"workers": 0}` in both arms after a
documented multiprocess-loader stall; training remains at 8 workers.

Observed historical Raw→Raw development mAP50:95: **19.0756%**. Original training
history resumed after epoch 12; the selected best checkpoint is epoch 6.
See [training provenance](../../../docs/ISSUE21_RESULTS.md) and the
[development report](../../../docs/development-five-v1/RESULTS.md).
One seed and a development point estimate do not establish a reliable method gain.
The final holdout remains sealed. See [historical replay](../../../docs/HISTORICAL_REPLAY.md).

[#33 verified reuse](../../../docs/ISSUE33_RESULTS.md) later authenticated these
same original checkpoints and the #35 handoff; it did not train another detector.
