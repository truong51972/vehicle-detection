# Issue #33 — independent Raw/v2 preparation and native training

Issue #30 concluded with no demonstrated useful preprocessing improvement.
P4/P5 retain **Raw and frozen Gamma-or-Skip v2**, with the existing five-class
order `car, truck, bus, motorcycle, bicycle`. A v3 policy and S20 expansion are
not prerequisites. No policy discovery, confirmation or final-test evaluation
occurs in these phases.

## P4: paired inputs and actual routes

Open [04_train_data_preparation.ipynb](../notebooks/04_train_data_preparation.ipynb),
configure section 1 and select **Restart & Run All**. Defaults read the preserved
protocol at `runs/issue5-v1/protocol.json`, subsets at `runs/issue6-v2/subsets.json`,
policy at `docs/issue6-v2/policy.json`, original `data/bdd100k/` and existing
`runs/preprocessing-five-v1/` exports. Saved six-class label prerequisites are
`runs/issue6-v2/conversion/S10` and `runs/issue5-v1/conversion`.

The phase validates the frozen S10 membership, policy/implementation, taxonomy,
class mapping, label geometry, source hashes and matching Raw/v2 label bytes.
It deep-verifies all selected exported images/labels without replacing their
historical verification receipts. It then replays Raw and the exact frozen recipe
from original S10 pixels and JPEG95-encodes in memory, checking every output hash
and actual route against the exports. The shared Raw development input receives
the same source/label/encoding audit; no detection scores guide preparation.
Calibration, independent confirmation and final-test images are not opened.

Existing export directories are audited and never repaired in place. To prepare
missing exports, select an independent new `PREPARED` and `REPORT_ROOT`; P4 reuses
the existing label remapper and exporter. Source labels/pixels, frozen membership
and policy remain unchanged. A stale or partial existing export fails rather than
being silently replaced. Incomplete new output should be preserved for inspection;
use a fresh root for another preparation attempt.

Outputs in `docs/round2-training-v1/` are immutable `preparation.json` and
`routes.csv.gz`. The receipt binds source/role/policy identities, exact export
manifests and file inventories, plus per-class training support. The route file
contains every S10 image's source group, source/output/label hashes and actual
action. Deep replay intentionally reads and encodes the complete selected data;
it is not a lightweight report viewer.

## P5: authenticated reuse or fresh native training

Open [05_model_training.ipynb](../notebooks/05_model_training.ipynb) independently
and **Restart & Run All**. Its saved prerequisite is P4's receipt. It validates
that receipt and current export manifests/inventories without running P4, scanning
BDD or exporting images.

The default `TRAIN_ROOT=runs/training-five-s10-v1` and historical evidence
`docs/issue21/s10_results.json` select verified reuse. The old pair manifest,
native args, histories and all four checkpoints must match their historical
SHA256s. The current training preflight must also prove the same protocol,
subsets, frozen policy, five-class mapping, image/label identities, pinned COCO-80
initialization, seed, freeze boundary and native settings. Checkpoints are loaded
to verify their five-class heads, native version/settings and saved metrics against
the selected best/final CSV epochs. This is evidence of equivalence to the same
preserved runs, not a new replicate or a fresh accuracy evaluation.

Both branches use existing `finetuning.prepare_pair` / `train_branch` and native
Ultralytics: 20 epochs, batch 16, imgsz 640, seed 2026, freeze=11, AdamW and disabled
augmentation. Both select checkpoints on the same Raw development population.
Training is disabled by default; absent, changed or incomplete artifacts fail.
The notebook never downloads weights or initializes from a six-class fine-tuned
checkpoint.

If reuse fails, inspect the mismatch. For a required fresh paired experiment,
select new `TRAIN_ROOT` and `REPORT_ROOT`, retain the verified P4 receipt, set
`HISTORICAL_PATH=None` and `ALLOW_TRAINING=True`. Both branches start from the
same pinned `weights/yolo26n.pt` and use the same complete budget. Set the relevant
`RESUME` entry only to continue that new root's native `last.pt`; native epoch
resume is not bitwise uninterrupted continuation. These choices never trigger
preparation or policy tuning.

P5 writes `training.json`, unmodified native CSV copies and `convergence.png` in
the report root. Large datasets, weights and executed notebook copies stay
gitignored. The handoff binds P4, actual native checkpoint paths/hashes and
training settings for #35. It does not compare deployment accuracy or authorize
final-test access.

## Costs and interpretation

The original Raw run resumed at epoch 13 after an unconfirmed interruption;
its selected best checkpoint predates that interruption. Adaptive completed
without a recorded restart. Reuse preserves this disclosure. Native CSV clock
segments include training and epoch validation, excluding setup, downtime,
discarded partial epochs and final best-checkpoint validation. Historical GPU
memory peaks were not recorded and cannot be reconstructed from these files.
The single-seed runs do not establish a reliable v2 advantage.

Default reuse never replaces original weights, histories, args, export receipts,
policies, source data or PR #37–#39 evidence. The manifests retain the existing
absolute native dataset paths; relocating exports requires preparation under a
new root rather than editing the historical manifests.
