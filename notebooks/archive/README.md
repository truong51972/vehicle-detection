# Archived notebook evidence

These historical notebooks are preserved **byte-for-byte**, including every
saved output and execution count. The root README distinguishes Round 2 entrypoints
from the earlier workflow retained directly under `notebooks/`.

[manifest.json](manifest.json) maps original paths to archived paths, SHA256 and the
source revision at preservation (the original eight use `2df2241`, the merged
training base). Static tests
verify every archived byte against this manifest.

- [YOLO11 preparation](01_data_preparation.ipynb) and
  [YOLO11 training](02_training_benchmark.ipynb) retain historical path assumptions
  and checkpoint identities. They are never resumed as YOLO26.
- [Data protocol](phase2/01_data_protocol.ipynb) and
  [baseline/training](phase2/03_finetuning_benchmark.ipynb) preserve the original
  executed baseline and unexecuted training cells. Full S10 used the same Python
  runner externally; training outputs were never reconstructed into notebook cells.
- [Adaptive v2](phase2/02_adaptive_preprocessing.ipynb) and
  [Adaptive v1](phase2/evidence/02_adaptive_preprocessing_v1.ipynb) retain calibration,
  ablation, routing, paired-quality and example evidence.
- [Synthetic reanalysis v2](phase2/02b_synthetic_robustness.ipynb) and
  [original synthetic screen v1](phase2/evidence/02b_synthetic_robustness_v1.ipynb)
  preserve fixed-class uncertainty, original inference and the no-v3 decision.

For execution replay, check out the recorded source revision in a separate clone,
restore the original notebook paths there, install its locked environment and supply
its original manifests, BDD sources, checkpoints and local run artifacts. Replay
Adaptive v1 with `7380eed`, original synthetic inference with `98d7a1f`, and the
merged S10 workflow with `2df2241` (training implementation `a853add`). Existing
absolute paths require explicit relocation verification, never silent identity edits.

Some historical provenance JSON names original notebook paths. Those records remain
unchanged; use the manifest to locate the preserved bytes. Archive notebooks may
contain stale prose links/path assumptions by design. Current navigation and phase
instructions live in the root README. No old report, calibration or inference identity
was rewritten to claim a replay under new source code.

[Round 1 notebook 01](round1/01_data_init.ipynb) preserves the split-building and
six-class label-conversion entrypoint before issue #28 changed active notebook 01
to a read-only five-class audit. Its recorded revision is `99d9af2`. Fresh-kernel
legacy smoke tests still use this archived entrypoint to create synthetic fixture
prerequisites; Round 2 notebooks consume saved prerequisites only.

[Round 2 preprocessing screening](round2-screening/03_preprocessing_experiments.ipynb)
preserves the exact PR #37 notebook before the quality-derived exploration extension.
The old `docs/round2-preprocessing-v2/evidence_manifest.json` still names its original
active path; resolve that entry through this archive manifest when checking its hash.
