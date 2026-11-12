# V4 — Canny-guided vehicle detection

This branch adds the v4 policy and its complete notebook lifecycle to the current
main branch. V0–v3 notebooks, configs and existing evidence are preserved. V5
multiscale/comprehensive code is not part of this change.

## Start here

- [Policy and notebook instructions](../../notebooks/experiments/v4/README.md)
- [Actual results and matched Raw comparison](RESULTS.md)
- [Local issue documents](issues): research plan, exploration, training, evaluation.
  These are project documents, not newly created GitHub issues.
- [Validation](VALIDATION.md): original checks and current publication checks.

The notebooks under `notebooks/experiments/v4` are portable, with expensive work
disabled by default. Archived executed notebooks are under
`runs/v4-execution-2026-11-10`; use them as evidence rather than rerunning their
saved machine-specific overrides. Native run evidence is under
`runs/v4-2026-11-10`.

## Included evidence

- Native 50-epoch history, configuration, completion marker and both selected/last
  v4 checkpoints under `runs/v4-2026-11-10/train`.
- Native development metrics, matching statistics, curves, confusion matrices
  and labelled/predicted example panels under that run's `eval`.
- The matched Raw control's metrics/configuration/history in
  [evidence/matched_raw](evidence/matched_raw). Its historical local run alias was
  `v5_raw-2026-11-11`; it is a Raw policy control, not the v5 algorithm.
- Checkpoint/data/metric identities and file inventory in
  [PUBLISH_MANIFEST.json](PUBLISH_MANIFEST.json).

Original and processed BDD100K datasets, Python environments, temporary downloads,
font caches, PIDs and unrelated local previews are not included. The shared
COCO initialization remains an external prerequisite pinned by SHA256 in v4.yaml.
The complete exported BDD data must be prepared locally before a new train/eval;
the archived run is evidence, not a preinstalled dataset.

## Vietnamese reports

| Report | Word | PDF |
|---|---|---|
| Readable explanation of the project | [DOCX](../../reports/BDD100K_v0_v4_De_hieu_2026-11-12/Bao_cao_BDD100K_v0_v4_De_hieu_Tieng_Viet_2026-11-12.docx) | [PDF](../../reports/BDD100K_v0_v4_De_hieu_2026-11-12/Bao_cao_BDD100K_v0_v4_De_hieu_Tieng_Viet_2026-11-12.pdf) |
| Detailed methodology and results | [DOCX](../../reports/BDD100K_v0_v4_Methodology_2026-11-12/Bao_cao_BDD100K_v0_v4_Methodology_Tieng_Viet_2026-11-12.docx) | [PDF](../../reports/BDD100K_v0_v4_Methodology_2026-11-12/Bao_cao_BDD100K_v0_v4_Methodology_Tieng_Viet_2026-11-12.pdf) |

Each report folder includes `evidence.zip` with the original JSON/CSV evidence
receipts, hashes and report verification. Unzip beside the report to inspect
them. Machine paths inside those generated receipts record the original execution,
not portable source configuration. The reports use a fixed snapshot on
2026-11-12 at 15:39:25 UTC+7 and do not update automatically with later runs.

## Reproduce a new experiment

1. Install the project's locked dependencies and optional `study` environment.
2. Provide the official BDD100K images/annotations and pinned COCO checkpoint;
   check `configs/data.yaml` and the immutable split table.
3. Run v4 `01_experiments.ipynb` for calibration previews.
4. Choose a new `RUN_DATE`, explicitly enable `ALLOW_PREPARATION` in 02, then
   `ALLOW_TRAINING` in 03 and `ALLOW_INFERENCE` in 04.
5. For comparisons, match Raw batch/workers/runtime to v4. The extended
   `compare.ipynb` supports 13 valid Raw-vs-policy cells, but the existence of
   that interface does not mean all factorial cells have been evaluated.

The final holdout remains sealed. V4 is a negative single-seed exploratory result,
not a demonstrated improvement over matched Raw.
