# Epic #14 cleanup decisions — issues #15, #17, #18, #19

This PR only simplifies the **existing six-class implementation**. It does not
rewrite labels, source BDD data, groups, frozen Gamma-or-Skip v2 or historical
six-class runs. The separate derived five-class labels and training belong to
#20–#22. The final holdout remains sealed for #8.

## What was actually removed

| Module | Decision | Reason |
| --- | --- | --- |
| `finetuning.py` | **REMOVE custom trainer** | Ultralytics `YOLO.train(freeze=11)` provides native freeze, augmentation controls, `results.csv`, `best.pt`, `last.pt`, optimizer, validation and resume. No additional gradient/optimizer-step/checkpoint receipt implementation is needed. |
| `baseline.py` | **REMOVE duplicate `evaluate_pretrained`** | The only active caller now uses `evaluate_development` with a small COCO-80→BDD prediction adapter. Native `YOLO.val()` owns matching, AP, native plots and native metric serialization. The old baseline still exists at its original Git revision. |
| `evaluation.completed_checkpoints` | **REMOVE custom receipt parsing** | Notebook 04 passes explicit native `best.pt` paths; a valid checkpoint must not require `complete.json`, `identity.json`, or a custom trainer run directory. Older six-class checkpoints remain evaluable directly. |
| `loader_benchmark.py` and its test | **REMOVE redundant phase-local measurement** | Offline preprocessing timing, native validation speed and `results.csv` elapsed time answer the current CV question. Separate native CPU loader throughput is not required to compare detection quality. The deleted research helper remains in Git history. |
| `evaluation.py` | **SIMPLIFY** | Call native `metrics.mean_results()` and `metrics.to_json()`; keep one explicit development role guard, hashed checkpoint/cache identity, plots and native per-image matching statistics only for paired research. No custom mAP engine, checkpoint registry, runtime callback tree or hardware/version receipt. |
| `development_report.py` | **SIMPLIFY** | Keep the full native 3×2 matrix, natural-condition AP points, per-class support, top-error examples and **two predeclared paired contrasts** (Raw-trained vs Adaptive-trained, on each matched treatment). Remove broad, repetitive per-slice bootstrap and per-contrast ledger files. |
| `evaluation_stats.py` | **KEEP** | Ultralytics does not provide the project's source-group paired bootstrap or fixed-support missing-class semantics. It calls native `ap_per_class`, never reimplements IoU/matching. This is research logic, not a replaceable YOLO wrapper. |
| `datasets.py` / `offline.py` | **SIMPLIFY preflight** | Deep decode/hash/box checks run at export or explicit inference boundaries. Cached report and training preflight recheck frozen roles, IDs, source/label metadata and immutable manifests without rehashing every JPEG. Unexpected symlinks still fail. |
| `protocol.py`, `subsets.py`, `adaptive.py`, `preprocess.py`, `data.py` | **KEEP** | Group leakage controls, frozen S10 membership, official BDD/Scalabel/Ultralytics format adaptation and pixel-only CV policy are domain-specific. Rewriting them risks changing frozen data identities for a cosmetic LOC reduction. |
| `stress.py`, `stress_eval.py`, `robustness_spec.py` | **KEEP isolated** | No canonical notebook imports the old stress flow; research tests and eventual #8 specification retain it. Removing it would sacrifice evidence and future gated work without simplifying runtime. |
| `notebooks/archive/` | **KEEP untouched** | Original historical notebook bytes and hashes are preserved; an old implementation is replayable from its recorded source revision. |

## Execution and integrity boundaries

- Each native training branch starts independently from the pinned original
  `yolo26n.pt`; `freeze=11`, shared declared seed/settings, 20 epochs and
  original augmentation policy remain unchanged. Native `args.yaml` is compared
  against every declared training setting before reuse or resume; this does not
  reintroduce a custom trainer or checkpoint receipt. Resume is native, not bitwise.
- Raw/Adaptive exports retain identical image IDs, source roles, label digests,
  source geometry and JPEG95 behavior. The original six-class `train` category
  is **not** modified here. The derived five-class taxonomy belongs to #20.
- Cheap preflights intentionally **do not** detect a JPEG whose bytes changed
  in place but whose metadata did not. Explicit deep export verification or
  fresh development inference verifies file hashes and label geometry.
- Report-only `Restart & Run All` reuses hashed artifacts. Historical
  `development-native-v1/v2` and `development-analysis-v1/v2` cache identities
  remain read-only. New cache **identity schemas** use `development-native-v3`
  and `development-analysis-v3` even when the notebook output directory is
  named `development-native-v1`; changing checkpoint, dataset, mapping, config
  or source-group plan changes the cache key.
- Native overall/per-class metrics are not comparable across different
  six/five-class denominators. Group bootstrap has no AP imputation for missing
  required classes; when fewer than two valid fixed-class draws exist, the
  comparison is `unsupported` rather than reporting an invented CI. The two
  overall contrasts are *exploratory*, not evidence of training-seed uncertainty.
  No final-test inference is triggered.

## Validation limits

Default CI checks source/notebook syntax, lint, formatting and dependency-light
fixtures. Native-CPU training, live model validation and fresh-kernel notebook
smokes require `uv run --extra study pytest -q`; if not executed, do not
describe them as passing. No full-data retraining, holdout release or new
experimental result is part of this cleanup PR.

Research: Ultralytics's `YOLO.train`, `YOLO.val`, `Metric/DetMetrics` and
`DataExportMixin.to_json` interfaces in pinned `ultralytics==8.4.198`.
