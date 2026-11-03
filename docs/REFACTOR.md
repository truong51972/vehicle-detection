# Four independent notebook refactor

Based on merged PR #12 (`2df2241`). The prior interface had two historical YOLO11
notebooks plus four phase2 entrypoints and two evidence snapshots. The active
interface is now exactly four phase-named notebooks; all eight originals retain
identical bytes under `notebooks/archive/`. No historical result or policy changed.

## Call-site audit and final inventory

Every original module was inspected together with its imports, notebook uses and
corresponding tests. Decisions are deliberately narrower than a wholesale rewrite:

- **SIMPLIFY [common.py](../src/vehicle_detection/common.py)**: add one immutable `freeze_json` used by training and
  new operations, plus concise artifact preflight. Retain JSON/digest/file/image
  helpers and identity-checked cache loading. Callers: all active notebooks,
  `finetuning`, `evaluation`; tests: [test_evaluation_cache](../tests/test_evaluation_cache.py), [test_development](../tests/test_development.py).
- **KEEP [data.py](../src/vehicle_detection/data.py)**: native BDD/Scalabel/Ultralytics conversion, class aliases,
  coordinate roundtrips and discovery. `choose_splits` remains a legacy API used
  only by archived YOLO11 notebooks and [test_study](../tests/test_study.py); canonical roles use protocol.
- **KEEP [protocol.py](../src/vehicle_detection/protocol.py)**: four-role allocation, duplicate/group safeguards, metadata
  and GT integrity. Its freeze wrapper performs protocol validation and is not a
  duplicate of plain JSON publication. Callers: phases 01/02/03/04, subsets,
  baseline, research; tests: [test_protocol](../tests/test_protocol.py), [test_adaptive](../tests/test_adaptive.py).
- **KEEP [subsets.py](../src/vehicle_detection/subsets.py)**: scientifically meaningful nested whole-group drift search.
  Callers: phase 02, training, research; tests: [test_adaptive](../tests/test_adaptive.py). No substitute split
  algorithm or ID changes were introduced.
- **KEEP [preprocess.py](../src/vehicle_detection/preprocess.py)**: CV primitives and pixel-only routing. **KEEP [adaptive.py](../src/vehicle_detection/adaptive.py)**:
  calibration/recipe validation and training-pair binding. **KEEP [offline.py](../src/vehicle_detection/offline.py)**:
  export/encoding/alignment. Their source bytes stay unchanged because frozen policy
  and pair identities bind these implementations. Callers: phase 02, shared dataset
  validation, training; tests: [test_study](../tests/test_study.py), [test_adaptive](../tests/test_adaptive.py), [test_development](../tests/test_development.py).
- **MERGE role validation from [finetuning.py](../src/vehicle_detection/finetuning.py) into [datasets.py](../src/vehicle_detection/datasets.py)**: one implementation
  checks exact IDs, roles, owned paths, geometry and normalized GT boxes for both
  training and evaluation. Add development-pair export applying the already frozen
  policy to original sources. Callers: phase 02, training, evaluation; tests:
  [test_finetuning](../tests/test_finetuning.py), [test_development](../tests/test_development.py), fresh-kernel smoke.
- **KEEP [baseline.py](../src/vehicle_detection/baseline.py)**: the existing six-vehicle COCO adapter and original baseline
  replay path remain available. New evaluation reuses its settings/adapter rather
  than reproducing postprocessing/AP. Tests: [test_pretrained](../tests/test_pretrained.py).
- **MERGE saved-match aggregation/bootstrap from [stress_eval.py](../src/vehicle_detection/stress_eval.py) into
  [evaluation_stats.py](../src/vehicle_detection/evaluation_stats.py)**: no AP implementation was invented; aggregation still
  calls native `ap_per_class`. Share the small matching-statistics mixin. Historical
  research imports are re-exported for compatibility. Tests: [test_stress](../tests/test_stress.py),
  [test_stress_uncertainty](../tests/test_stress_uncertainty.py), fresh-kernel native development evaluation.
- **KEEP [stress.py](../src/vehicle_detection/stress.py), [robustness_spec.py](../src/vehicle_detection/robustness_spec.py) as research-specific modules**: callers
  are archived synthetic notebooks and their regression tests. They are absent from
  the canonical notebook execution imports. Moving/deleting these would break
  evidence access without simplifying the active path. `stress.freeze` remains a
  compatibility helper for original research; no new active caller uses it.
- **SIMPLIFY [finetuning.py](../src/vehicle_detection/finetuning.py)**: extract dataset checks and reuse shared immutable JSON;
  preserve the native trainer loop and safety checks. New identity version
  `paired-s10-v2` and new run root prevent reuse of historical code identities.
- **KEEP [__init__.py](../src/vehicle_detection/__init__.py)**: no eager training/model imports or new public façade.
- **ADD [evaluation.py](../src/vehicle_detection/evaluation.py)**: development-only native checkpoint validation/cache adapter
  for pretrained and six-class models, original completion-receipt inspection, no
  import of `finetuning` or trainer construction. Callers: phase 04; tests:
  [test_development](../tests/test_development.py), [test_notebook_smoke](../tests/test_notebook_smoke.py).

## Necessary native training integrations

`PairedTrainer` still subclasses native `DetectionTrainer`, without a custom loop:

- `_setup_train` verifies architecture-derived freeze boundary, six-class head,
  native loader membership, equal initialization and unchanged paired batch size.
- `optimizer_step` checks first effective gradients/updates and rejects automatic
  OOM batch changes; the parent owns optimization.
- Epoch callbacks verify frozen parameters/buffers and record measured time,
  optimizer steps and peak allocation.
- Native checkpoint callbacks atomically publish a separate resume snapshot and
  hash receipt so an interrupted write cannot destroy the previous durable epoch.
- `check_resume` binds optimizer state, architecture/settings/version and paths.
  Native EMA/FP16 epoch-boundary resume remains non-bitwise continuation.

None of these checks is replaced by a framework. Removing them would weaken explicit
paired-budget, frozen-backbone or interruption/data-loss guarantees.

## Integrity costs and cache boundaries

Preparation and training/resume retain deep image/label/hash checks as explicit
integrity boundaries. Cheap inventory checks alone cannot detect in-place image
changes; this refactor makes no unmeasured speed claim and does not remove those
checks. Redundant deep cross-pair checks remain cheap manifest parity checks where
individual exports were already verified.

Normal report replay checks immutable export metadata/inventories, labels/roles,
checkpoint/plot/statistic hashes and complete evaluation identity without decoding
all input images or constructing a model. Opting into new inference deep-verifies
inputs. New evaluations bind new implementation hashes; historical source identities
are retained verbatim in training receipts and archived evidence.

## Validation

The fixture suite executes each canonical notebook from a fresh Jupyter kernel,
using root and `notebooks/` launch paths. It runs actual native CPU training/validation
on tiny synthetic inputs, then reruns phase 04 with fatal guards on upstream phase
functions, `YOLO.train`, trainer construction and model inference. Static checks
separately validate the four active paths and archived byte preservation.
Missing/stale checkpoints/manifests, role isolation, Raw/Adaptive parity and cache
invalidation are covered alongside existing conversion/resume/metric tests.
Full-data/GPU evidence, if collected subsequently, belongs to a new development
record rather than this structural refactor. Final holdout is never routine validation.

Executed validation: **93 tests passed**, including all fresh-kernel smoke runs;
Ruff lint/format, source/wheel builds and `git diff --check` passed. No new
full-data training or final-test inference was run for this refactor.

## Updated notebook navigation and partial development follow-up

The updated #13 interface names phase 02 `02_data_preprocessing.ipynb`. All four
notebooks have numbered H1/H2 titles, a one-paragraph purpose, Inputs/Outputs/
Prerequisites, stable linked outlines, optional numbered H3 studies, and context
around plots. Training separates pinned initialization, Raw, Adaptive, convergence
and checkpoint summaries. Evaluation separates cache execution, class/condition
analysis, uncertainty, costs/errors and exports. A lightweight structural test checks
these requirements only for active notebooks; archived bytes remain unchanged.

Before: 13 Python modules. The structural extraction adds three shared operations
modules (16 total); two focused development helpers bring the current count to 18.
[development_report.py](../src/vehicle_detection/development_report.py) aggregates saved native matches with explicit point-only or
bootstrap analysis; [loader_benchmark.py](../src/vehicle_detection/loader_benchmark.py) measures native CPU loading without a
model/trainer. Callers: notebook 04; tests: [test_development_report](../tests/test_development_report.py),
[test_loader_benchmark](../tests/test_loader_benchmark.py), and fresh-kernel smoke. These are phase-local operations,
not an orchestration framework. Extracted dataset-validation code now has its own
hash in fresh training/resume identities.

Active callers: [data initialization](../notebooks/01_data_init.ipynb),
[data preprocessing](../notebooks/02_data_preprocessing.ipynb),
[training](../notebooks/03_training.ipynb), and
[evaluation/reporting](../notebooks/04_eval_and_report.ipynb).
[Fresh-kernel smoke](../tests/test_notebook_smoke.py) verifies optional missing
loader evidence does not trigger measurement or prevent cached-result reporting.
Full resampling and new loader measurements stay explicit; no reporting cell
freezes model selection automatically.

The six-class development matrix completed before the user requested stopping
further benchmarks; see [the separately attributed evidence](DEVELOPMENT_RESULTS.md).
No five-class labels, remapping or new training experiment was implemented.

Final validation after the navigation update: **97 tests passed**, including
fresh-kernel execution, optional loader absence and phase-04 cache-only guards;
Ruff lint/format, source/wheel builds and diff checks passed. Only tiny fixture
training ran during checks. Full six-class resampling/loader measurement stopped
at the user's request; no five-class migration or fresh scientific training ran.
