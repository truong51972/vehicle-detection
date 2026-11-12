# BDD100K / YOLO26n versioned preprocessing experiments

Does an offline enhancement policy (brightness, contrast, local illumination, edge detail) applied
before YOLO26n help detect vehicles, especially two-wheelers, on BDD100K? Each version
pairs one offline policy with a detector **fully fine-tuned** on that policy's output.

| | Policy | Config |
|---|---|---|
| [v0](notebooks/experiments/v0/README.md) | Raw (JPEG95 only) | [v0.yaml](notebooks/experiments/v0/v0.yaml) |
| [v1](notebooks/experiments/v1/README.md) | Gamma-or-Skip | [v1.yaml](notebooks/experiments/v1/v1.yaml) |
| [v2](notebooks/experiments/v2/README.md) | Local-shadow | [v2.yaml](notebooks/experiments/v2/v2.yaml) |
| [v3](notebooks/experiments/v3/README.md) | Adaptive local contrast | [v3.yaml](notebooks/experiments/v3/v3.yaml) |
| [v4](notebooks/experiments/v4/README.md) | Canny-guided detail | [v4.yaml](notebooks/experiments/v4/v4.yaml) |

## Protocol

- **Data:** one frozen five-class split ([data.yaml](configs/data.yaml),
  [splits.csv.gz](configs/splits.csv.gz)): car, truck, bus, motorcycle, bicycle; railway
  `train` is dropped but negative images are kept. Groups never cross roles.
- **Roles per run:** Ultralytics `train` = S20 (9,781 images), `val` = calibration
  (6,985; selects `best.pt` each epoch), `test` = development (13,975; reported once
  in 04). The 9,730-image final holdout stays sealed: no pixels, hashes or inference.
- **Policy:** applied offline to original pixels, then JPEG95 once. GT, IDs and
  geometry are identical for every version; each detector is trained, selected and
  evaluated on its own policy's images.
- **Version config:** each `notebooks/experiments/<version>/<version>.yaml` owns its name,
  policy recipes, pinned checkpoint (project-relative path + SHA256), export threads
  (`preparation.workers`) and every native Ultralytics training/evaluation argument.
  Shared protocol (splits, taxonomy, data root, seed, JPEG95) stays in `configs/`.
- **Training:** v0–v3 share the canonical recipe: full fine-tune (`freeze: 0`) from the pinned COCO
  `weights/yolo26n.pt`, SGD lr0 0.01, 50 epochs, batch 32, AMP, imgsz 640,
  geometry-only augmentation (mosaic/flip/scale/translate; HSV off so photometric
  augmentation does not dilute the policy). A version may change `imgsz` (e.g. 960) if
  its processing needs it — change `training.imgsz` and `evaluation.imgsz` together.
  V4 uses batch16/workers0 on a 6 GiB GPU; comparisons require a matched
  batch16 Raw control. See [v4 results](docs/v4/RESULTS.md).
- **Evaluation:** native FP32 NMS-free validation, conf 0.001, max_det 300; AP50:95
  macro over five classes, plus dark (raw brightness < 0.12) / bright and Apply/Skip
  cohorts from saved per-image matches. Single seed; CIs from source-group bootstrap
  are conditional on the trained models.

## Run layout

Each run is dated and self-contained (original/processed images remain gitignored; selected v4 evidence is published):

```text
runs/<version>-<YYYY-MM-DD>/
  config.yaml             # snapshot of <version>.yaml at preparation
  data/                   # images|labels/{train,val,test}, data.yaml, metadata.csv
  train/                  # native args.yaml, results.csv, weights/, complete.json
  eval/                   # metrics.json, paired_stats.npz, native plots
```

Saved artifacts use project-relative paths (`data.yaml` has no `path:` key, so Ultralytics
resolves splits next to it). Only Ultralytics-native `args.yaml`/checkpoints record the
absolute runtime paths it requires. Reuse checks bytes and settings: a policy change needs a
new run date; a training-arg change after training also needs a new run date.

## Notebook workflow

Use the project `.venv` kernel. Every notebook is independent **Restart & Run All**,
inspects saved results by default and only does expensive work after an explicit switch.

Committed `ALLOW_PREPARATION`, `ALLOW_TRAINING` and `ALLOW_INFERENCE` defaults are all
`False`. Leave them disabled to inspect saved artifacts; missing artifacts are reported
without starting work. Set the relevant flag to `True` deliberately to run that phase.
Source auditing and CPU preprocessing benchmarks also default to disabled.

1. [00_data_init.ipynb](notebooks/00_data_init.ipynb): frozen membership; `AUDIT_SOURCES=True`
   audits original non-final sources.
2. `v*/01_experiments.ipynb` (v1–v4): policy preview on calibration originals; read-only.
3. `v*/02_train_prepare.ipynb`: `ALLOW_PREPARATION=True` exports today's run
   (`RUN_DATE=None`) or a given `RUN_DATE="YYYY-MM-DD"`.
4. `v*/03_train.ipynb`: `ALLOW_TRAINING=True` trains the newest run (`RESUME=True` continues
   an interrupted one).
5. `v*/04_eval.ipynb`: `ALLOW_INFERENCE=True` evaluates the selected checkpoint on development.
6. [compare.ipynb](notebooks/experiments/compare.ipynb): evidence-aware v0–v4
   report, pinned to the published run dates by default. **Restart & Run All needs
   no GPU, weights, original pixels or dataset exports.** It reads run config
   snapshots, `eval/metrics.json` and `train/complete.json`; histories/native args
   and paired NPZ statistics are optional diagnostics. The archived matched Raw
   control has `metrics.json` directly in `docs/v4/evidence/matched_raw/`.
   All 13 permitted Raw-vs-policy cells remain visible; absent, stale, incomplete
   or incompatible evidence is disclosed. No policy-to-policy inputs are allowed.

   `configs/comparison_evidence.json` pins archive artifact hashes, frozen split/GT
   attestations and runtime sources. This validates saved evidence without claiming
   to revalidate exported pixels/boxes. Historical batch-32 v0–v3 and matched
   batch-16 Raw/v4 have separate baselines, plots and cost panels. The archived
   `v5_raw-2026-11-11` alias never substitutes for historical `v0/v0`; v4's diagonal
   does not complete its 2×2. Unindexed runs are descriptive-only. Review and extend
   the provenance index explicitly for new runs/caches; never refresh hashes simply
   to admit a changed artifact.

   New semantically valid cells have status `unindexed` (pending attestation),
   with absolute descriptive metrics in coverage and their own chart. They are
   excluded from verified factorial effects, CIs and matched rankings. `stale`
   means a changed hash or invalid identity, not merely an absent index entry.
   After opt-in inference, review the live `inspect_matrix` cache validation,
   export/checkpoint/metadata/NPZ identities, training recipe, split/GT and runtime
   evidence. Add the reviewed profile and source hashes manually to
   `configs/comparison_evidence.json` in a reviewable change, then reload the report.
   The notebook never updates the evidence index automatically.

   `ALLOW_INFERENCE=True` separately runs missing cells for `INFERENCE_POLICIES`
   (one compatible profile per call), preserving full export/checkpoint/cache
   validation. It never starts training or re-export, and refuses stale/incomplete
   caches. New cross-results live in `runs/comparisons/`.
   `BOOTSTRAP_RESAMPLES=500` enables paired source-group 95% percentile CIs for
   all six contrasts of complete, matched 2×2 studies when NPZ statistics and group
   metadata exist. These intervals exclude training seed/checkpoint-selection
   uncertainty and multiple-comparison adjustment. Deltas are percentage points.
   Saved per-run cohort summaries have their own descriptive, non-paired charts
   with image support, represented classes and unrecorded GT counts labeled.
   Fixed-ID cohort comparisons still require validated routing metadata and paired
   statistics; a missing/unsupported cohort is N/A, not AP=0.
   A qualitative gallery requires compatible box predictions,
   GT and fixed development pixels, absent from the published report inputs.
   `MEASURE_PREPROCESS_COST=True` measures v0–v3 CPU routing/operators;
   `MEASURE_V4_PREPROCESS_COST=True` independently enables v4. Both need only a
   validated Raw development JPEG95 export, with no detector/profile preflight.
   Records include sampling seed/count/repeats/ID hash, proxy run, OpenCV version,
   CPU/platform and thread count plus mean/p50/p95. Native GPU timings exclude
   that policy cost; they are never summed into unmeasured end-to-end latency.

   `EXPORT_REPORT=True` writes PNGs, CSV tables and a provenance/settings/source-hash
   manifest into a **new** directory under `reports/comparison/`. Its generated
   `README.md` embeds the PNGs for GitHub viewing when published. Choose a new
   `REPORT_DIR` for each snapshot; raw experiment artifacts and existing reports
   cannot be overwritten. Export is off by default and never rewrites notebook
   outputs or the evidence index.
   See the [generated development snapshot](reports/comparison/published-v0-v4/README.md)
   for the published figures, CSVs and provenance manifest (bootstrap and CPU timing off).

```sh
uv sync --locked --extra study
uv run --extra study jupyter lab notebooks/
```

Extract BDD100K under `data/bdd100k/` (see [data/README.md](data/README.md)). The local
`weights/yolo26n.pt` must have SHA256
`9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`. Nothing is downloaded
automatically.

## Checks

```sh
uv run --extra study pytest -q
uv run --extra study ruff check .
uv run --extra study ruff format --check .
```

Tests use tiny synthetic data (including fresh-kernel notebooks and native CPU training);
they are never scientific evidence.

## Historical study

[docs/report.html](docs/report.html) is a frozen, self-contained snapshot of the earlier
frozen-backbone study (`freeze=11`, no augmentation, 20 epochs). Its preprocessing gains
stayed within ±0.15 AP points of Raw while resolution 640→960 gave about +4 points. Its
evidence files were removed in this refactor; to re-render
[report.qmd](docs/report.qmd) or replay that study, check out commit `948e1d3`.

## V4 package

See [v4 publication guide](docs/v4/README.md) for notebooks, local issue bodies,
executed artifacts, matched Raw results, checkpoints and Vietnamese reports.
