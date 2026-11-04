# Epic #14 — Five-class handoff for Codex (WIP)

Branch: `feat/epic14-five-class-prep` (do **not** merge until validated).

## Already prepared

- `src/vehicle_detection/five_class.py` provides explicit six→five YOLO-ID remapping,
  dropping **railway train only** (`3→DROP`) and preserving all surviving numeric
  coordinates and image IDs. Negative-only frames remain present.
- Existing exporter and verifier functions derive class names and mapping
  from `conversion.json` / `export.json`, without adding taxonomy parameters
  to the call chain. Historical six-class behavior remains the default.
- Notebook 02 has a **new section 8** deriving a separate S10 training and
  development pair under `runs/preprocessing-five-v1/`. It reads previous
  six-class conversion/metadata and does not materialize final-test images.
- Its default `FIVE_CLASS_ONLY=True` skips all historical section 2–7 work.
  Section 8 reads verified frozen subsets/conversions, checks the original
  annotation hashes, and uses the existing exporters and deep role/geometry checks.
  Only fresh six-class preparation should set `FIVE_CLASS_ONLY=False`.
- `tests/test_five_class.py` exercises mapping, preserved source files,
  all-negative frames and paired native export verification. It also checks all
  six input IDs, unchanged whitespace/CRLF bytes and rejection of broken output
  symlinks. The fresh-kernel notebook smoke reruns five-class-only preparation
  with upstream work blocked, historical byte/mtime checks and a reserved-pixel guard.
- `support_audit` summarizes class presence/instances, object sizes, new negative
  frames and natural conditions for all four frozen roles **from annotations only**.
  It does not open final-test pixels.

## Next implementation tasks (keep small)

1. **#20:** The five-class-only Run All path and real BDD100K verification below
   are complete. Review the evidence before closing the issue. S20 remains
   membership/annotation-only; final-test images stay sealed.
2. **#21:** Adapt the *existing* `prepare_pair`/training notebook to consume
   `runs/preprocessing-five-v1/datasets/S10` and
   `runs/preprocessing-five-v1/development/raw`; preserve native YOLO
   `freeze=11`, equal optimizer/seed/settings, independent pretrained
   initialization, explicit versioned output paths and `last.pt` resume.
   Add a tiny CPU smoke fixture; **do not fabricate full-data training**.
3. **#22:** Extend the existing COCO→BDD baseline adapter for the five-class map
   `{2:0, 7:1, 5:2, 3:3, 1:4}`. Keep the historical six-class adapter intact.
   Make only the changes needed in the **existing** evaluator and notebook 04
   to read five-class native labels, preserve the six-class historical adapter,
   and report the existing 3×2 paired analysis. Use native YOLO AP/matching.
   Do not implement custom metrics, a generic taxonomy registry, model wrapper,
   or another cache/experiment framework. Keep bootstrap exploratory with
   missing-class AP unsupported. No final-test pixels; opt-in inference only.

## Run and verification

```sh
uv sync --locked --extra study --python 3.11
uv run --extra study pytest -q tests/test_five_class.py
uv run --extra study pytest -q
uv run --extra study ruff check .
uv run --extra study ruff format --check .
```

Run notebook 02 using existing BDD100K and frozen six-class artifacts. For the
preserved local study, configure section 1 with:

```python
FIVE_CLASS_ONLY = True
PROTOCOL_PATH = ROOT / "runs/issue5-v1/protocol.json"
FROZEN_SUBSETS = ROOT / "runs/issue6-v2/subsets.json"
TRAIN_CONVERSION = ROOT / "runs/issue6-v2/conversion/S10"
DEV_CONVERSION = ROOT / "runs/issue5-v1/conversion"
FROZEN_POLICY = ROOT / "docs/issue6-v2/policy.json"
```

Check `runs/preprocessing-five-v1/summary.json`,
negative-frame counts, matched IDs/labels and `verified.json` under both
branches. Inspect actual artifacts before authorizing full training.

## Local verification — 2026-11-04

- Full study suite: **94 passed**, including fresh-kernel six-class preparation
  and five-class-only reuse. Ruff lint/format, whitespace checks and `uv build` pass.
- Real notebook-02 Run All from the frozen paths above produced **4,890 S10** and
  **13,975 development** images per branch. All role IDs are exact; surviving
  coordinate bytes and Raw/Adaptive label bytes match for every selected image.
  Deep role checks validate mapped class IDs, source hashes and box geometry.
- Negative frames remain present: **34 S10**, **121 development** per branch.
  Development includes one newly negative frame after dropping railway boxes.
- Frozen roles remain **48,903 / 6,985 / 13,975 / 9,730** for training pool,
  calibration, development and reserved test. Railway annotation support matches
  **S10 5 / S20 13 / pool 99 / development 21 / reserved 15**.
- **66,441 historical/source files** retain size/mtime; historical non-image
  contents and original annotation JSONs retain SHA256. Validation guards reject
  writes into historical roots and any reserved-image open/decode. No guard fired;
  all unselected export role directories remain empty.
- New image storage totals **4.411 GiB**. S20 and final-test audits are annotation-only.
  Local receipts: `runs/preprocessing-five-v1/summary.json`, `validation.json`,
  `source-preservation-before.json` and `02_data_preprocessing.executed.ipynb`.
- The validation resumed after replacing an image-write mock that retained image
  arrays with a plain guard. Export timing fields are instrumented/resumed values,
  not performance evidence. No new model training, AP results or final-test release.

## Guardrails

- Do not overwrite original annotations, six-class YOLO labels, archived
  notebooks, checkpoint weights, `runs/preprocessing-v1` or `runs/data-v1`.
- Do not change or recalibrate Gamma-or-Skip v2, resplit datasets, or open final-test
  pixels. S20 remains membership-only until a separate decision.
- The five-class and six-class macro-mAP denominators differ: never claim accuracy
  improved simply because a class was removed.
- Keep #14 open until #20–#22 are genuinely completed; this draft PR is only
  preparatory code, not experimental results.
