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
- `tests/test_five_class.py` exercises mapping, preserved source files,
  all-negative frames and paired native export verification.
- `support_audit` summarizes class presence/instances, object sizes, new negative
  frames and natural conditions for all four frozen roles **from annotations only**.
  It does not open final-test pixels.

## Next implementation tasks (delegate, keep small)

1. **#20 (blocker before real Run All):** Notebook 02 currently executes the
   historical six-class export cells before section 8, even when only five-class
   outputs are wanted. Make the smallest change that allows five-class derivation
   from **existing** verified artifacts without rewriting six-class files. Do not
   add a notebook orchestrator or generic export framework. Then validate on
   actual BDD100K, including source preservation, counts, role IDs and disk cost.
   Mark #20 complete only after that verification.
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

After fixing the historical-export side effect above, run notebook 02 using
existing BDD100K and frozen six-class artifacts. Check `runs/preprocessing-five-v1/summary.json`,
negative-frame counts, matched IDs/labels and `verified.json` under both
branches. Inspect actual artifacts before authorizing full training.

## Guardrails

- Do not overwrite original annotations, six-class YOLO labels, archived
  notebooks, checkpoint weights, `runs/preprocessing-v1` or `runs/data-v1`.
- Do not change or recalibrate Gamma-or-Skip v2, resplit datasets, or open final-test
  pixels. S20 remains membership-only until a separate decision.
- The five-class and six-class macro-mAP denominators differ: never claim accuracy
  improved simply because a class was removed.
- Keep #14 open until #20–#22 are genuinely completed; this draft PR is only
  preparatory code, not experimental results.
