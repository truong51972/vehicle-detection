# Epic #14 — Five-class handoff for Codex (WIP)

Branch: `feat/epic14-five-class-prep` (do **not** merge until validated).

## Already prepared

- `src/vehicle_detection/five_class.py` provides explicit six→five YOLO-ID remapping,
  dropping **railway train only** (`3→DROP`) and preserving all surviving numeric
  coordinates and image IDs. Negative-only frames remain present.
- `export_variant`, `verify_alignment`, `verify_role`, `export_pair` and
  `export_development_pair` accept optional class information. Historical
  six-class callers still use their original defaults.
- Notebook 02 has a **new section 8** deriving a separate S10 training and
  development pair under `runs/preprocessing-five-v1/`. It reads previous
  six-class conversion/metadata and does not materialize final-test images.
- `tests/test_five_class.py` exercises mapping, preserved source files,
  all-negative frames and paired native export verification.

## Next implementation tasks (delegate, keep small)

1. **#20:** Validate new section 8 on actual BDD100K; profile cost/disk use,
   count old/new instances by role and attributes, confirm no role leakage and
   zero source modifications. Because notebook 02 still executes the historic
   six-class stages first, decide whether a small explicit `FIVE_ONLY` execution
   path is needed to avoid rereading/rewriting those outputs on subsequent runs.
   Do not invent a new export framework. Mark #20 complete only after verification.
2. **#21:** Adapt the *existing* `prepare_pair`/training notebook to consume
   `runs/preprocessing-five-v1/datasets/S10` and
   `runs/preprocessing-five-v1/development/raw`; preserve native YOLO
   `freeze=11`, equal optimizer/seed/settings, independent pretrained
   initialization, explicit versioned output paths and `last.pt` resume.
   Add a tiny CPU smoke fixture; **do not fabricate full-data training**.
3. **#22:** Extend the existing COCO→BDD baseline adapter for the five-class map
   `{2:0, 7:1, 5:2, 3:3, 1:4}`. Keep the historical six-class adapter intact.
   Parameterize development evaluation, saved matching stats, class support,
   3×2 report and notebook 04 cache identities by explicit class names.
   Ensure one seed's bootstrap is described as exploratory and omit missing AP.
   No final-test pixels; no full inference without explicit opt-in.

## Run and verification

```sh
uv sync --locked --extra study --python 3.11
uv run --extra study pytest -q tests/test_five_class.py
uv run --extra study pytest -q
uv run --extra study ruff check .
uv run --extra study ruff format --check .
```

Then run notebook 02 using a checkout containing the existing BDD100K and
frozen six-class artifacts. Check `runs/preprocessing-five-v1/summary.json`,
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
