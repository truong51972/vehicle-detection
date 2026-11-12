> Status update (2026-11-12): v4 implementation and exploratory full training/evaluation are complete. The matched Raw comparison does not improve overall AP. These are local issue documents, not newly created GitHub issues. Proposed independent pilot/confirmation gates remain unperformed; see [results](../RESULTS.md).

# [v4 / P4] Evaluate Canny-guided detail and report the paired Raw/v4 2×2 comparison

## Prerequisite

One completed, frozen v4 run and a comparable current-recipe Raw run. Do not compare
new full fine-tuning against historical frozen-backbone results as matched arms.

## Work

- [ ] Run v4 `04_eval.ipynb` only for v4 model / v4 processed development; native
  FP32 validation, conf 0.001, IoU 0.7, max_det 300, imgsz 640, five-class original GT.
- [ ] Save native overall/class metrics, paired matches, source hashes, selected
  checkpoint identity, Apply/Skip and raw dark/bright cohort support and metrics.
- [ ] Keep cross-version evaluation exclusively in `compare.ipynb`: A=Raw/Raw,
  B=Raw/v4, C=v4/Raw, D=v4/v4. Reuse valid caches; incomplete v4 must not block v0–v3.
- [ ] Report B-A, D-B, C-A, D-C, D-A and interaction `(D-C)-(B-A)` with paired
  source-group bootstrap CIs, shared group draws and per-class/cohort support.
- [ ] Inspect background-edge false positives, halos/clipping, small motorcycles
  and bicycles. Keep identical policy-defined Apply/Skip IDs in every 2×2 arm.
- [ ] Report policy CPU mean/p95 separately from detector timings, export time and
  training cost. Label Raw-JPEG proxy timings; do not call native detector timings
  end-to-end latency or infer preprocessing speed from file-export wall time.
- [ ] Update v4 README with actual metrics and honest negative/inconclusive results.
  Preserve config/model/data hashes and limitations without overwriting old evidence.

## Acceptance / boundaries

No development policy tuning or final-test access. One-seed development comparisons
and conditional bootstrap CIs do not establish generalization. Comparisons must
identify current training/checkpoint-selection settings and any deviations.
An absent cohort is not AP zero; no fabricated outputs or retrospectively passed gates.

**References:** v4 parent, exploration and training issues; #45, #47; `compare.ipynb`.
**Status:** comparison support implemented locally; no real evaluation executed.
