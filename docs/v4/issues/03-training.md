> Status update (2026-11-12): v4 implementation and exploratory full training/evaluation are complete. The matched Raw comparison does not improve overall AP. These are local issue documents, not newly created GitHub issues. Proposed independent pilot/confirmation gates remain unperformed; see [results](../RESULTS.md).

# [v4 / P2–P3] Freeze Canny policy, export matched S20 data and fully fine-tune YOLO26n

## Prerequisite

Complete the v4 exploration/confirmation decision. Do not promote a failed pilot
or start training merely because the notebooks exist. A separate, explicitly
labeled exploratory negative training hypothesis must be frozen if proposed.

## Work

- [ ] Commit the policy, v4 YAML, decision and exact hashes before real export/train.
- [ ] Use `02_train_prepare.ipynb` and `ALLOW_PREPARATION=True` to export one new
  dated v4 run from original sources; never overwrite another version/run.
- [ ] Preserve frozen 9,781 S20 train IDs, 6,985 calibration IDs and 13,975 development
  IDs, source groups, five-class GT and railway-only negatives. Verify every ID
  and label byte against Raw, identical dimensions, deterministic JPEG95 and Skip bytes.
- [ ] Confirm train/calibration/development all use the identical frozen offline
  Canny policy. No on-the-fly filtering or changes to model channels/architecture.
- [ ] Use `03_train.ipynb`, independently initialized from the same pinned COCO
  YOLO26n checkpoint: seed 2026, freeze 0, SGD lr0=0.01, 50 epochs, batch 16, AMP,
  imgsz 640, geometry-only augmentation. All native args remain in v4.yaml.
  Local workers=0 and batch16 are declared hardware adjustments; see EXECUTION_PLAN.md.
- [ ] Select `best.pt` on v4's processed calibration split, per current main.
  Report native args/results, selected epoch, wall time and model/config/data hashes.
- [ ] Run export/Skip/parity tests and fresh-kernel opt-out/opt-in fixture checks.

## Acceptance / boundaries

Native shared preparation and Ultralytics trainer only. `RUN_DATE` isolates all
artifacts under `runs/v4-YYYY-MM-DD`. Training/preparation switches default False.
Preserve v0–v3 policies/notebooks/results and original sources. Final-test pixels
stay sealed. No historical frozen-backbone metrics are relabeled as this full
fine-tune; any changed hardware/batch/settings are disclosed before comparison.

**References:** v4 parent and exploration issues; #41, #47; v4 README.
**Status:** lifecycle code implemented locally; no real export or training executed.
