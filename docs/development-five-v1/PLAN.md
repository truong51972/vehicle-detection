# Five-class development evaluation plan — issue #22

Recorded 2026-11-04 before new five-class development inference. The paired
20-epoch S10 training curves and selected epoch-6 checkpoints were already observed.
This is exploratory development evidence, not a preregistration of training.

- Evaluate all 13,975 frozen development IDs in the same 3×2 matrix:
  pretrained / Raw S10 / Gamma-or-Skip S10 × Raw / Gamma-or-Skip JPEG95 exports.
  Class order: car, truck, bus, motorcycle, bicycle. COCO mapping is
  `{2:0, 7:1, 5:2, 3:3, 1:4}`; drop railway predictions. Fine-tuned heads use
  their native five IDs. Retain the historical six-class evidence independently.
- Use pinned Ultralytics native matching/AP, FP32, imgsz640, batch32,
  conf0.001, iou0.7, max_det300, quantize32, nms=False, rect=True,
  augment=False, seed2026, deterministic=True, workers8. Save native per-image
  matches, plots, checkpoint/export/settings identities; reuse valid results.
- Report overall and per-class mAP50:95, AP50, precision/recall and support.
  Natural weather/timeofday/scene slices are descriptive. Object-size slices
  select images containing original-pixel boxes <32², 32²–96², or ≥96²;
  they overlap and retain all GT/predictions, so are **not area-restricted AP**.
  Include class/size instance support separately. Absent AP is undefined.
- Two primary paired contrasts: Adaptive-trained minus Raw-trained, separately
  on Raw and Adaptive inputs. Use 100 seeded source-group bootstrap draws with
  a fixed GT-supported class denominator (minimum five source groups/class),
  no AP imputation/redraws. Report invalid draws and class support. Percentile
  intervals are exploratory, conditional and unadjusted for multiplicity;
  they quantify development-group variation, not training-seed variation.
  Report other input/baseline differences descriptively without selected claims.
- Keep model validation timing, preprocessing/export cost and training cost
  separate. Native mean timings exclude offline enhancement and application I/O;
  they are not online p50/p95 or end-to-end latency. Export timings from #20's
  instrumented/resumed run are not valid performance measurements.
- Stop at S10. Retain pretrained plus both independent paired best checkpoints
  for the later one-time holdout protocol; do not select an Adaptive winner from
  one seed. No new training, recalibration, synthetic stress or S20 work here.
- No direct six-vs-five macro-mAP improvement claim: denominators differ.
  A common-five algorithmic comparison would require comparable GT/model settings
  and is outside this evaluation. Freeze exact candidate/artifact hashes after
  checking the development matrix. Preparing #8's scope does not release holdout.

Final-test pixels remain sealed. No notebook imports a training runner or invokes
upstream phases; inference and bootstrap are explicit options.
