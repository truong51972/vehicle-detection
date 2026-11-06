# Issue #45 — bounded local-shadow pilot

Declared before new pixel measurements or inference on 2026-11-06, starting from
`f202198` (completed S20 v0/v1). This is exploration, not a frozen v2 policy.

## Evidence and competing explanations

The [S20 report](ISSUE43_EXECUTION.md) gives v0/v1 development mAP50:95
20.6124%/20.6879%, with motorcycle 7.7450%/8.1438% and bicycle
9.8465%/10.2159%. Aggregate near equality and mixed class effects do not establish
a useful gain. The [round-3 report](ISSUE30_RESEARCH_ROUND3.md) describes small,
distant targets, low night-time TP coverage, severe-motion loss and routing gains
that failed to replicate. Its clean examples include lost TP and increased FP;
none assigns AP to an individual image. S20 has no saved subgroup/error analysis,
so those representative errors are historical pretrained exploration evidence,
not newly diagnosed S20 errors.

1. Local illumination, rather than whole-image mean brightness, may hide shadowed
   vehicles in otherwise bright frames. Test one bounded local exposure correction.
2. Blur and small targets may instead be limited by spatial information/model
   capacity. This operator cannot restore missing detail; record those subgroups.
3. Pixel enhancement may amplify noise or shift the pretrained distribution.
   Retaining Raw/v1 and stopping is a valid outcome.

Do not retry old Denoise 5, Sharpen→Bilateral, CLAHE grids or quality thresholds.
Do not change resolution, model capacity, training settings or data roles.

## One fixed treatment

Local shadow gain: compute grayscale illumination at maximum side 512 using
INTER_AREA, Gaussian sigma 16 at that scale, then
`gain = clip(sqrt(0.25 / max(illumination / 255, 0.0625)), 1, 2)`.
Resize the smooth gain field to source dimensions using INTER_LINEAR. Multiply
the original BGR channels by that field, clip to [0,255], cast to uint8.
No denoising/sharpening, learned routing, GT or scenario labels enter the operator.
This preserves geometry and scales original detail rather than replacing it with
a smoothed image. Clipping, noise amplification and color changes remain risks.
These constants are mechanism-based hypotheses, not fitted estimates.

Controls: exact `configs/v0.yaml` Raw and frozen `configs/v1.yaml` Gamma-or-Skip,
through the existing native operator. All arms encode JPEG95 exactly once.

## Sample, metrics, budget and stop rule

- Use only the saved calibration-exploration IDs, in the existing seed-2026
  deterministic whole-group order. Exclude every group in round 3's consumed
  512-image membership. Take the next whole groups to a target of 256 images;
  stop before inference if overshoot exceeds 16 images or any class has fewer
  than five source groups. Do not select by labels/outcomes or reuse these groups
  as independent confirmation. Record exact IDs/groups/GT and data/code hashes.
- Never open confirmation/development/final pixels. Do not open the reserved
  confirmation plan. Original sources are read-only; v0/v1 exports/runs stay intact.
- Exactly three native pretrained five-class evaluations on **natural clean data**,
  at most 816 image passes (3 × 272), no synthetic scenarios or parameter search.
  Use the pinned YOLO26n checkpoint and existing VehicleValidator/MatchingStats,
  FP32 640, batch 32, NMS-free, workers 0, seed 2026. Native dataset-level
  mAP50:95/mAP50, per-class precision/recall/AP, paired ID/label-byte checks.
- Diagnostic image subgroups: dark (existing brightness <0.12), blurred
  (existing sharpness <50), bright (brightness >=0.12), small two-wheeler
  (at least one original motorcycle/bicycle box area <32²). Keep every original
  GT in a subgroup image. Report per-class support; no per-image AP. Subgroup
  macro AP averages represented classes only, explicitly separate from five-class
  primary AP. Class/subgroup gates apply only at >=5 source groups per class;
  unsupported effects remain inconclusive, never imputed as zero.
- Matched direct CPU timing: one process, eight OpenCV threads, warm up all arms,
  three repeats of every source, rotate arm order. At most 2,448 timed calls;
  exclude reads, JPEG, inference and verification equally. Require candidate
  mean <=10 ms and p95 <=25 ms. Record wall time separately.
- Advancement requires primary mAP50:95 >=0.2 percentage points above **both**
  controls, no class AP50:95 loss >2 points, no supported subgroup/class loss
  >1 point, and dark subgroup observed-class macro gain >=0.5 points above both.
  All five dark-subgroup classes need >=5 source groups for advancement; missing
  support means insufficient evidence, not a pass. mAP50 is descriptive.
- Compute 100 paired seed-2026 source-group bootstrap draws with existing native
  AP helpers. Report conditional exploratory intervals and missing-class draws;
  a primary lower bound <=0 blocks advancement. No multiplicity/confirmation claim.
- Inspect representative fixed-confidence (.25), IoU .50 TP/FP changes and GT
  frames, including dark/blur/small-two-wheeler cases. These are errors/counts,
  never individual AP estimates.
- Stop this direction if any gate fails. Do not retune or try a runner-up on
  confirmation. No fine-tuning override is declared: weak pretrained evidence
  alone does not justify spending a new S20 run for this mechanism.

If all gates pass, freeze at most this one candidate and a separate disjoint
confirmation design before accessing any confirmation pixels. Only a defensible
selection then warrants `configs/v2.yaml` and downstream lifecycle notebooks.
S20 remains exactly 9,781 group-nested IDs and development exactly 13,975 IDs;
all five-class GT/negatives and shared Raw checkpoint selection stay fixed.

## Execution

Use `notebooks/experiments/v2/01_experiments.ipynb`, section 1:
`ALLOW_PILOT=True`. Default mode only reads committed evidence. The equivalent
API is `vehicle_detection.v2_pilot.run(ROOT, allow_inference=True)`.
Local outputs are isolated at `runs/issue45/pilot/`; an existing directory blocks
execution, including an incomplete run. Never overwrite it to conceal failures.
Compact measurements belong at `docs/issue45/`; executed notebooks, source
exports, native matching arrays and checkpoints remain ignored/local.
