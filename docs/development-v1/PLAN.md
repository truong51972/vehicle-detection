# Development completion and scale/model selection plan

Recorded **2026-11-03 16:28:30 UTC**, after the S10 training/Raw epoch outcomes
were already visible and before new 3×2 development inference in this work.
S10 is exploratory for scale selection. These rules are prospective only for the
remaining development work; they are not a preregistration of S10.

## Fixed work and estimands

- Retain the exact S10 Raw/Adaptive best checkpoints selected by the original,
  equal-budget Raw-development protocol. No new optimizer steps, seed, epoch,
  augmentation, split, model architecture or preprocessing policy is chosen here.
- Complete six evaluations on all 13,975 development IDs: pretrained / Raw-trained /
  Gamma-or-Skip-trained × Raw / Gamma-or-Skip inputs. Use native six-class AP,
  imgsz640, batch32, FP32, conf0.001, IoU0.7, max_det300, rect=True, nms=False,
  deterministic seed2026 and the pinned Ultralytics runtime. Retain per-image native
  matching statistics so reporting never repeats inference.
- Six-class mAP50:95/mAP50 and per-class AP are descriptive, with missing classes
  undefined. Report weather/timeofday/scene supports and point metrics on the
  original natural-condition slices; never reweight or rebalance them.
- Paired overall comparisons: each model's Adaptive-minus-Raw input effect; both
  trained models minus pretrained on Raw inputs; Adaptive-trained minus Raw-trained
  on Raw and Adaptive inputs. Condition uncertainty compares the two trained models
  separately within each input treatment. This is seven overall contrasts and two
  sets of natural-condition contrasts; no selected favorable slice is promoted.
- Each contrast uses identical IDs/GT and source-group draws. Freeze its eligible
  class target from GT before comparing predictions: at least five source groups
  containing the class; condition needs at least five total source groups. Keep the
  same class denominator for point delta and interval. Unsupported classes/slices
  retain descriptive points and reasons without an interval. Use 100 seeded paired
  group bootstrap draws, report invalid draws and class-specific denominators, no
  imputation or redrawing. Nominal percentile 95% intervals are exploratory,
  conditional on class presence, without multiplicity adjustment. They quantify
  development-group variability, not training-seed variability or generalization
  to all cameras. A five-group engineering gate is not statistical adequacy.
- Measure native CPU loader throughput separately on the same 4,890 S10 exports,
  batch16, imgsz640, workers8, disabled augmentation/cache, one warm-up epoch and
  three measured epochs. Exclude trainer/model/GPU work; report wall times, actual
  image counts, OS-cache/warmup caveats and environment. Never substitute aggregate
  GPU training throughput for this measurement.

## Scale and checkpoint decision

Stop at paired S10 for this study unless independent confirming evidence is later
specified. Existing Raw-development curves peak at epochs 4/6 and decline over the
remaining equal budget, and one seed cannot establish a reliable Adaptive benefit.
The new matrix diagnoses input effects; it is not permission to tune the already
frozen policy or to expand only a favorable branch.

A future S20 proposal must freeze its resource budget and confirmation seeds before
training, start both branches from the same original pretrained source on exactly
the same frozen S20 IDs, and use new identities. Require at least 0.01 absolute
Raw-development mAP50:95 improvement over the respective locked S10 checkpoints
for **both** branches, no supported class drop larger than 0.01 without documented
review, and independently confirmed gains under the declared budget. Any S10
continuation must be labeled curriculum, never independent from-pretrained S20.
No S20/S40/full run is authorized by this plan alone.

For the final three-model comparison retain pinned pretrained plus **both** existing
S10 best checkpoints and v2 Gamma-or-Skip, provided completion, identity, finite
metrics and semantic checks pass. Keeping the paired models avoids selecting a
winner from one seed. Freeze exact hashes, S10 membership, role manifest, settings,
policy, development receipts and stopping rationale in a metadata-only lock.
A scale curve has only the observed S10 point per branch; never invent S20 values.

The lock is preparation for #8, **not final-test release**. Final pixel/geometry/
content-leakage checks and the explicit one-time release remain outstanding.
