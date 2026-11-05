# Issue #33 — verified reuse of the completed Raw/v2 S10 study

**Both existing five-class native runs qualify for reuse. No new GPU training,
policy selection or inference was needed.** The separate P4/P5 notebooks completed
in fresh kernels using their documented defaults and saved prerequisites. The
[execution contract](ISSUE33_IMPLEMENTATION.md) describes both reuse and fresh-run
behavior.

## Input parity

[P4's immutable receipt](round2-training-v1/preparation.json) verifies exactly
**4,890 S10 images per branch**, including **34 negative frames**, and the shared
**13,975-image Raw development input**. Raw and frozen Gamma-or-Skip v2 retain the
same IDs, five-class labels and geometry. Original annotation/image hashes,
frozen protocol/subsets/policy, class remapping and JPEG95 encoding all match.

Every selected export passed deep image/label verification. Replaying the frozen
recipe from original pixels reproduced all output hashes and actual routes.
[The per-image source/route manifest](round2-training-v1/routes.csv.gz) records
**1,107 Gamma** and **3,783 Skip** decisions; these are execution counts, not AP
improvement claims. No new labels, membership, policy thresholds or original
images were changed. Existing exports and verification receipts were read only.

## Checkpoint equivalence and costs

[P5's handoff](round2-training-v1/training.json) authenticates the actual pair
manifest, native args, native histories and all four best/last checkpoints against
the preserved [#21 evidence](issue21/s10_results.json). The current P4 exports and
training settings match that historical study: original COCO-80 initialization,
five-class mapping, policy/source identities, seed 2026, freeze=11, 640-pixel
inputs, batch 16, FP32 and the full 20-epoch native budget. Every checkpoint loads
with the correct five-class head, matching native version/settings and saved
best/final history metrics.

Both best checkpoints select **epoch 6**; both runs finish **epoch 20**. Raw's
recorded training/validation time is **1,477.653 seconds**, including a native
resume at epoch 13 after an interruption of unconfirmed cause. Adaptive records
**1,439.300 seconds** without an observed restart. The best checkpoints precede
Raw's interruption; reuse preserves it rather than claiming an uninterrupted
replicate. Native times exclude setup, downtime, discarded partial-epoch work
and final best-checkpoint validation. GPU memory peaks were not recorded.

[Learning curves](round2-training-v1/convergence.png) and the unmodified
[Raw](round2-training-v1/raw_results.csv) /
[Adaptive](round2-training-v1/adaptive_results.csv) histories are copied into the
new report root. These are historical native training/Raw-development metrics,
not new evaluation results or evidence of reliable single-seed superiority.

## Validation and #35 handoff

The real-data P4 replay ran from the repository root in **252.49 seconds**. Its
guard recorded exactly **18,865 original S10/development source images** and zero
reserved-source or protected-write violations. P5 ran independently from the
`notebooks/` directory in **15.70 seconds**, with **zero image reads**, zero guard
violations and `YOLO.train` blocked. Neither kernel rescanned BDD or re-exported
the existing inputs. Executed copies remain local under
`runs/round2-training-v1/`.

The [validation receipt](round2-training-v1/validation.json) records checks and
hashes for the new evidence. Fixture checks exercise missing-export preparation,
unchanged repeated reuse, false-route detection, reserved-role rejection,
historical artifact tampering and independent fresh-kernel P4/P5 reuse. Tiny
fixture training remains implementation validation, not research evidence.

**Proceed to #35 with the exact Raw/v2 checkpoints identified in `training.json`.**
There is no v3 arm, S20 expansion, fresh confirmation consumption or final-holdout
evaluation. Original six-class/five-class data, model artifacts and PR #37–#39
evidence remain preserved. This handoff satisfies #33 through verified reuse;
it does not claim a new training replicate or close #35's development analysis.
