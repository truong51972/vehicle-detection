# v3 — Adaptive local contrast at trained S20/960

The current continuation tests bounded, highlight-protected, luminance-only local
contrast against PR #52's trained Raw S20/960 detector. It preserves the earlier
Gamma/Sharpen pilot as an immutable negative result. No historical policy is
renamed or retrained.

**Decision: stop before confirmation/training.** The adaptive arm's measured
+0.147 AP-point primary gain is below the prospectively required +0.2 points.
All declared uncertainty, rare-class, regression, route-control and cost checks
are retained in the [complete result](../../../docs/ISSUE47_V3_960_RESULTS.md).
No v3 policy/configuration, S20 export, checkpoint or development metric is claimed.

The independent notebook lifecycle is complete:

1. [01_experiments.ipynb](01_experiments.ipynb): research design, native paired
   exploration, per-class/cohort results, uncertainty, routes and CPU cost.
2. [02_train_prepare.ipynb](02_train_prepare.ipynb): verifies the failed gate and
   records skipped preparation; no original images or local exports are needed.
3. [03_train.ipynb](03_train.ipynb): records skipped independent training and the
   PR #52 recipe that would have applied only after confirmation.
4. [04_eval.ipynb](04_eval.ipynb): records why there is no trained v3 development
   evaluation. Calibration input experiments are not relabeled as model training.

Default Restart & Run All verifies committed evidence in a fresh kernel, from
this directory or the repository root. Each notebook reads saved prerequisites
without executing another notebook. There is no override for failed scientific
gates. `ALLOW_PILOT=True` in 01 is the original one-shot acquisition switch;
existing/incomplete attempts block execution and must never be removed to retry.

The [prospective plan](../../../docs/ISSUE47_V3_960_PLAN.md), exact compressed
membership and implementation were committed at `55d5b04` before measurement.
All 1,479 remaining exploration images / 926 source groups were fixed after
excluding recorded Issue #6/#11 and later searches. Three native 960 arms share
original five-class GT/negatives and JPEG95: Raw, always-on local contrast, and
pixel-routed local contrast. Only the adaptive arm could advance. The unchanged
PR #52 Raw best checkpoint is the detector, not its original COCO initialization.

The trained reference's 9,781 S20 / 13,975 development IDs and full 20-epoch recipe
are unchanged. Candidate search opens neither development nor confirmation pixels.
Final-test pixels remain sealed; no S100 or v4 is introduced. A negative result
for this bounded frozen-detector policy does not reject all adaptive mechanisms
or establish the outcome of consistent transformed-domain training.

The [read-only comparison](../compare.ipynb) is the only cross-version UI. It keeps
historical S20/640 v0/v1/v2, PR #52 Raw960 development and current calibration
results distinct. Large pixels, weights, matching arrays and executed notebooks
remain ignored; complete feature/read/draw evidence is losslessly compressed.

Historical evidence: [original Gamma/Sharpen notebook](../../research/issue47/04_gamma_sharpen_pilot.ipynb),
[original plan](../../../docs/ISSUE47_PLAN.md), and
[unchanged original result](../../../docs/ISSUE47_RESULTS.md). Its failed gates
are not revised by this continuation.

## Issue closure (no follow-up PR required)

The [cross-study evidence and closure assessment](../../../docs/ISSUE47_V3_960_RESULTS.md#issue-47-closure-assessment-across-completed-studies) brings the historic denoising, ordered-filter, routing, Gamma/Sharpen, local-shadow and current trained-Raw960 studies into one bounded decision. None passed its own preregistered promotion/confirmation criteria; the full remaining fresh exploration pool has been consumed. This fulfills the issue's explicitly permitted negative-result branch without fabricating an S20-trained v3. Conditional preparation, training and development evaluation remain explicitly skipped, while the confirmation and final-test partitions stay sealed. Once PR #53 merges, Issue #47 can close as a completed negative study; a materially different mechanism would need a new hypothesis and fresh, legitimate evaluation data, not repeated tuning on these measurements.

No previous committed scientific evidence, frozen operator, 01 acquisition notebook or historical model was modified by this closure summary.
