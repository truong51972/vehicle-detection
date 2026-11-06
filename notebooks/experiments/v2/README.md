# v2 — Local-shadow (exploratory)

Purpose: measure the fixed local-shadow policy and its matching detector. The
earlier filter/routing exploration remains historical provenance, separate from
the old algorithm name “Adaptive v2” (today's v1). Adaptive-routing v3 belongs
in a future separate issue/PR; it does not replace this negative v2 experiment.

Status: **exploratory S20 train/eval completed; no confirmed replacement policy**.
The [issue #45 local-shadow pilot](../../../docs/ISSUE45_RESULTS.md) completed
three paired native evaluations on 256 additional exploration images. It failed
gain, dark-regression, uncertainty and mean-cost gates. That historical result is
unchanged. A separate, [committed post-pilot plan](../../../docs/ISSUE45_FINETUNING_PLAN.md)
then froze the same local-shadow recipe as an experimental v2 training arm.
The [premeasurement declaration](../../../docs/ISSUE45_PILOT.md) fixes its recipe,
sample budget and stop rules. `01_experiments` defaults to verified evidence;
`ALLOW_PILOT=True` opts into a new attempt and refuses an existing local output.
The [completed S20 comparison](../../../docs/ISSUE45_S20_EXECUTION.md) records
policy-matched development mAP50:95 **20.4930%**, below v0/v1, with higher CPU
preprocessing cost. It is development, single seed, not a released improvement.
[01_experiments](01_experiments.ipynb) independently reads saved pilot and S20
decisions; exact historical search execution is available via
[Git replay](../../../docs/HISTORICAL_REPLAY.md).

- [02_train_prepare](02_train_prepare.ipynb): explicitly prepare frozen S20 and
  development inputs from originals, preserving all five-class GT/negatives.
- [03_train](03_train.ipynb): one native 20-epoch independent initialization,
  selecting on the same v0 Raw development export; no training without opt-in.
- [04_eval](04_eval.ipynb): evaluate/inspect only the local-shadow policy–detector
  pair. Cross-version actions and acquisition opt-ins are not part of this notebook.

Use [compare.ipynb](../compare.ipynb) as the only cross-version comparison UI.
It verifies/reads committed metrics, support, timing and plots in a fresh kernel
without local runs/weights/datasets. It never trains, exports sources, runs
inference or invokes diagnostic replays. The existing S20 execution report is
historical provenance at `c118ab2`; its acquisition commands are not the current UI.

`configs/v2.yaml` exposes no tuning knobs. Native data/models remain local under
`runs/v2/`; execution/diagnostics under `runs/issue45-s20/`; compact evidence under
`docs/issue45-s20/`. Existing v0/v1 runs and final-test pixels remain untouched.

The preserved screening used 20 bounded controls/operators/sequences/routes,
five-class calibration exploration/confirmation, seed 2026 and the pinned pretrained
YOLO26n checkpoint. Denoise 5 failed disjoint confirmation, so Gamma-or-Skip remained.
Later quality-derived routing and a 512-image exploration replication also retain
their own gates, frozen configurations and original attribution. None supplies a
replacement policy–trained-detector pair. Shared development/holdout membership
remains unchanged and the final holdout stays sealed.

Actual evidence lives under `docs/round2-preprocessing-v2/`,
`docs/round2-quality-v2/`, `docs/round2-research-v3/` and corresponding gitignored
`runs/round2-*/` roots. They were not copied or relabeled as new `runs/v2/` measurements.
Exact tested parameters remain in those historical plans/search records, not a new
experiment registry. See [screening results](../../../docs/ISSUE30_RESULTS.md),
[quality review](../../../docs/ISSUE30_QUALITY_REVIEW.md) and
[replication](../../../docs/ISSUE30_RESEARCH_ROUND3.md).

Issue #45's new pilot evidence lives separately under `docs/issue45/` and local
`runs/issue45/pilot/`; it does not overwrite or relabel the historical research.
The later training extension tests a separately declared hypothesis; it does not
retroactively pass those gates. A future distinct policy needs a new preregistered
study; no threshold retuning, confirmation or release is justified here.
