# v2 — Later preprocessing exploration

Purpose: investigate whether bounded filters, sequences or quality-based routing
improve over Skip and frozen Gamma-or-Skip. This stable folder identifies the later
exploratory study. It is separate from the old algorithm name “Adaptive v2”.

Status: **exploratory; replacement policy not locked for paired training**.
The [issue #45 local-shadow pilot](../../../docs/ISSUE45_RESULTS.md) completed
three paired native evaluations on 256 additional exploration images. It failed
gain, dark-regression, uncertainty and mean-cost gates; **v2 remains untrained**.
The [premeasurement declaration](../../../docs/ISSUE45_PILOT.md) fixes its recipe,
sample budget and stop rules. `01_experiments` defaults to verified evidence;
`ALLOW_PILOT=True` opts into a new attempt and refuses an existing local output.
No v2 detector has been trained or evaluated. There are no placeholder downstream
notebooks or fabricated runs. [01_experiments](01_experiments.ipynb) independently
reads actual saved decisions; exact historical search execution is available via
[Git replay](../../../docs/HISTORICAL_REPLAY.md).

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
No `configs/v2.yaml` or downstream lifecycle phases are added after a failed gate.
A future distinct policy needs a new preregistered pilot; no threshold retuning or
confirmation is justified by these measurements.
