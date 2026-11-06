# v3 — Adaptive exposure/detail exploration

V3 tests pixel-derived routes to Skip, mild Gamma, mild Sharpen, or the bounded
ordered Gamma → Sharpen composition. It does not reuse historical “Adaptive V3”
or modify completed v0/v1/v2 experiments.

**Decision: stop.** The completed pilot failed the primary gain, supported truck
regression, route-support and paired-interval gates. No policy is frozen and no
confirmation, S20 preparation/training or development evaluation was run.

Start with [01_experiments.ipynb](01_experiments.ipynb), independently executable
from this directory or the repository root. Default Restart & Run All verifies
and displays committed compact evidence without models, local data or inference.
Read the [committed premeasurement plan](../../../docs/ISSUE47_PLAN.md) and
[research decision](../../../docs/ISSUE47_RESULTS.md) for populations, hypotheses,
gates, missing support and limitations. The plan/code were committed before new
pixel measurement; exact sample/model/settings/code hashes are saved alongside
the [version-local evidence](../../../docs/issue47/summary.json).

The single pilot uses 1,027 natural calibration-exploration images in 649 whole
source groups, excludes both prior consumed rounds and reserved confirmation,
and compares four arms on identical original IDs/five-class GT with JPEG95.
Only the complete preregistered route may advance; components are diagnostics.
There is no threshold tuning, runner-up promotion or post-pilot training override.

`ALLOW_PILOT=True` authorizes the fixed matrix only on a fresh output root with
the original study dependencies/sources/checkpoint. An existing or incomplete
attempt blocks execution. The completed real acquisition notebook/log live under
ignored `runs/issue47/`; do not delete evidence to rerun consumed groups.

02/03/04 and `configs/v3.yaml` are intentionally absent: the declared gates do not
justify preparation, training or development evaluation. A future direction
needs a distinct preregistered hypothesis on remaining unconsumed groups, not
retuning this pilot. The [read-only comparison](../compare.ipynb) remains the only
cross-version UI and distinguishes this pretrained pilot from completed trained
S20 arms and historical S10 results. Final-test pixels stay sealed.
