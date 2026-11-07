# Issue #47 — separate method discovery

Open [01_method_discovery.ipynb](01_method_discovery.ipynb) from this directory or
the repository root. It reviews a saved diagnosis, ranked mechanisms and a fixed
640/960 native resolution diagnostic on PR #48's already-consumed images.
Default Restart & Run All reads only hash-verified compact committed evidence;
no datasets, models, study dependencies or inference are needed for inspection.

Read the [premeasurement discovery plan](../../../docs/ISSUE47_DISCOVERY_PLAN.md)
and [results/next-method decision](../../../docs/ISSUE47_DISCOVERY_RESULTS.md).
The fixed Gamma/Sharpen pilot remains an immutable negative result. Issue #47
remains open; this notebook does not replace the v3 experiment lifecycle or
insert resolution results into the cross-version S20 comparison.

`ALLOW_DIAGNOSIS=True` explicitly derives a new report from original annotation
metadata and verified saved matches. `ALLOW_RESOLUTION=True` explicitly authorizes
the committed two-evaluation diagnostic on a fresh output root. Original image
pixels, remaining exploration groups, reserved confirmation, development and
final-test pixels stay unopened. The exact old Skip JPEG95 exports/label bytes
are copied into owned files under `runs/issue47-discovery/resolution/`, including
new native caches. No old export or run can be overwritten by the validator.

Existing/incomplete attempts block acquisition. Keep `docs/issue47-discovery/`
and ignored `runs/issue47-discovery/` together for provenance; do not delete outputs
to repeat a failed measurement. The executed acquisition notebook and kernel log
remain local under that run root. The compact evidence is separately committed.

Class FN counts come from GT totals minus native fixed-confidence TP counts.
GT-size distributions and image-cohort AP cannot identify which individual boxes
were missed. Sparse classes remain inconclusive, absent AP is never zero-filled,
and image-cohort AP is not AP-small. Sequential GPU validation batch p95 is not
per-image latency or a matched CPU preprocessing benchmark.

Issue [#51](https://github.com/truong51972/vehicle-detection/issues/51) continues
with [03_full_resolution_training.ipynb](03_full_resolution_training.ipynb) and
the [finalized full-S20 protocol](../../../docs/ISSUE47_FULL_TRAINING_PLAN.md).
Both complete 20-epoch arms and all four matrix cells are reported in the
[full-S20 results](../../../docs/ISSUE51_RESULTS.md). All 42 actual CUDA selection
passes verified fixed-640 tensors; native selected epochs are 9 and 10. The
positive AP estimate remains inconclusive because dark-bus coverage conflicts.
Default Restart & Run All reads hash-verified committed evidence without datasets,
weights or study dependencies. Acquisition requires explicit guarded opt-ins and
cannot overwrite completed attempts. The notebook and its original pre-acquisition
note remain frozen at the acquisition revision. The interrupted CPU postprocessing
and subsequent saved-match continuation are documented in the results. Historical
experiments remain unchanged; no policy is promoted and Issue #47 stays open.
