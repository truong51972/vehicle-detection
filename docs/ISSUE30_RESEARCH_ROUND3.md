# Issue 30 — autonomous exploration round 3

This increment follows the revised [research contract](https://github.com/truong51972/vehicle-detection/issues/30)
and [workflow comment](https://github.com/truong51972/vehicle-detection/issues/30#issuecomment-6662593339).
It preserves PR #37/#38 and tests a small set of independently selected hypotheses.
`round2-research-v3` is the third exploration round's directory name, **not a v3
policy**. Confirmation, development and final holdout are not used here.

The [premeasurement research log](round2-research-v3/hypotheses.md), committed in
`fa25cea` before inference, records observations, competing explanations, rejected
directions, the fixed noise-range rule, budgets and the advancement rule. The
[plan](round2-research-v3/plan.json) binds code, model, settings, data identities,
source-group support and historical inputs. Measurements use the existing
Ultralytics validator/AP and checked JPEG95 exporter. No filter/evaluator library
is added or rewritten.

## Design and interpretation

The old 256-image exploration sample had already supported hundreds of routing
comparisons. This round expands the same deterministic whole-group exploration
prefix to 512 images and reports the new 256 separately. Both are exploration;
the new slice is now consumed for research and cannot serve as future independent
confirmation. Sampling preserves original roles, whole leakage groups, labels and
geometry. No rare-class oversampling or outcome-based image selection occurs.

Three fixed treatments compete against Raw/Skip and frozen v2: PR #38's exact
unconfirmed route; static bilateral; and a conservative noise-range bilateral
rule, with Skip outside the range. The range endpoints come from historical
exploration quality distributions, not new detection scores. Runtime receives
pixels only. It does not receive GT, scene attributes or synthetic scenario labels.

Five sentinel scenarios compare clean, mild/severe noise, severe underexposure
and severe motion. Their score weights each of the four families equally, sharing
the noise family's weight between its two endpoints. It **is not** the full
historical ten-scenario score. The preregistered stopping rule governs whether to
measure the other five scenarios. No runner-up is tried on confirmation.

Source support in the new slice is 154 car groups, 60 truck, 34 bus, **8 motorcycle
and 16 bicycle groups**. Those minimum counts remain small. Native dataset-level
AP50:95/AP50, class/scenario deltas and subgroup support are retained, together
with illustrative confidence-.25 / IoU-.50 TP/FP counts. Individual images never
receive AP labels. The original small-object diagnosis describes annotation
sizes and image-level counts; it does not attribute every missing detection to
small size or claim causal proof.

Paired uncertainty resamples original `source_group`, preserving the same seed,
group draws and five-class macro across candidates and scenarios before averaging
families. Missing-class draws are recorded and excluded without redraw or
imputation. The 100-draw percentile intervals are exploratory and do not correct
for prior search, multiple treatments or selection. They cannot establish a
confirmed replacement.

## Reproduction and evidence

From the repository root, with the existing study environment and frozen data,
protocol, five-class conversion tools, pretrained checkpoint and saved PR #38
quality input available:

```sh
# Explicitly authorize only this preregistered missing exploration matrix and timing.
.venv/bin/python -m vehicle_detection.research_round3 --infer --measure-runtime

# Cache-only native replay: no model construction or timing is allowed by default.
.venv/bin/python -m vehicle_detection.research_round3

# Verify hashes/native AP, reproduce uncertainty, errors and scientific figures.
.venv/bin/python -m vehicle_detection.research_round3_report
```

The runner can also be called from a notebook cell with
`from vehicle_detection.research_round3 import run; run()`; the existing notebook
03 bytes are preserved. Completed inference is reused after deep source/export
and native-cache validation. The output root is fixed deliberately: changed code,
inputs, thresholds or budgets require a new versioned experiment, not overwriting
this evidence. Native runs and 512 source rows remain in ignored
`runs/round2-research-v3/`; compact identities, membership and scientific figures
are committed. Reconstruct sample rows with the existing deterministic
`calibration_sample(..., target_images=512)` and compare their plan hashes.

The report command verifies `evidence.json` on subsequent use. Runtime and native
measurements remain fixed; uncertainty/figures are deterministic analyses of the
saved matches. Full cached runner replay still performs expensive preservation
checks and is not a quick report viewer. The Markdown/PNG/JSON reports need no
model or dataset to inspect.
