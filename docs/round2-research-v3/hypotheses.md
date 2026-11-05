# Round 3 premeasurement research log

Contract: [issue #30](https://github.com/truong51972/vehicle-detection/issues/30)
and [workflow change](https://github.com/truong51972/vehicle-detection/issues/30#issuecomment-6662593339).
This root names research round 3, **not policy v3**. No confirmation is authorized
by this experiment. PR #38's 1,025-image reserved plan will not even be read.

## Observations before new inference

PR #37's denoise winner failed confirmation; this is a warning against selecting
another small-sample maximum, not permission to optimize against that result.
PR #38's more expensive route remains unconfirmed and reduced fixed-confidence
clean TP while increasing FP. No prior configuration or evidence is modified.

Only historical **exploration** outcomes inform these hypotheses. Bilateral AP
was above Raw for mild/medium synthetic shot/read noise (20.54/18.51% vs
19.47/18.08%) but below it for severe noise (13.76% vs 14.11%). It reduced clean
AP (21.60% vs 21.88%). Gamma 0.70 reduced clean and noise AP. More blanket
brightness correction, wider parameter grids and expensive NLM are not justified
for this bounded round. This does not rule out those methods generally.

High-pass medians are 0.00182 clean, 0.01666/0.03228/0.05798 for noise severities
1/2/3; severe motion is 0.00088. The 95th clean percentile, 0.005460915116929616,
activates on 5.08% clean images and 100% of each noise severity in the old sample.
An upper bound, the midpoint of medium/severe noise medians
0.04512847858740014, is a **single fixed hypothesis**, not an optimized threshold.
Synthetic labels inform experiment design only; runtime sees pixels alone.

Visual inspection of the four most TP-deficient clean frames shows many distant,
small vehicles, crowded scenes and a tunnel exit. Old clean frames contain 2,851
five-class targets, 1,179 below 32x32 pixels; Raw yields 929 TP50 at confidence .25.
The dataset also has substantially lower fixed-confidence detection coverage at
night (204/937 targets) than daytime (631/1641). These image-level counts are
illustrative, not per-object error attribution or per-image AP labels. Geometry,
object scale and detector capacity may dominate some failures; smoothing/contrast
cannot recover missing spatial information. No detector-resolution change or
training experiment belongs to this preprocessing comparison.

## Ranked competing hypotheses and expected signals

1. **Generalization/null:** the locked PR #38 route's gain is largely sample
   selection. Replicate its exact configuration on 256 previously unevaluated
   exploration images. Failure blocks promotion even if pooled 512-image AP rises.
2. **Conservative noise-range routing:** avoid Gamma, texture sharpening and
   clean/severe-noise smoothing. Skip above the upper noise bound; otherwise
   bilateral above the lower bound; otherwise Skip. Expected signal: mild-noise
   gain with near-Raw clean/severe results, fewer clean FP/TP regressions, lower
   cost than PR #38. No new feature, operator or search framework is required.
3. **Simpler static comparator:** bilateral alone may generalize as well as
   routing. Measure it to isolate whether the gate adds value. Reject it if clean,
   severe noise or class regressions outweigh any average gain.
4. **No useful preprocessing:** retain Raw/frozen v2 if neither fixed approach
   clears both controls and regression/cost constraints. Do not spend confirmation
   merely to obtain a release. More thresholds are not the default next step.

## Budget and stopping rule fixed before measurements

Use the existing deterministic whole-group exploration prefix, target **512**:
256 old + 256 new images. Exporting the full prefix reuses the existing validated
sampler/exporter; new/old native dataset-level AP is recomputed from native
matching statistics. The new 256-image slice is the primary advancement check,
not confirmation: it is exploration and now available to inform future research.
No role, group, labels or source pixels change. No selection by detector outcomes.

Five candidates (Raw/Skip, frozen v2, exact PR #38 leader, bilateral, noise range)
on five sentinels (clean, mild/severe noise, severe underexposure, severe motion):
**25 native evaluations, 12,800 image passes**. If any treatment gains at least
**0.2 AP percentage points against both controls** on the new slice's equally
weighted four-family sentinel score and passes clean >=-0.5 pp, other scenario
>=-1 pp and each class/scenario >=-2 pp, complete the remaining five scenarios
for the same fixed matrix. Otherwise stop. Maximum: **50 evaluations / 25,600
image passes**. Never interpret the sentinel score as the full ten-scenario score.

Direct timing: all five treatments in one eight-thread process, warmup, rotated
order, three repeats, all ten scenarios / 512 images (**76,800 calls**). Exclude
reads, corruption, JPEG, detector and artifact verification equally. Existing
selective router must match frozen-router pixels. Cost gates: every scenario
mean <=10 ms and p95 <=25 ms. Timings are CPU preprocessing, not deployment latency.
A runtime failure blocks a recommendation for confirmation; it cannot release.

Paired source-group bootstrap: **100 fixed-seed draws**, same groups across
scenarios and treatments, fixed five-class macro, audit missing-class draws.
Intervals are exploratory, conditional on this sampled population, with no
multiplicity correction. Small rare-class support limits interpretation.

No candidate is released in this round. A promising completed matrix would justify
prelocking a future single confirmation, with gates and source-group uncertainty;
otherwise report no demonstrated improvement and retain the established baseline.
