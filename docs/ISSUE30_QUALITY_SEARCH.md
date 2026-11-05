# Issue 30 — quality-derived exploration increment

This increment starts the remaining work in
[issue #30](https://github.com/truong51972/vehicle-detection/issues/30).
PR #37's measured rejection, frozen v2, original sources and evidence files stay
unchanged. **No new v3 is released and issue #30 is not complete.**

The old evidence manifest also binds the active notebook's original bytes. Those
bytes are preserved at
`notebooks/archive/round2-screening/03_preprocessing_experiments.ipynb`, with the
original hash/revision in `notebooks/archive/manifest.json`. When checking the old
evidence manifest, resolve its notebook entry to this archive; every other entry
remains at its recorded path. The old manifest itself is unchanged.

Notebook `03_preprocessing_experiments.ipynb`, section 6, provides an independent
opt-in exploration stage. Set `RUN_QUALITY_SEARCH=True`. Keep `ALLOW_INFERENCE=False`:
the new stage does not construct a detector, export images, train or evaluate
confirmation/development/final pixels. The earlier sections retain their historical
replay behavior. The new stage needs the existing local exploration caches, original
exploration images and preserved PR #37 reports.

New caches use `runs/round2-quality-v1/`; compact reports use
`docs/round2-quality-v1/`. Plans bind code, numeric versions, OpenCV threads, prior
sample/plan identities and cost limits. Replaying saved quality uses its checksum;
changed code, budgets or inputs require a new output root. Preserved native result
and matching hashes, inference identities, fixed five-class GT, input receipts and
cost inventory are checked before searching. No historical evaluator/source file
was edited, so its original cache identities remain replayable.

## Exploration and search

The quality audit measures the saved exploration sample in all ten scenarios
before treatment/final JPEG95. It retains brightness, contrast, high-pass noise,
Laplacian sharpness, median-residual noise and clipped pixel fractions. Its report
contains per-scenario/severity quantiles, zero rates, source/class support,
condition/instance counts and representative IDs. The notebook plots overlapping
quality ranges and displays clean representative images; all per-scenario
representatives remain available by ID for inspection.

Noise and sharpness remain proxies: texture raises both, clipping can collapse
median noise to zero, and overlapping clean/perturbed ranges do not identify a
corruption. Brightness/contrast/sharpness use the existing maximum-512-pixel-side
quality implementation; high-pass noise uses the existing full-resolution diagnostic.
The new stage does not calibrate either proxy into a physical sensor measurement.

Thresholds are the pooled exploration 25th/50th/75th percentiles of four routing
features, in both directions. The search considers at most 384 one-rule configs:
four features × three thresholds × two directions × sixteen measured actions.
Actions reuse the already measured settings and ordered one/two-filter sequences
from PR #37; no unmeasured parameter/sequence is assigned detection performance.
It then tests up to twelve ordered two-rule configs from the top four distinct
single-rule assignments. Equivalent routing assignments are deduplicated. The
four-entry beam bounds computation and can miss better combinations.

Routing is first-match on input quality, with Skip as the fallback. Native matches
are selected per image from the chosen measured action, then aggregated by the
existing Ultralytics AP helper over the **entire paired dataset**. No image receives
a dataset-level AP label. Scenario labels select the paired measurement cache and
objective weighting; neither scenario labels nor GT enters an inference rule.

The explicit cost constraints are mean ≤10 ms and p95 ≤25 ms in **every scenario**.
These are engineering budgets, not measured deployment requirements. Cost includes
the measured quality diagnostic plus the selected action's recorded preprocessing
time, excluding source reads/corruption/JPEG/detector time. This is a screening
estimate from separate timings, not a new direct end-to-end latency benchmark.
Changing either budget requires a new plan/root before inspecting new routing scores.

Score uses the preserved equal-family/equal-severity native five-class mAP50:95
objective. Candidates must strictly beat both controls (ties retain a baseline),
meet five-group class support, and satisfy the existing clean/scenario/class
regression tolerances. Both native mAP50 and mAP50:95, cost, route frequencies and
rejection reasons are retained; a selected route also retains subgroup AP/support
and representative TP-loss examples. Internal reuse of the old gate function does
not turn exploration into confirmation. Outputs are `search.json`, never `policy.json`.

## Confirmation boundary and remaining work

The metadata-only fresh sample stays within the frozen confirmation half. It
excludes both merged leakage groups and original source groups used in PR #37's
exploration/confirmation samples and the current optimization sample. Selection
uses a deterministic whole-group prefix, without class balancing or looking at AP.
The run cache contains exact frozen rows; the compact report contains available
group/image counts, selected class/condition support and whether the target was met.
Exhausted populations are reported without fabricating a valid sample.

Historical v2 already used calibration. These groups are unused within this round,
not a historically untouched holdout; available membership alone does not prove
adequate rare-class support or statistical precision.

Further work is explicit: inspect the distributions and exploration decisions;
justify and measure any additional bounded operator values; freeze the chosen config
and rationale; wire the new sample into a separate verified export/evaluation path;
measure direct mean/p95 runtime; and perform one new native confirmation with
source-group uncertainty. The existing historical exporter intentionally accepts
only its original deterministic samples. Do not feed the fresh sample through it
by weakening its validation or relabeling its identity. Development stays descriptive
and final holdout stays sealed. #33 must keep the rejected Denoise 5 blocked and
make an explicit baseline/scope decision if no valid improvement is confirmed.

## First measured exploration replay

The local replay measured 2,560 quality records (256 exploration images × ten
scenarios) with detector construction blocked. It retained 119 distinct cost-feasible
routing assignments. The exploratory selection first applies Sharpen 0.15 →
Bilateral when contrast >0.2335527894; otherwise Gamma 0.70 when high-pass noise
<0.0019378723; otherwise Skip. These are the pooled exploration contrast upper
quartile and high-pass noise median;
`search.json` retains their full precision and exact ordered config.

Family-balanced native mAP50:95 is 0.1851148, versus Skip 0.1820893 and frozen v2
0.1822372, an exploratory gain of 0.2878 percentage points against v2. Estimated
per-scenario mean cost ranges from 6.93 to 9.96 ms; the highest p95 is 14.13 ms.
These timings include the measured diagnostic overhead and remain screening estimates.
The quality stage explicitly uses one OpenCV thread (`QUALITY_THREADS=1`), matching
its recorded measurements; the preserved operator costs used PR #37's eight-thread
configuration. This mixed timing estimate adds separate diagnostic overhead to
historical action time, not a measurement of a single runtime configuration.
The search meets the existing point-regression tolerances on its optimization data;
this is not independent evidence, uncertainty or a v3 acceptance decision.

The unused confirmation population contains 3,204 images / 1,962 groups after the
declared exclusions. Its fixed 256-image prefix has 146 groups, 20 motorcycle
instances across 10 groups and 30 bicycle instances across 16 groups. This clears
the five-group engineering floor; new native confirmation, direct latency and
source-group uncertainty are still pending. Exact rows stay in the run cache,
with compact support and immutable exploration evidence under
`docs/round2-quality-v1/`. Do not treat this candidate as a prerequisite satisfied
for #33 or close #30 on this exploratory result.
