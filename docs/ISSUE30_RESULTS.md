# Issue 30 — Round 2 results: confirmation rejected the leading candidate

**No new v3 policy is released. Frozen Gamma-or-Skip v2 remains unchanged.**
The highest exploration score belonged to colored nonlocal-means denoising at
strength 5. Its gain did not reproduce on disjoint confirmation IDs: the
family-balanced mAP50:95 fell below both controls and several preregistered
regression limits failed. The single development transfer check did not establish
an advantage. Other candidates were not tested on confirmation; this result does
not establish that every possible filter or route fails.

All **260 native evaluations** are saved: 200 exploration, 30 confirmation and
30 development transfer evaluations, totaling **66,680 image passes**. This is
measured five-class BDD100K / pinned pretrained YOLO26n evidence, not fixture,
training or final-holdout results. [The implementation contract](ISSUE30_IMPLEMENTATION.md)
describes the operator grid and validation boundaries.

## Frozen method and populations

The [plan](round2-preprocessing-v2/plan.json) was fixed before new detection
measurements. The initial target of 128 images failed annotation-only rare-class
support checks, so the v2 plan expanded the seeded whole-group prefix to target
256 before comparing AP. It did not balance, oversample, relabel sources or change
any original role. The three group-disjoint samples contain:

- Exploration: 256 images, 155 groups; 9 motorcycle and 12 bicycle groups.
- Confirmation: 260 images, 156 groups; 9 motorcycle and 14 bicycle groups.
- Development transfer: 256 images, 146 groups; 8 motorcycle and 17 bicycle groups.

[Membership](round2-preprocessing-v2/membership.csv.gz),
[full frozen sample rows](round2-preprocessing-v2/samples.json),
[class support](round2-preprocessing-v2/sample_support.csv) and
[annotation support audits](round2-preprocessing-v2/support_audit.json) bind the IDs,
groups, source hashes, boxes and conditions. Rare-class support remains small:
confirmation contains only 10 motorcycle and 23 bicycle instances. The five-group
minimum is an engineering floor, not statistical adequacy.

Twenty preregistered treatments use exactly paired IDs, five-class labels,
geometry and native evaluation settings. Every branch starts from original pixels,
uses the same clean or seeded synthetic corruption, applies its treatment, then
encodes JPEG95 once. Controls are Skip and frozen v2. There are ten scenarios:
clean and synthetic underexposure, shot/read noise and motion blur at severities
1–3. No synthetic label enters quality routing. The synthetic families are controlled
approximations, not calibrated camera-physics models.

The selection score averages native five-class mAP50:95 equally across four
families: clean, underexposure, noise and motion. Severities receive equal weight
within a synthetic family. This is an explicit research objective, not an estimate
of deployment prevalence. The five-class target stays fixed. Compare treatments
**within a sample**; AP levels across these different samples are not paired gains.

## Exploration: a gain that required confirmation

The [complete ranking](round2-preprocessing-v2/ranking.json) and
[scenario deltas](round2-preprocessing-v2/exploration_deltas.png) retain negative
findings alongside gains. Family-balanced mAP50:95 was 18.8825% for Denoise 5,
18.2237% for frozen v2 and 18.2089% for Skip. The observed Denoise 5 gain was
**+0.6588 AP percentage points versus v2**, or +0.6736 versus Skip. Its clean
mAP50:95 also appeared better on exploration: 23.2660% versus Skip's 21.8779%.

The [shortlist](round2-preprocessing-v2/shortlist.json) locked Denoise 5 and hashes
of its exploration evidence **before** confirmation. It was not replaced after
confirmation results became available.

The bounded search also found:

- Denoise 2 scored 18.4154%; its weaker strength ranked below Denoise 5.
- The best sequence, Sharpen 0.15 → Bilateral, scored 18.2984%, below Denoise 5.
  Bilateral alone scored 18.2621%; the sequence's small observed increment is not
  evidence of a reliable gain. Reversing the sequence scored 18.1687%.
- Gamma 0.70 → CLAHE 1.2 scored 17.4456%; the reverse scored 17.0762%.
  Order affected the point estimate, and both remained below Skip.
- Both Denoise 2 / Gamma 0.70 orders scored below Denoise 2 alone. Adding a second
  filter was not automatically beneficial.
- The two quality-routing hypotheses scored 18.1315% and 18.1963%, both below
  Skip's 18.2089%. Their thresholds were not retuned using confirmation or development.
- Gamma 0.85/0.70 and CLAHE 1.2/2.0 all scored below Skip on this exploration
  objective. These are results for the tested settings and sample, not universal
  claims about those operator families.

## Confirmation: reject the candidate under the fixed gates

[Confirmation plots](round2-preprocessing-v2/confirmation_comparison.png) and the
[immutable decision](round2-preprocessing-v2/decision.json) document the rejection.
Family-balanced native mAP50:95 / mAP50 were:

- Skip: **16.7337% / 25.7824%**.
- Frozen v2: **16.8230% / 25.8472%**.
- Denoise 5: **16.2714% / 25.7123%**.

Denoise 5's mAP50:95 difference was **−0.4623 AP percentage points versus Skip**
and **−0.5516 versus v2**. Both fail the required nonnegative family-balanced gain.
Additional failures were observed:

- Clean mAP50:95 fell from Skip's 19.7736% to 19.2592%, a **−0.5144-point** change,
  beyond the 0.5-point clean-regression tolerance. V2's clean score was 19.8161%.
- Medium synthetic noise fell from Skip's 17.6927% to 16.5063%, a **−1.1864-point**
  change, beyond the 1-point scenario tolerance.
- Under severe synthetic low light, bus AP50:95 fell from v2's 37.6368% to
  33.3581%, a **−4.2787-point** change, beyond the 2-point class/scenario tolerance.
  Bus regressions also occurred in several other scenarios; medium-noise bicycle
  AP regressed beyond its class tolerance as well.

The gates are point-estimate engineering tolerances, not significance tests.
[Paired uncertainty](round2-preprocessing-v2/family_uncertainty.json) uses 100
source-group bootstrap draws, the fixed five-class macro and the same draws across
scenarios before family aggregation. All 100 family draws were valid. The
confirmation difference versus v2 had an exploratory 95% percentile interval of
**[−1.4674, +0.0919] AP percentage points**; versus Skip it was
[−1.3111, +0.1544]. These intervals include zero and do not establish a positive
gain or statistically certain harm. No multiplicity correction or reselection
was applied. [Per-scenario/class intervals](round2-preprocessing-v2/paired_uncertainty.json)
and [draw support](round2-preprocessing-v2/bootstrap_draws.csv.gz) remain available.

## One fixed development transfer check

The predeclared development sample was evaluated once, after the candidate and
confirmation decision were fixed. [Development plots](round2-preprocessing-v2/development_comparison.png)
show family-balanced mAP50:95 / mAP50 of:

- Skip: **12.3886% / 19.2740%**.
- Frozen v2: **12.3942% / 19.3328%**.
- Denoise 5: **12.3742% / 18.8411%**.

The mAP50:95 difference versus v2 was −0.0199 AP percentage points, with an
exploratory paired 95% interval of **[−0.5122, +0.4865]**. A small clean point gain
on this sample did not produce a family-balanced advantage. These descriptive
results neither reverse the confirmation rejection nor select another treatment.
No parameter, threshold, candidate or sample was changed using development scores.

## Cost, decisions and illustrative failures

[Accuracy versus cost](round2-preprocessing-v2/accuracy_cost.png) records the
tradeoff. Mean confirmation preprocessing time was approximately **165.7 ms/image**
for Denoise 5, **3.2 ms** for v2 and **0.21 ms** for Skip on the recorded machine
with eight OpenCV threads. These are offline CPU treatment timings, including
quality/routing work where used and excluding corruption, input reads, JPEG and
policy-artifact verification. They are not end-to-end deployment latency or GPU
batch inference measurements. Native detector speed stays in each native result.

Denoise 5 is a static filter: its actual confirmation decision was Denoise on all
260 images in every scenario. [Decision counts](round2-preprocessing-v2/decision_distribution.csv)
and [the plot](round2-preprocessing-v2/decisions.png) do not imply adaptive routing.
The two tested routing alternatives did not win exploration.
[Failure examples](round2-preprocessing-v2/failure_examples.png) show cases with the
lowest native IoU50 true-positive changes at confidence 0.25. Orange boxes are
frozen GT, not predicted boxes. These examples illustrate outcomes; they do not
replace AP or identify a mechanism for every regression.

## Validation, interruption and handoff

All 260 native result identities, result checksums, metric plots and saved matching
files are verified by the [evidence receipt](round2-preprocessing-v2/validation.json).
Reaggregated fixed-class AP agrees with native point differences to 1e-8. The
original inference kernel stopped during the final development export after 259
results. The last case resumed with identical code/plan and source/holdout guards;
[the resume receipt](round2-preprocessing-v2/resume.json) records zero rejected
accesses and unchanged protected inputs. No completed inference was repeated.

The implementation passed **130 tests**, Ruff lint/format, whitespace checks and
`uv build`. **A cache-only fresh-kernel Run All passed all eight code cells with zero error
outputs.** Model construction was blocked, all protected-input assertions passed,
and both source-write and reserved-image guards recorded zero rejections. The
[receipt](round2-preprocessing-v2/preservation.json) and local executed notebook
`runs/round2-preprocessing-v2/03_preprocessing_experiments.executed.ipynb` preserve
that check. The deep cached replay took 63.84 minutes; saved-result reuse still
verifies source/export bytes and is not a lightweight report-viewing path. Use
the compact reports/plots for inspection without that full validation cost.

[The handoff decision](round2-preprocessing-v2/handoff.json) retains frozen v2's
original policy ID and releases **no new v3 config**. Issue #30 remains open for a
justified policy-selection outcome; #33 must not treat the rejected denoise
candidate as a frozen v3 prerequisite. Any further search needs a new declared
experiment and a confirmation strategy that does not screen runner-ups on this
already-used confirmation sample. No model training, S20 expansion or final-test
access is part of these results. This attempt does not prove which preprocessing
would be best after fine-tuning.
