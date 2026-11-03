# Issue #11: calibration-only synthetic robustness research

**Statistical review update:** this is the original v1 descriptive study record.
Its bootstrap intervals and shortlist gate are superseded by the
[fixed-estimand v2 review](ISSUE11_BOOTSTRAP_REVIEW.md). Native six-class point
estimates, negative findings and no-v3 conclusion are unchanged. The original
executed notebook is preserved in the evidence archive; current notebook 02b
contains the corrected analysis from the same saved detections.

[Issue #11](https://github.com/truong51972/vehicle-detection/issues/11) extends
the Raw/Adaptive study with independent corruption/operator experiments before
fine-tuning or final evaluation. The self-contained
[executed research notebook](../notebooks/archive/phase2/02b_synthetic_robustness.ipynb)
contains configuration, reasoning, actual native detector metrics, uncertainty,
quality diagnostics and selected real predictions. Large datasets, checkpoints
and full per-image statistics stay under gitignored `runs/issue11-v1/`.

## Research design and sources

The corruption taxonomy is informed by
[Hendrycks & Dietterich (ICLR 2019)](https://arxiv.org/abs/1903.12261), a
classification benchmark, not evidence that these BDD corruptions reproduce real
cameras. [Foi et al. (2008)](https://webpages.tuni.fi/foi/papers/Foi-PoissonianGaussianClippedRaw-2007-IEEE_TIP.pdf)
motivates signal-dependent Poisson shot noise plus Gaussian read noise for sensor
raw data. Here inverse gamma 2.2 is applied to processed JPEG pixels as an
approximation; sensor calibration, demosaicing and camera ISP effects are absent.

The native-pixel severity parameters are frozen in
[study.json](issue11-v1/study.json):

- Shot/read noise: photon scales 800/200/50 and read sigmas 0.003/0.008/0.02.
- Motion: 5/11/21-pixel normalized line PSFs, with one deterministic per-ID
  orientation held constant across severity levels.
- Defocus: disk PSFs with radii 1/2/4 pixels.
- Underexposure: pseudo-linear exposure factors 0.7/0.45/0.25; overexposure:
  1.3/1.7/2.3, including irreversible clipping.
- Low contrast/veil: uniform white-airlight blends 0.1/0.25/0.4. This is a
  controlled contrast stress, not a depth-aware atmospheric fog simulator.

Names mild/moderate/severe describe levels within this protocol, not calibrated
camera-physics severity. Images, PSNR and feature distributions show their actual
effect. Some mild changes can improve AP; the clean reference is not a guaranteed
maximum AP. Severe blur, clipping and noise can make original labels visually
ambiguous. The cohort is retained to disclose that limit rather than selecting
images after detector outcomes.

Seven independently screened restorations cover Gamma, CLAHE, NLM, bilateral
filtering, unsharp, darkening Gamma and percentile luminance stretching. Bilateral
and percentile stretch go beyond the four old routes. Parameters are explicit in
the registry/config; they do not use BDD weather/time/scene labels at inference.
[OpenCV filtering](https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html)
and [NLM documentation](https://docs.opencv.org/4.x/d5/d69/tutorial_py_non_local_means.html)
motivate the candidate implementations, not detector-benefit claims.

Each family gets two plausible independent operators at moderate severity, plus
clean safety controls. Frozen Adaptive v2 is a separately measured baseline.
Rain streaks/glare/JPEG perturbations are deferred: the first two lack a justified
scene/optics model here; JPEG recompression is already controlled. Learned
deblurring/dehazing and blind deconvolution would add dependencies and PSF/depth
assumptions unsupported by this pilot. No extra training branches are created.

## Population, parity and native evaluation

Only the frozen issue #5 **calibration** role is used. A deterministic search over
64 whole-group permutations selects **256 images / 158 groups**, minimizing
multi-label/condition drift. No IDs change roles. All variants retain the source
ID/group and byte-identical native-converted boxes, verified at 0.01 pixel.

Support is **254 car / 75 truck / 40 bus / 1 train / 10 motorcycle / 12 bicycle
images**, with one negative frame and 2,830 boxes. The pilot includes 137 daytime,
101 night and 18 dawn/dusk frames; 131 clear, 33 overcast, 21 rainy, 20 snowy,
18 partly cloudy, 32 undefined-weather and one foggy frame. A single foggy frame
cannot validate a natural-fog restoration claim. Train-class AP and many other
rare strata remain extremely uncertain.

The fixed budget is **45 datasets**, each with the same 256 source IDs:
clean Raw; 18 corrupted Raw conditions; 12 independent moderate-severity screens;
eight clean-input operator controls; six moderate-severity frozen-v2 controls.
This is a screening matrix, not a full factorial or independent replication.

Every branch performs source decode → corruption → optional restoration → one
JPEG95 encode. Restored branches do not receive an extra intermediate JPEG encode.
Original images remain unchanged. Pixel coordinates, shape, dtype and GT are
invariant. Source/label/group/seed/parameter/code/library hashes bind exports;
dataset/checkpoint/settings/hardware/adapter/plot/statistics hashes bind metric
caches. Changed identities fail rather than accepting stale results. Partial
experiments can rerun safely against their unchanged identities.

Pretrained YOLO26n is pinned to SHA256
`9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`.
The checked native COCO-80→six-class adapter is reused without modifying model
weights/names or AP matching. Settings retain imgsz 640, batch 32, FP32,
confidence 0.001, IoU 0.7, max_det 300, deterministic seed 2026 and `nms=False`.
Workers are explicitly zero for the bounded pilot. The full inference protocol
and software/hardware identities are saved. There is no training or development/
final-test pixel access.

## Historical v1 measured results (intervals superseded)

The nominal intervals below are retained for provenance only. Use analysis v2's
fixed-target estimates, intervals and gates for current inference decisions.

The notebook completed a real-data run and a complete cache replay on **2026-11-03**.
Clean calibration Raw **mAP50:95 is 0.157082**; this is a separate population from
the previously measured development baseline. The
[experimental matrix](issue11-v1/experimental_matrix.csv),
[per-class results](issue11-v1/per_class.csv) and
[supported image slices](issue11-v1/supported_image_slices.csv) retain all positive,
null and negative outcomes. [Slice support](issue11-v1/image_slice_support.csv)
flags inadequate class support. Size-presence image slices overlap and are not
COCO object-area AP_small/AP_medium/AP_large.

Only **CLAHE on moderate uniform veil** passes the exploratory gates:
corrupted Raw **0.147666 → 0.159629**, recovery **+0.011963 AP** (1.196 percentage
points), nominal paired interval **[+0.003985, +0.026767]**, restore p95
**8.90 ms**. Clean CLAHE is 0.160922, so this aggregate clean control does not
show harm. One foggy source frame and a controlled uniform veil do not establish
natural-fog efficacy or a supported blind router. This is a shortlist, not a v3 release.

Examples of null/adverse outcomes:

- Moderate shot/read noise: bilateral recovers **+0.003791 AP**, interval
  **[-0.006486, +0.009587]**, at p95 **6.86 ms**. NLM recovers **-0.000829 AP**,
  interval **[-0.009057, +0.005692]**, at p95 **777.22 ms**.
- Moderate defocus: unsharp scores **0.162522**, above clean Raw, but the matched
  defocused Raw is **0.166604**. Correct recovery is therefore **-0.004082 AP**.
  Reference MSE nevertheless improves: better photometric fidelity is not AP recovery.
- Underexposure CLAHE gives **+0.004687 AP**, with an interval crossing zero.
  Darkening Gamma on overexposure gives **-0.000865 AP**. These tested settings
  are not supported winners; other parameters have not been ruled out.
- Frozen v2 gives **-0.001192 AP** on moderate noise, with nominal interval
  **[-0.005902, -0.000208]**; it gives zero recovery on moderate veil. Retaining
  v2 means retaining the controlled baseline, not endorsing it as universally robust.

Mild defocus and exposure changes can improve this pilot's AP before restoration;
motion level 3 drops Raw AP to **0.043644**. The severe cases and full severity
grid are embedded for inspection. No levels were changed to manufacture damage
or recovery after viewing metrics. Only one strength per operator is screened,
and restored conditions are moderate severity; no exhaustive tuning or independent
replication is claimed.

## Estimator evidence and priority masking

[Estimator error rates](issue11-v1/estimator_errors.csv) use known synthetic
condition labels. Clean means no *added* corruption, not a pristine sensor image.
Thresholds are fitted in-sample and are diagnostic operating points, not a deployed
router. Texture and unknown original camera noise remain unresolved.

The separate archived v2 sample had 229/256 zero legacy-noise values. This new
group-aware clean pilot has **240/256** zeros. Even severe added shot/read noise
maxes at **0.023256**, below the fixed **0.035** floor: that rule misses **100%**
of added-noise examples at all three levels. Lowering the old threshold to its
clean 95th percentile still misses **98.05%** of mild noise.

The candidate full-resolution high-pass statistic detects **100%** of the three
known noise levels at its clean q95 threshold, with **5.08%** clean false positives
(by construction) and **15.63%** false positives on overexposure level 3. It is
promising for these synthetic references, not a validated sensor-noise estimator.

The clean-q05 Laplacian blur rule detects **19.53%** of moderate defocus and
**57.81%** of moderate motion. It falsely flags **37.5%** of underexposure level 3
as blur. Absolute sharpness confounds photometric changes with blur.
[The masking audit](issue11-v1/v2_priority_masking.json) finds eight archived-v2
samples satisfying the initial sharpening band; **all eight are masked by Gamma**.
Independent operator tests therefore provide evidence that route activation could
not supply, without forcing sharpening or denoising to become winners.

## Costs and reproducibility

The 45 image datasets occupy **2.023 GiB**; the preflight allowance was **4.488 GiB**
including scratch headroom. This is a sampled size heuristic, not a worst-case
bound. [Memory estimates](issue11-v1/memory_budget.json) separately account for
native uint8 BGR frames, the 675 MiB clean-reference cache, floating-point operator
workspace and native input batches. Process/GPU peak memory was not measured.
Native average model stage timing is recorded per case; it is not batch-1 p50/p95
or application end-to-end latency. Corruption and restoration CPU p50/p95 are separate.

The [summary](issue11-v1/summary.json) pins every export/result/statistics identity,
environment, cost and decision. The full run's caches were reused on replay;
the final cache check forbids model construction. All **31 prior v1/v2 artifacts
and helper files** remain byte-identical, checked against
[the preservation inventory](issue11-v1/prior_evidence_hashes.json).
No source images, weights, large exported roots or full per-image statistics are
committed. Selected notebook image embeds are intentional offline review evidence.

## Interpretation and downstream decision

Primary recovery is **mAP(restored) − mAP(corrupted Raw)** on the same condition,
IDs and checkpoint. Degradation is **mAP(clean Raw) − mAP(corrupted Raw)**.
Naturally distributed clean results are separate from synthetic robustness;
there is no blended real-world mAP. Quality metrics never substitute for AP.

Uncertainty uses 100 paired source-group bootstrap draws and native Ultralytics
AP reaggregation of saved per-image IoU matches. These are exploratory percentile
intervals, without multiplicity correction. Rare classes can disappear from a
draw, affecting the supported-class macro average. They are not confirmation of
an algorithm selected on this same pilot, nor uncertainty about corruption realism.

An operator shortlist requires at least 0.005 AP recovery, positive recovery CI
lower bound, clean-input loss at most 0.01 AP and restore p95 at most 25 ms.
Even a passing screen does not establish a pixel-only router or independent
confirmation. The declared outcome is **no v3 deployment from this screen**;
retain frozen v2 until a separately declared calibration-group confirmation
supports a router and its clean safety. V1/v2 evidence is preserved unchanged.

The [versioned v3 decision](issue11-v1/v3_decision.json) records passing/failing
support and the retained v2 identity. The
[final robustness specification](issue11-v1/final_robustness_spec.json) is an
unexecuted specification for #8: natural clean final test and separately labeled
synthetic conditions/severities, source-ID parity, paired Raw/Adaptive inputs,
frozen checkpoints/settings and group uncertainty. Final pixel/geometry/content
leakage checks and model/policy freeze remain mandatory release gates. No final
pixels are accessed by writing the specification.

## Checks and remaining work

Local tests cover corruption determinism, geometry/dtype/source safety, synthetic
noise/blur sensitivity, group atomicity, calibration-only membership, label parity,
version/config/cache isolation, tampered output/source rejection and paired native
AP/GT/bootstrap behavior. Notebook checks compile every cell and verify complete
execution and embedded figures; they do not substitute for the recorded GPU results.
**75 tests passed** locally; Ruff lint/format, source/wheel builds and whitespace
checks pass. Remote CI/review status is tracked on the PR separately.

Independent calibration-group confirmation, multiple corruption seeds, a validated
blind v3 router and actual #7/#8 training/final testing are **unexecuted**. The
current decision preserves v2 as a baseline and shortlists CLAHE/veil for further
confirmation. Broader restoration-strength searches and real sensor/depth/PSF
validation remain future work; this first screen does not prove their absence of benefit.
