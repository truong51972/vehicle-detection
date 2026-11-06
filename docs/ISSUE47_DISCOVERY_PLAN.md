# Issue #47 — discovery after the negative Gamma/Sharpen pilot

This is a separate research-first study after PR #48, not a replacement of its
evidence or a completed v3 policy. Issue #47 stays open. The next decision is
**one controlled resolution diagnostic**, before writing another enhancer.
No completed v0/v1/v2 or PR #48 artifact is edited or rerun in place.

## Diagnosis before mechanism selection

The saved PR #48 native matches, original annotation metadata and committed
pixel features permit class-level AP, TP/FP/FN and size-associated image cohorts.
They do not identify matched GT-box indices or saved prediction geometry. Thus
individual missed-box size and FP location/size cannot be reconstructed from
this cache. Do not call cohort AP “AP-small”, assign misses to particular boxes,
or attribute a miss causally to darkness/resolution. No new matcher is introduced.

An annotation/cache audit finds 31 motorcycle GT with only one confidence-.25 /
IoU-.50 TP, and 86 bicycle GT with 14 TP. Of images containing motorcycle/bicycle,
28/29 and 55/59 are bright under the original <0.12 darkness definition; 28/29
and 54/59 were skipped by the old route. Median sqrt(box area) at 640 is 22.89
pixels for motorcycles and 26.38 for bicycles; 9/31 and 13/86 boxes fall below
16 pixels. Some are small, but **most are not in that smallest bucket**. These
observations motivate testing resolution, not asserting it explains every miss.

Derive all/brightness/contrast/noise/low-sharpness/old-route/scene-attribute cohorts
from saved features and original annotation metadata; retain every original GT
in each image. Use historical bins only, not a fitted threshold grid. Include
absent-GT-class FP counts without imputing absent-class AP as zero. Scene labels,
GT and class support are analysis metadata, never runtime routing inputs.

## Literature and ranked hypotheses

Read primary abstracts/official implementations, accessed 2026-11-06. External
results motivate hypotheses; none supplies a YOLO26n/BDD100K effect size here.
The ranked, reviewer-readable table is in the independent research notebook.

1. **Resolution diagnostic — selected now.** [SAHI](https://arxiv.org/abs/2202.06934)
   studies retaining small-object detail through slicing. Use native `imgsz`
   640 versus **one fixed larger size, 960**, to test sensitivity without a new
   slicing dependency, changed suppression, crops or GT transformations. This
   does not reproduce SAHI and is not an offline preprocessing policy.
2. **Task-aligned photometric training — strongest subsequent method hypothesis.**
   [Image Enhancement Guided Object Detection](https://pubmed.ncbi.nlm.nih.gov/37220059/)
   explains why separately optimized visual enhancement need not aid detection;
   its parallel training branch is not implemented here. [Exposure-Aware Training](https://www.mdpi.com/2313-433X/12/6/245)
   studies moderate exposure/noise degradation during training, with stronger
   degradation potentially harming semantics. It is evidence from other models/
   datasets, not validation of a borrowed noise recipe. [Ultralytics HSV](https://docs.ultralytics.com/guides/yolo-data-augmentation)
   offers a native brightness control, but it is **not** a calibrated camera-noise
   model. A future diagnostic would compare two independently initialized native
   short training runs, using training-pool data and consumed calibration for
   analysis; it needs its own committed recipe/budget before execution.
3. **Highlight-safe, luminance-only constrained local contrast — defer.**
   [RUAS](https://arxiv.org/abs/2012.05609) learns an unrolled Retinex illumination
   model; it is not equivalent to the failed hand-written local-shadow gain.
   A bounded OpenCV luminance-only correction with a global-correction ablation
   could distinguish local structure benefit from mere brightening, but current
   errors do not yet identify correctable shadowed vehicle structure. Retrying
   an unrestricted CLAHE/gamma grid is not justified. The supplied AAAI article
   link was access-blocked and its title/claims were not verified; it is not used
   as evidence for this selection.
4. **One learned comparator, SCI — defer; no model downloaded.**
   [SCI's paper](https://arxiv.org/abs/2204.10137) describes single-block inference
   and low-light face/segmentation applications, not this five-class benchmark.
   Its [official CVPR README](https://raw.githubusercontent.com/vis-opt-group/SCI/master/CVPR/README.md)
   provides easy/medium/difficult weight choices and explicitly distinguishes
   those weights from its paper's reported models. The inspected official README/
   repository listing did not establish a code/checkpoint license. Before any
   later single-checkpoint comparison, resolve license, select/hash one checkpoint
   without outcome shopping, verify geometry/color/range and directly measure CPU
   mean/p95. This round does not implement or rank multiple learned enhancers.

## Frozen diagnostic population and controls

Commit this plan and implementation **before diagnostic inference**. Use exactly
the **1,027 consumed PR #48 images / 649 original source groups**, sorted-ID hash
`d925d907e4ea2da0ff1301a418d7a5a194ff0f4c114f73aba2adb0b0e8c45084`.
This is discovery on known exploration data, never independent evidence. Do not
select images by new outcomes, append groups, consume remaining exploration or
read reserved confirmation/development/final pixels. Original annotations remain
metadata. The reserved confirmation plan stays unread. No original image read,
new filter/export encoding or GT-based runtime decision is needed.

Verify the old export receipt and copy its Skip JPEG95 images and exact labels
into **new owned files** under `runs/issue47-discovery/resolution/data`. Native
label/image caches and evaluation outputs must also stay under that new root,
so prior runs cannot be overwritten. New compact evidence belongs exclusively
under `docs/issue47-discovery`. Existing/incomplete attempts block re-execution.

Exactly two native five-class pretrained evaluations, **2,054 image passes**:
640 and 960 with the same pinned YOLO26n checkpoint, FP32, batch 32, workers 0,
seed 2026, rect=True, NMS-free, confidence .001, max_det 300, original paired GT.
Use PR #48's unchanged native validator/settings; change only `imgsz`. Verify
loader IDs, native GT and agreement of native versus saved-match AP. Record
whether the fresh 640 control reproduces the historical native metrics within
absolute 1e-6; if not, retain measurements but invalidate the paired interpretation
until the discrepancy is understood. No repeated rescue run is authorized.

## Metrics, uncertainty, cost and falsification

Report native five-class mAP50:95/mAP50, per-class and overlapping image-cohort AP,
class TP/FP/FN at confidence .25 / IoU .50 and example IDs. GT-size distributions
stay separate from cohort AP. Require >=5 original source groups for a supported
class effect; sparse cohort/class effects stay inconclusive rather than blocking
every route mechanically or changing the original PR #48 gate retroactively.

Use 100 paired seed-2026 **original source-group** bootstrap draws, fixed five-class
primary macro, no missing-class imputation/redraw. Report per-class intervals
and missing-class draws. These are exploratory conditional intervals with no
multiplicity adjustment or claim of independence.

For an actionable resolution-sensitive signal require: a reproduced 640 control;
primary gain >=0.2 AP percentage points; motorcycle/bicycle AP50:95 mean gain
>=0.5 points with both classes represented by >=5 source groups; positive primary
interval lower bound; no supported primary-class loss >2 points or dark/bright
class loss >1 point; and native mean inference time at 960 <=3x 640. Preserve
all failed gates. A pass recommends a later independent diagnostic, never a v3
preprocessing policy or a full training run. A failure rejects this diagnostic's
usefulness, not all possible resolution/training mechanisms.

Record native preprocessing/inference/postprocessing mean milliseconds per image,
phase wall time, and synchronized validation-batch mean/p95 in the same process.
Batch p95 is **not per-image p95**; sequential 640/960 runs include different
setup/shape effects, so these timings are exploratory, not a deployment SLA.
There is no additional offline CPU filter; the exact same already encoded images
are used. Do not represent GPU resize/validation timings as a new matched CPU
preprocessing benchmark, or compare them to earlier CPU timing runs.

Stop after these two evaluations regardless of the result. No extra size, SAHI,
new enhancer, tuning, training or confirmation is authorized by this plan.
Use the findings to prioritize the next separately declared method study. Full
S20 preparation/training still requires a defensible frozen policy and independent
confirmation on appropriate untouched groups, with the original IDs/GT/settings.
