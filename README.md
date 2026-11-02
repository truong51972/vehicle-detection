# Vehicle detection under adverse visual conditions

A reproducible BDD100K study of whether pixel-quality-aware preprocessing improves
vehicle detection enough to justify its latency. Implements [issue #1](https://github.com/truong51972/vehicle-detection/issues/1).

The experiment trains one raw-image YOLO baseline, then freezes its checkpoint and
compares raw, gamma, CLAHE, denoise, sharpen and an adaptive policy on paired images.
There is no web UI or hardware simulation.

## Install and check

Python 3.11+ and [uv](https://docs.astral.sh/uv/) are required. Linux/WSL is recommended;
the image export uses absolute symlinks (native Windows needs symlink privileges).
Minimal Linux containers also need OpenCV's system libraries (`libgl1`, `libglib2.0-0`
on Debian/Ubuntu). Core and detector share one OpenCV distribution.

```sh
uv sync --locked
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run vd --help
```

The default install supports survey, preparation and all synthetic tests. Tests
create tiny BDD fixtures in pytest temporary directories, use a fake detector at
the inference boundary, and **never download data or model weights**. Passing tests
is not evidence of YOLO accuracy or GPU performance.

Training/inference additionally require the optional Ultralytics/PyTorch stack:

```sh
uv sync --locked --extra detector
```

This extra is large; choose a PyTorch build compatible with your machine if the
resolved default does not support your GPU. Only load trusted PyTorch checkpoints.
Training may download the configured initial checkpoint; survey and prepare do not.

## Dataset and validation

Place the BDD100K detection images and labels under `data/bdd100k/` (gitignored).
Common nested Kaggle layouts are discovered recursively, including:

```text
data/bdd100k/
  bdd100k/bdd100k/images/100k/train/*.jpg
  bdd100k/bdd100k/images/100k/val/*.jpg
  bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json
  bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json
```

Flat split JSON arrays, Scalabel `{frames: [...]}` files, split directories of
single-frame JSON, and old single-image `{frames: [{objects: [...]}]}` records are
supported. Paths/filenames must identify official `train` or `val`. Multi-frame
video tracking exports are not detection-image annotations and are rejected.
The 1.45 GB 2018 training JSON is streamed with ijson rather than loaded whole.

Images alone, segmentation masks, lane polygons, and image-level tags are
insufficient. Both official splits must contain `box2d` annotations. The survey
warns about partial annotation/image coverage: a download with roughly 2k usable
annotations is not the complete BDD100K detection dataset. The canonical
`images/100k/` copy takes precedence over overlapping `10k`/segmentation image
copies; this choice is counted in the audit. Otherwise byte-identical image copies
are deduplicated and counted. Duplicate IDs, unresolved ambiguous image matches
and conflicting annotation sets fail with guidance; isolate one
version of the detection labels if multiple copies are installed.

Category IDs are fixed: `0 car`, `1 truck`, `2 bus`, `3 train`, `4 motorcycle`,
`5 bicycle`. BDD 2018 `motor` maps to motorcycle and `bike` to bicycle. Person,
rider, traffic signs/lights, other categories and polygons are ignored and counted.
Occluded/truncated boxes remain included. Empty vehicle labels remain valid
negative examples. Malformed, nonfinite or reversed vehicle boxes exclude the
whole image during discovery, avoiding a false negative created by dropping only
its annotation. Missing files are excluded and counted. Out-of-bounds boxes and
unreadable selected images fail preparation; boxes are **never clipped or rescaled**.
Repair the source or explicitly curate an annotated subset, then prepare a new run.

```sh
uv run vd survey --data data/bdd100k --out runs/survey
uv run vd prepare --data data/bdd100k --preset pilot --out runs/prepared
```

Survey artifacts: `survey.json` (split/class box and image counts, metadata,
filtering audit, annotation hashes, sampled health/quality and sample IDs),
`boxes.csv` (pixel-space size/aspect distributions), `survey.png`. Health and
bounds checks in survey are **sampled**; prepare validates every selected image.

Prepare artifacts: `manifest.json`, `dataset.yaml`, `images/{train,tune,holdout}`
symlinks and corresponding YOLO label files. The manifest saves pixel boxes,
source SHA-256 hashes, seed, sampling config and IDs. Existing exports are never
overwritten. Source data must remain at its original absolute path.

## Experimental controls

Official train is divided into train and tune (default 80/20); video groups stay
together when `videoName` is available, otherwise one image is one group. Official
val supplies the final holdout. Pilot limits are applied after partitioning: up to
160 train, 40 tune and 80 holdout images, deterministically sampled with seed 42.
Full uses all eligible data. There is no class/condition balance guarantee in the
pilot; inspect support before drawing conclusions.

ID, video-group and path overlap are checked; byte-identical images crossing the
selected partitions are rejected using SHA-256. This does not detect visual
near-duplicates without shared group metadata. Every selected image and its box
bounds are validated before export. The training YAML exposes train and tune;
it does not expose holdout. Exports and source hashes are rechecked before training.

The raw baseline uses standard Ultralytics training/augmentation, with no custom
preprocessing treatment. Only the tune partition selects `best.pt`. Training saves
best/last checkpoint hashes, initial weight hash when available, `training.json`,
Ultralytics arguments/results CSV, best tuning metrics, last-epoch metrics and
software provenance. Benchmark loads one frozen checkpoint with the exact six-class
mapping and reuses it across the complete comparison matrix.

## One-command pilot and full run

```sh
uv run --extra detector vd run --preset pilot --data data/bdd100k --device 0 --out runs/pilot
# CPU is supported, but YOLO training may be slow:
uv run --extra detector vd run --preset pilot --device cpu --out runs/pilot-cpu
```

This performs survey → prepare → raw training → holdout benchmark → Markdown report.
The pilot trains `yolo11n.pt` for 3 epochs at 640 pixels with batch 8 and clean
images only. It is a feasibility check, **not a research-grade experiment**.

```sh
uv run --extra detector vd run --preset full --device 0 --out runs/full
```

Full trains for 50 epochs, uses all eligible data and evaluates clean, dark,
blurred, noisy and hazy paired conditions. Budget disk/time on the owner's machine;
no real detector results are shipped or implied by CI. Override `--model`,
`--epochs`, `--batch`, `--imgsz` (multiple of 32), `--seed` and `--device` as needed.
Record and lock your choices before examining final holdout results.

Stages can also be run independently:

```sh
uv run --extra detector vd train --prepared runs/prepared --device 0 --epochs 3 --out runs/train
uv run --extra detector vd benchmark --prepared runs/prepared --weights runs/train/weights/best.pt --device 0 --split tune --out runs/tune-baseline
```

**Calibrate policy parameters or compare candidates on tune only.** Freeze the
chosen YAML/configuration and detector, then run the final holdout once:

```sh
uv run --extra detector vd benchmark --prepared runs/prepared --weights runs/train/weights/best.pt --device 0 --config policy.yaml --split holdout --out runs/holdout
```

The default policy is a transparent, **uncalibrated heuristic**, not an optimized
or learned result. No automatic policy selection occurs. Do not retune after
viewing the holdout report. Comparing architectures is a separate experimental axis.

## Configurable policy

Presets live in `src/vehicle_detection/presets/`. `--config` recursively merges
YAML overrides into the preset; `operators` replaces the complete ordered list.
Unknown keys, duplicate operators/treatments, unsupported conditions and invalid
parameters are rejected before expensive work starts. For example, `policy.yaml`:

```yaml
operators:
  - name: denoise
    enabled: true
    params: {strength: 5.0}
  - name: gamma
    enabled: true
    params: {gamma: 0.65}
  - name: clahe
    enabled: true
    params: {clip_limit: 2.0, grid_size: 8}
  - name: sharpen
    enabled: false
    params: {amount: 0.5, sigma: 1.0}
policy:
  max_filters: 2
  brightness_below: 0.28
  contrast_below: 0.16
  noise_above: 0.035
  sharpness_min: 10.0
  sharpness_below: 100.0
benchmark:
  treatments: [raw, gamma, clahe, denoise, adaptive]
```

The operator registry is detector-independent and contains geometry-preserving
BGR uint8 functions with explicit parameter bounds. Gamma uses an exponent below
1 to brighten. CLAHE modifies only Lab luminance. Denoise uses colored non-local
means. Sharpen uses a bounded unsharp mask. Input arrays are validated and not
mutated. No geometric augmentation or dehazing operator is added by this project.

Quality features use a maximum 512-pixel image side: normalized mean grayscale
brightness, normalized grayscale standard deviation (contrast), Laplacian variance
(sharpness), and median absolute residual against a 3×3 median filter divided by
0.6745×255 (noise indicator). Black/white clipping fractions are diagnostic. These
are approximate indicators and can respond to scene texture, not ground-truth
quality labels. Calibrate on the intended tuning data at these fixed feature units.

Adaptive routing evaluates low brightness → gamma, low contrast → CLAHE, high
noise → denoise and moderate blur without high noise → sharpen; execution follows
the YAML order and stops at `max_filters` (0–2). If nothing qualifies, it skips.
Very low sharpness is not treated as recoverable blur. The same original input
scores choose the entire sequence; metadata never enters routing. Each image logs
its scores, exact sequence/parameters and preprocessing/inference timings.

## Benchmark outputs and timing

Each treatment sees the same IDs, unchanged boxes, stress realization, checkpoint,
resolution and confidence/NMS settings. Stress parameters and seed are saved:
clean original pixels; multiplicative darkening; Gaussian blur; seeded Gaussian
noise; white-airlight haze blend. These are synthetic robustness probes, **not BDD
weather labels or physical models of all adverse conditions**. Treatment order is
deterministically interleaved per image to reduce systematic timing bias.

`runs/.../benchmark/` contains:

- `experiment.json`: manifest/weights/config/source/software/hardware identities,
  training provenance, sample IDs, metric settings and split.
- `records/<condition>/<treatment>/<id>.json`: atomic checksummed predictions,
  paired GT, quality, routing and timings. These are the resumable computation cache.
- `decisions.jsonl`: per-image policy/quality/timing log (no prediction arrays).
- `results.csv` and `results.json`: overall mAP50, mAP50:95, precision, recall,
  per-class AP/support, weather/time/scene strata, condition-matched raw deltas,
  mean/p50/p95 latency, throughput and overhead. Unsupported-class AP is null.
- `comparison.png`, `samples/` and `report.md`: accuracy/cost plots, paired
  before/after panels, low-delta failure cases and a methods/limitations report.
- `report_timing.json`: wall-clock report generation time, including record reads,
  matching, aggregate/stratum metrics, failure diagnostics, samples, plots and writes.

Reporting computes box matches once per image/condition/treatment and reuses them
for aggregate AP, metadata strata and per-image failure diagnostics. Matches are
cached only in memory for raw and the current treatment. Aggregate AP still ranks
all predictions in its population; it is never averaged from per-image AP.

Metrics use classwise confidence-ordered greedy one-to-one IoU matching at
0.50:0.05:0.95, a precision envelope and 101-point mean interpolated AP. There is no
COCO area/crowd weighting; this is **not the official COCO/BDD leaderboard evaluator**.
Macro averages exclude classes without ground truth. Precision/recall are computed
at IoU 0.50 and confidence 0.25 by default; AP retains predictions down to 0.001.
Metadata strata include support and a small-sample flag (default fewer than 30
images). Always inspect per-class support and losses as well as gains.

Ultralytics may clip detections wholly in square-inference letterbox padding to
zero source-image area. These predictions are discarded consistently across
treatments, with per-image `discarded_degenerate_predictions` counts in the records
and decisions log. Nonfinite values, invalid classes, reversed boxes and
out-of-bounds predictions still fail validation. Ground-truth boxes are unchanged.

Latency uses batch 1, FP32 and fixed square resolution; warm-up is excluded and
CUDA/MPS work is synchronized. Preprocessing includes adaptive features, routing
and filters. Inference includes detector resize/normalization, forward, NMS and
CPU result transfer. Measured end-to-end is preprocessing plus inference. It
excludes disk read/decode, stress construction, diagnostic-only quality extraction,
metric calculation and artifact writes. Throughput is 1000/mean end-to-end ms;
there is no pipeline overlap. Timings are host wall-clock measurements, not kernel
profiling. Compare mAP50:95 against the acceptable latency budget you set in advance;
the code does not select a policy using holdout scores.

Sharpening cannot reconstruct information destroyed by real blur. Retain
no-improvement and harm cases. Synthetic stress gains need not transfer to natural
adverse scenes. Optional bootstrap confidence intervals, automated threshold search
and repeated seeds are not implemented; for a full study run at least two separate
training seeds after freezing methodology, keeping all treatments within each run
on its identical checkpoint/manifest.

## Resume and provenance

```sh
uv run --extra detector vd benchmark --prepared runs/prepared --weights runs/train/weights/best.pt --device 0 --config policy.yaml --out runs/holdout --resume
# Also resumes completed preparation/training and the benchmark in a one-command run:
uv run --extra detector vd run --preset pilot --device 0 --out runs/pilot --resume
```

Resume requires identical manifest, source images, checkpoint, configuration,
metric implementation, package source hash, software versions and hardware.
Completed shards are validated and reused; partially written `.tmp` shards are
recomputed. A changed identity fails instead of mixing results. For different
configs, versions or weights use a new output directory. No preprocessed image
cache survives across experiments.

Only one process may write an experiment at a time. A hard kill may leave `.lock`;
confirm the old process has stopped before removing that file. Corrupted completed
shards fail; after diagnosing the corruption, remove only the damaged shard and
resume to recompute it. A killed training phase is not automatically resumed:
keep its diagnostic artifacts and start training in a new output directory.
Resumed timings may span sessions and machine load changes, so final latency claims
need a fresh uninterrupted benchmark. Keep source data stable during a run.

## Validation boundary

Synthetic unit/integration tests cover conversion, bounds, streaming schemas,
partition/leakage checks, source/manifest integrity, operator invariants,
policy skip/routing, paired stress, AP sanity cases, train/tune isolation,
interrupted benchmark resume, identity rejection, reports and CLI errors.
The detector boundary is mocked; real YOLO training, full holdout evaluation,
GPU timing, threshold calibration and multi-seed conclusions require owner-side
runs. None are claimed by these tests. Generated data, models and runs stay out of Git.

References: [BDD100K/Scalabel label schema](https://github.com/ucbdrive/bdd100k/blob/master/doc/format.md),
[Ultralytics training API](https://docs.ultralytics.com/modes/train/),
[Ultralytics prediction API](https://docs.ultralytics.com/modes/predict/).

## Real-data fine-tune + evaluation (required before merge)

Synthetic CI exercises the pipeline with a fake detector to catch deterministic
logic regressions; it **does not validate actual YOLO fine-tuning, preprocessing
compatibility, or detection accuracy**. With local BDD100K images and detection
annotations, the following **unmocked** end-to-end pilot is the integration test:

```sh
uv sync --locked --extra detector
uv run --extra detector python scripts/real_pilot.py \
  --data data/bdd100k --device 0 --out runs/real-pilot
# Without CUDA: replace --device 0 with --device cpu (may be slow).
```

This executes actual Ultralytics fine-tuning on the raw train partition, checkpoint
selection on train-derived tune, and inference/evaluation with frozen `best.pt`
on paired official-validation holdout images for raw, fixed and adaptive filters.
It verifies nonempty best/last weights, training epochs, provenance/checkpoint
hashes, original split separation, paired prediction shards, output completeness,
and finite quality metrics. A success attestation is written to
`runs/real-pilot/real_validation.json`. All outputs remain gitignored.

To verify an already completed pilot without training again:

```sh
uv run --extra detector python scripts/real_pilot.py \
  --out runs/real-pilot --verify-only
```

**PR acceptance gate:** report the actual device, model, training epochs, train/tune/
holdout sizes, mAP50/mAP50:95 for raw and adaptive, delta versus raw, and average
end-to-end latency; attach the command/output and relevant redacted artifacts to
the PR. A run using fake predictions, a skipped integration check, or successful
survey/prepare alone **does not meet this gate**. Re-run with a fresh output path
if any inference/evaluation settings change; never tune using holdout results.

The smoke pilot is a feasibility check only. For research conclusions, calibrate
on tune and run the full frozen-checkpoint paired holdout experiment. Real data
and downloaded checkpoints are never checked into the repository.
