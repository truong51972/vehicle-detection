# Historical YOLO11n execution snapshot (superseded protocol)

This is the PR #3 snapshot, preserved for traceability. The current study follows
[parent #4](https://github.com/truong51972/vehicle-detection/issues/4), starting
with [issue #5's YOLO26n data protocol and pretrained baseline](ISSUE5_RESULTS.md).
The local YOLO11 notebook may contain later progress; the numbers below describe
only the original recorded observation. Original paths/identities are historical.

The full raw YOLO11n run is active on the available BDD100K data and GPU. The
snapshot below proves actual full-data training and native evaluation; it is
**one epoch of a 100-epoch maximum run, not a completed baseline score**.
Preprocessing experiments remain deferred until this raw baseline completes.
Issue #1 stays open.

## Actual native epoch-1 result

All **55,890 training images** and **13,973 train-derived tuning images** were
retained by native loaders, with `fraction=1.0`. Both native scans reported **zero
corrupt entries** (432 train / 131 tune background images retained). The loader
count/fraction assertions passed before gradients began.

Native epoch-1 tune metrics, read directly from the saved Ultralytics CSV:

- mAP50:95: **0.15715**; mAP50: **0.26684**.
- Precision: **0.53480**; recall: **0.26859**.
- Train box/cls/DFL losses: **1.39628 / 0.95838 / 1.03514**.
- Tune box/cls/DFL losses: **1.34194 / 0.85272 / 1.00280**.
- Native elapsed training plus tuning time: **278.771 s** for epoch 1.

Training advanced to epoch 2 with real loss/mAP/time curves saved directly in
notebook 2. Subsequent epochs continue locally. This early warm-up observation
must not be compared with Phase 1 pilot scores or treated as a preprocessing gain.
Final checkpoint selection and completed-baseline evaluation remain pending.

[Machine-readable native row, configuration, hardware and identities](phase2_raw_epoch1.json)
were captured from the actual CSV and immutable periodic checkpoint. The periodic
`weights/epoch0.pt` stores native zero-based epoch 0 and optimizer state, SHA256:
`be2c9e596396aa59f372603176dffcb1125cba28618b70943079d9ad2943bdeb`.
This is the epoch-1 progress checkpoint, not a claimed final/best model. Best/last
continue changing during training; periodic epoch snapshots are retained.

## Training protocol and environment

- Base: merged Phase 1 `d069727` (PR #2); the same two notebooks and toolkit.
- YOLO11n, six vehicle classes, 100 epochs maximum, patience 20, batch 32,
  imgsz 640, seed 2026, CUDA 0, workers 8, checkpoint every epoch, native resume.
- Native `Metric.fitness()` in Ultralytics 8.4.198 uses `[0, 0, 0, 1]` for
  precision/recall/mAP50/mAP50:95: best/early stopping select **tune mAP50:95**.
- Initial weights SHA256:
  `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`.
- Python 3.11.14, torch 2.15.0+cu132, CUDA runtime 13.2, Ultralytics 8.4.198;
  RTX 5060 Ti, 16,311 MiB reported by NVIDIA-SMI. NVIDIA driver 620.67.01 /
  Windows KMD 621.01; Intel i5-14600KF, 20 logical CPUs; WSL, mounted D: data/artifacts.
- Native warnings measured mounted-file reads around **15–17 MB/s** on sampled
  small JPEGs. This is a storage limitation, not a GPU inference measurement.

## Verified full preparation

The annotation source contains **69,863 train / 10,000 val frames**, rather than a
nominal 70k-train count. The run uses all available eligible annotated train data.

- Train-only preflight: **69,863 images / 765,202 vehicle boxes**, zero header,
  geometry or missing-train-image errors.
- Deterministic 80/20 train/tune, no ID/video-group/byte-content-hash overlap.
- Official BDD100K/Scalabel→COCO→native YOLO conversion and exhaustive selected-box
  roundtrips passed at **0.01 pixel**, including empty negative labels.
- All **69,863 raw JPEG end markers** passed (`jpeg_integrity.json`), preventing
  silent native JPEG re-encoding. The exporter rejects such invalid raw JPEGs.
- Full raw export, deep hash/decode checks, complete export reuse and final deep
  verification passed. Notebook 1 completed **2026-11-02 18:47 UTC**, about **2 h 15 m**
  including full discovery/conversion/export/verification/reuse on mounted D:.
- Raw source links add **0 copied-image MiB**, with **3,863.045 MiB** logical bytes.
  Original materialization recorded **217.007 s** feature work and **1,281.840 s**
  summed read/decode/feature/link/label/output preparation. Whole notebook wall time
  also includes hashing, traversal, conversion, deep verification and reuse.
  These setup costs are not GPU inference latency or portable latency benchmarks.

Selection identity:
`d8f6086b184e1460ee375d80246b6c9ca4127339777767bcf32803af30d54d27`.

**9,904 final holdout IDs** remain reserved after excluding both Phase 1 exposure
lists (96 images). Their pixels were not decoded, hashed, exported, previewed or
evaluated. Official-val annotation discovery is used for reservation only; EDA,
recipe previews, training and tuning use official train.

## Execution and recovery

The existing notebooks execute sequentially with `nbclient`, `timeout=None`.
The local runner is retained at `runs/phase2-full/execute_notebooks.py`; execution
log: `runs/phase2-full/execution.log`. Notebook outputs are periodically saved;
only one writer/trainer may use this root. The local training process continues
updating notebook 2 after the PR snapshot; machine/process availability is required.

Native files live under
`runs/phase2-full/experiments/yolo11n-seed2026-v1/raw_original/`: `study.json`,
`args.yaml`, `dataset_counts.json`, `results.csv`, best/last/per-epoch checkpoints.
Use notebook 2 for native resume; mismatched settings, data/output paths or
checkpoint/CSV epochs fail before overwriting history. Preserve artifacts and
recover a matching checkpoint/history pair if an interruption leaves a mismatch.
Intentional mid-training interruption/resumption has **not** been exercised.

The committed notebook snapshot contains real preparation evidence and epoch-1
training curves. Unexecuted post-training evaluation cells contain no results;
native per-class AP, confusion/PR, exported-file timing and diagnostic images are
scheduled after raw training completes. No mock or pilot result substitutes for
these pending measurements. The preparation notebook's repeated terminal progress
is condensed; figures/tables and actual results remain intact.

## Checks and remaining work

```sh
.venv/bin/python -m pytest -q        # 18 passed; native converter/raw JPEG/holdout guards
.venv/bin/ruff check .              # passed
.venv/bin/ruff format --check .      # passed
uv build                           # source and wheel built
git diff --check                   # passed
```

The initial draft commit passed
[GitHub Actions run 19](https://github.com/truong51972/vehicle-detection/actions/runs/41486183058).
Helper/CI checks require no BDD/model download or GPU and do not establish training
completion.

Remaining Phase 2 work:

- Complete full raw fine-tuning (or documented native early stopping), preserve
  final best/last hashes and run the scheduled full tune evaluation.
- Calibrate Gamma, CLAHE, Denoise, Sharpen and Adaptive on train/tune only; benchmark
  against JPEG95-only control with identical IDs, GT, frozen raw checkpoint and
  consistent native settings. Defaults remain uncalibrated; no treatment score exists.
- Measure representative mean/p50/p95 costs with warm-up and synchronized CUDA
  timing. Native average exported-file speed is not online end-to-end or batch-1 latency.
- Freeze the final model/recipes/protocol, then prepare/evaluate untouched holdout
  with full leakage checks. Complete scoped robustness, repeated seeds/uncertainty
  and representative failure analysis before closing Issue #1.
