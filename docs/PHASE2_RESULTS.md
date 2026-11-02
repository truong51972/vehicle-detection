# Phase 2 execution evidence

This is an in-progress execution record, not a completed full-study result.
No Phase 1 pilot metric or fabricated result is used as a Phase 2 baseline.

## Frozen baseline configuration

- Merged Phase 1 base: `d069727` (PR #2).
- Two existing notebooks; existing BDD100K/Scalabel/native YOLO conversion.
- Full available annotated official-train data, deterministic 80/20 train/tune,
  seed 2026; no subset limits or training fraction.
- Raw-original source links only. YOLO11n from the local `yolo11n.pt` checkpoint.
- 100 epochs maximum, patience 20, batch 32, imgsz 640, workers 8, device 0.
- Native per-epoch best/last checkpoints and `save_period=1`; native resume.
- Installed Ultralytics `Metric.fitness()` uses weights `[0, 0, 0, 1]` for
  precision/recall/mAP50/mAP50:95: native best/early stopping select tune mAP50:95.
- Final holdout IDs reserved with both Phase 1 exposure lists excluded;
  no final holdout pixel preparation or evaluation.

Hardware availability was verified with `/usr/lib/wsl/lib/nvidia-smi` and PyTorch:
NVIDIA GeForce RTX 5060 Ti, 16,311 MiB; Python 3.11.14, torch 2.15.0+cu132,
CUDA runtime 13.2. A live GPU process alone is not proof of completed training.

## Actual execution

The notebooks were launched sequentially using `nbclient`, timeout disabled for
full training. The local execution log is `runs/phase2-full/execution.log`.
Notebook output is periodically saved during this run; native checkpoint/CSV files
are the recovery source. Only one writer may use these notebook files/run root.

Train-only preflight completed on all **69,863 images / 765,202 vehicle boxes**:
all headers readable, all boxes valid, zero errors and no missing train images.
The supplied annotation files contain 69,863 train frames and 10,000 val frames;
this is a full run over the available annotated dataset, not a 70k-count claim.
Full decode/hash partition checks completed: **55,890 train / 13,973 tune**,
with no ID/video-group/content-hash overlap. **9,904 final holdout IDs** are reserved
and 96 previously exposed Phase 1 test IDs are excluded. No holdout image was decoded.

Selection identity:
`d8f6086b184e1460ee375d80246b6c9ca4127339777767bcf32803af30d54d27`.

Official conversion is running, including exhaustive selected-box roundtrips.
Raw export and GPU training follow automatically in the sequential notebook run.
No training metric has been measured yet.

Initial pretrained weights SHA256:
`0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`.

Baseline metrics will be recorded here once measured. A running
training cell and unexecuted evaluation cells must not be described as success.
The notebook metadata records whether execution is pending, running or complete.

## Remaining work

- Complete full raw training (or documented native early stopping), save best/last
  hashes, and evaluate the selected checkpoint on the full tune partition.
- Calibrate fixed parameters and adaptive thresholds on train/tune only, comparing
  JPEG95-only control, Gamma, CLAHE, Denoise, Sharpen and Adaptive with the same frozen
  checkpoint, split IDs, labels and metric settings.
- Measure preprocessing costs and representative latency with warm-up and
  synchronized CUDA timing; distinguish exported-file inference from online cost.
- Finalize and freeze the experiment protocol; only then prepare/evaluate untouched
  official-val holdout after full leakage checks.
- Complete natural/synthetic robustness, rare-class analysis, repeated seeds and
  uncertainty as scoped in Issue #1. Do not close Issue #1 for this first run.

No controlled treatment result, final holdout score, robustness gain or reliable
latency ranking is claimed. Native resume support is implemented; an intentional
mid-training interruption has not yet been exercised in Phase 2.

## Verified local checks

```sh
.venv/bin/python -m pytest -q        # 17 passed; includes native converter fixture
.venv/bin/ruff check .              # passed
.venv/bin/ruff format --check .      # passed
uv build                           # source and wheel built
git diff --check                   # passed
```

These checks do not establish GPU training completion. Phase 1 execution remains
available in the merged PR; the two working notebooks now record Phase 2 only.
