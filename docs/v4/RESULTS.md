# V4 — completed BDD100K development evaluation

V4 is a fixed Canny-guided detail policy, independently full fine-tuned from
the pinned COCO YOLO26n checkpoint for 50 epochs on 9,781 S20 images. Its own
processed calibration set (6,985 images) selects best.pt. Evaluation uses the
same 13,975 development images and five-class GT as the matched Raw control.

## Matched batch16 comparison

| Policy | Precision (%) | Recall (%) | mAP@50 (%) | mAP@50:95 (%) |
|---|---:|---:|---:|---:|
| Raw, matched local control | 56.2579 | 39.7743 | 42.5961 | 25.8215 |
| V4, Canny-guided detail | 56.7137 | 39.4288 | 42.0345 | 25.6708 |
| V4 minus Raw (percentage points) | +0.4558 | -0.3455 | -0.5616 | -0.1507 |

**Conclusion:** v4 does not improve either overall AP measure over the matched
Raw control in this single-seed experiment. Precision is slightly higher, but
recall and both AP measures are lower. This is not a general proof about every
edge detector or every training seed.

- V4 run: `runs/v4-2026-11-10`.
- Matched Raw source alias: `runs/v5_raw-2026-11-11`, a Raw baseline run under the
  later execution suite. Its relevant evidence is copied to
  [evidence/matched_raw](evidence/matched_raw); no v5 runtime code is required.
- Exact overall results: [comparison.csv](comparison.csv).
- V4 full native metrics: [metrics.json](../../runs/v4-2026-11-10/eval/metrics.json).

Both runs use seed2026, imgsz640, 50 epochs, freeze0, SGD, AMP, batch16/workers0,
matching augmentation, five-class taxonomy, fixed splits and initialization.
Runtime was PyTorch2.7.1+cu126, Ultralytics8.4.198 and OpenCV5.0.0 on GTX1660Ti6GiB.
The OpenCV runtime differs from the repository's locked `<5` dependency bound;
these are archived local measurements, not a claim that this environment was
created by an unmodified `uv sync --locked`. A new run must record its runtime.

## Class results at AP@50

| Class | Raw (%) | V4 (%) | Delta (pp) |
|---|---:|---:|---:|
| car | 72.0140 | 72.0872 | +0.0732 |
| truck | 46.7364 | 47.0376 | +0.3012 |
| bus | 42.9552 | 43.0579 | +0.1027 |
| motorcycle | 21.7510 | 19.9016 | -1.8494 |
| bicycle | 29.5237 | 28.0882 | -1.4355 |

The two-wheeler regressions deserve follow-up error analysis. These numbers
do not establish the cause of the regressions or AP by object size/occlusion.
Routing applies v4 to 13,750/13,975 development images (98.39%). Native speed
figures exclude the external offline policy and are not end-to-end latency.

## Evidence and interpretation

Native train history, args, completion marker, best/last weights, evaluation
curves, confusion matrices, examples and per-image matching arrays are included
under the v4 run. Executed notebooks are under `runs/v4-execution-2026-11-10`.
Original/processed BDD datasets and machine caches are excluded.

The saved v0–v3 outputs from 2026-11-09 use batch32 and a different runtime.
They are preserved and discussed separately in the reports, not merged into
this matched batch16 ranking. The 13-cell comparison interface also supports
optional cross-input evaluations, but a complete factorial study has not been
performed merely by publishing that interface.

Final-test images were not decoded, hashed or evaluated in this execution.
No new bootstrap CI, multi-seed result, independent confirmation or successful
prospective pilot gate is claimed. Earlier documents describing unexecuted
plans are retained as dated historical notes with current-status addenda.
