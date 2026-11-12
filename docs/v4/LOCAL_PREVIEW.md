> Historical note: the text below records the state at its original date.
> BDD100K training/evaluation has since completed; see [current results](RESULTS.md)
> and [publication guide](README.md). Earlier absence-of-data/tool statements are historical.

# v4 ? local execution result (2026-11-10)

The actual v4 policy was executed in a fresh Jupyter kernel on 12 fixed seed-2026
samples from an existing local **non-BDD training dataset**. This run tests pixel
processing, not the frozen BDD100K detector experiment. No detector was trained,
no predictions were evaluated, and no mAP value exists.

- Input: `D:/Self_driving/data/vehicle_v6_originals/images/train`.
- Applied: 12/12; skipped: 0/12. This small sample does not establish general routing rates.
- Maximum absolute channel change before JPEG encoding: 12/255.
- Mean CPU policy time: 20.778 ms/image, one OpenCV thread, two warmups and five
  repetitions per image. Source sizes vary; this is not end-to-end detector latency.
- Pinned `yolo26n.pt` was found locally, verified against the configured SHA256,
  and copied to `weights/yolo26n.pt`; the source checkpoint was not changed.

Local artifacts (gitignored):

- `runs/previews/v4-local-2026-11-10/index.html` ? 12-image gallery.
- `runs/previews/v4-local-2026-11-10/executed_preview.ipynb` ? executed notebook.
- `runs/previews/v4-local-2026-11-10/results.csv` ? per-image measurements.
- `runs/previews/v4-local-2026-11-10/summary.json` ? exact recipe and aggregate status.
- `sample-01.png` through `sample-12.png` ? Original / Canny / v4 / change panels.

BDD100K originals and detection annotations are absent, as confirmed by the user.
The full experiment cannot produce valid BDD100K training/evaluation metrics until
the dataset is acquired and checked against the frozen IDs, image hashes and GT.
Do not substitute this preview dataset for the frozen population.
