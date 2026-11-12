> Historical note: the text below records the state at its original date.
> BDD100K training/evaluation has since completed; see [current results](RESULTS.md)
> and [publication guide](README.md). Earlier absence-of-data/tool statements are historical.

# v4 full execution — local plan, 2026-11-10

The user explicitly requested downloading BDD100K and running the full v4 lifecycle.
This authorizes execution of the fixed exploratory Canny hypothesis; it does not
assert that an independent detector pilot or confirmation has passed. There is
no threshold tuning or policy search during this run.

## Inputs and hardware

- Public archive: `https://zenodo.org/records/7332349/files/bdd100k.zip?download=1`.
  Publisher size: 5,829,172,026 bytes; MD5: `10807adc4280714cbc38d4399f4a9352`.
- Use the frozen split table unchanged. Store originals at its exact relative
  source paths, preserving JPEG bytes; extract only frozen memberships and the
  official detection JSONs. Ignore macOS metadata and unrelated official test images.
- Verify all original non-final image hashes, dimensions and per-frame vehicle GT
  against the existing table before preparation. Do not silently resplit or rehash
  a changed dataset into a new protocol. Stop and report any incompatibility.
- The pinned `weights/yolo26n.pt` has the required SHA256
  `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`.
- NVIDIA GTX 1660 Ti, 6 GiB. AMP batch32 synthetic forward/backward reached 4.935
  GiB with about 5.01 GiB GPU memory free. System RAM was 15.8 GiB total, about
  3 GiB available. Therefore v4 uses training/evaluation batch16 and training
  workers0, declared before real preparation/training. v0–v3 configs are unchanged.
- Keep 50 epochs, SGD, seed2026, 640, freeze0, nbs64, checkpoint and photometric/
  geometry settings as configured. Batch16 changes native gradient accumulation;
  any matched Raw control must use the same local settings.
- Local `.venv-study` reuses the existing CUDA torch 2.7.1+cu126 installation in
  `D:/Self_driving/.venv-gpu` without modifying it; installs Ultralytics **8.4.198**
  and ijson locally. This is not a complete locked `uv sync` environment.

## Execution

`runs/downloads/run_v4_full.py` runs as a hidden background process. It owns the
resumable download, verifies the publisher checksum, acquires exact source paths,
audits sources and runs **each v4 notebook in a separate fresh kernel**:

1. `01_experiments`: 64 calibration images, SHOW=8; pixel diagnostics only.
2. `02_train_prepare`: RUN_DATE=2026-11-10, ALLOW_PREPARATION=True.
3. `03_train`: same run date, ALLOW_TRAINING=True, 50 epochs; no previous run is resumed.
4. `04_eval`: same run date, ALLOW_INFERENCE=True, processed development only.

Executed notebooks and phase/error status are saved under
`runs/v4-execution-2026-11-10/`. Native artifacts are under `runs/v4-2026-11-10/`.
`status.json` says `complete` only after real development metrics are saved;
otherwise it reports the active phase or an actual failure. No mAP result is
fabricated while downloading, preparing or training.

The final holdout is acquired as bytes for membership accounting but is never
decoded, hashed, displayed, used for checkpoint selection or evaluated. A complete
Raw/v4 2×2 requires a comparable Raw run and is separate from this v4-only lifecycle.

## Validation before real execution

The full-study training/evaluation/comparison/notebook checks passed: **19 passed**,
including native tiny synthetic CPU training and fresh-kernel lifecycle execution.
One sampled original training image retrieved from the archive matched the frozen
SHA256 exactly; this is only an early compatibility check, not the full source audit.
Tiny synthetic tests and the earlier 12-image non-BDD preview are not study metrics.
