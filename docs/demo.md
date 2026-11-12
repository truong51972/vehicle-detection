# Local video inference demo

The Gradio app lets you choose a trained model (v0–v4) and a preprocessing policy
(v0–v4) independently. It reads existing version YAMLs and trusted local checkpoints;
it does not run training, evaluate AP, or modify study artifacts.

## Environment and launch

Use Python 3.11 and the committed `uv.lock`. The `demo` extra pins Gradio 6.31.0
and the study's Ultralytics 8.4.198. Gradio needs Pydantic 2; `study` requires
Pydantic 1. They are explicitly mutually exclusive in uv. Keep separate environments:

```sh
# From the repository root. Do not combine --extra demo and --extra study.
UV_PROJECT_ENVIRONMENT=.venv-demo uv sync --locked --python 3.11 --extra demo

# CPU (default)
UV_PROJECT_ENVIRONMENT=.venv-demo uv run --locked --extra demo python apps/gradio_demo.py

# First CUDA GPU; requires a working NVIDIA driver/CUDA-enabled PyTorch.
UV_PROJECT_ENVIRONMENT=.venv-demo uv run --locked --extra demo python apps/gradio_demo.py --device 0
```

Open http://127.0.0.1:7860. The app binds only to localhost and does not create a
public sharing link. `--port 7861` selects another port. Continue using the study
commands in the README with `.venv` (or `UV_PROJECT_ENVIRONMENT=.venv-study`).

Install the system **FFmpeg with libx264** when OpenCV cannot encode H.264 directly
(usual for pip OpenCV wheels). On Ubuntu: `sudo apt-get install ffmpeg`. Check with
`ffmpeg -encoders` and look for `libx264`. The app first tries OpenCV `avc1`; otherwise
it streams to an intermediate MPEG-4 MP4 and converts it to H.264/yuv420p with
FFmpeg. An unavailable encoder or failed conversion produces an error rather than
offering an unplayable download.

## Workflow

1. Upload an MP4, MOV or AVI video.
2. Select **Model version**. Startup checks each version's dated run directories,
   newest first. It selects the newest valid `train/weights/best.pt` whose saved
   `config.yaml` matches the version's policy. If `train/complete.json` exists,
   its checkpoint hash must also match. The checkpoint must load as a detector
   with exactly `car, truck, bus, motorcycle, bicycle` in that order.
3. Inspect **Checkpoint identities and unavailable versions** for the selected
   path, SHA256, training policy digest, model image size, rejected runs, and
   actionable errors. Unavailable versions are excluded from the model selector;
   the app still opens if none are available. No fallback to another version occurs.
   Restart the app after restoring weights. Only load checkpoints you trust:
   PyTorch checkpoints can contain executable serialized objects.
4. Select **Preprocessing version** independently. Its own current YAML supplies
   the recipe: v0 Raw, v1 Gamma-or-Skip, v2 Local-shadow, v3 Adaptive local contrast,
   v4 Canny-guided detail. Missing/invalid policy YAMLs are excluded and reported.
5. Choose detection confidence (default **0.25**) and CPU/GPU. This threshold is
   distinct from the study's AP-evaluation confidence threshold of 0.001.
6. Click **Run inference**. A progress indicator reports decoded/processed frames.
   The original and annotated videos appear on completion, with an MP4 download
   and a JSON report. **Cancel active job** stops processing at the next frame
   boundary; a running model call or final FFmpeg conversion finishes first.

The queue permits one inference job at a time, with at most four waiting requests.
Cancellation leaves queued requests in place. Cross-policy choices visibly display
**exploratory / out-of-distribution pair**, also recorded in the report. Models
were trained on their own policy's outputs; this UI makes no AP-improvement claims.

## Pipeline and metrics

Each frame follows **OpenCV BGR uint8 → selected `apply_recipe` → JPEG quality 95
encode/decode → selected detector**, using the model run's configured training
`imgsz`. Box/label/confidence rendering uses the original decoded frame; the policy
and JPEG stage preserve coordinates. Output dimensions and valid source FPS are
preserved. The final MP4 is checked for readability, dimensions, FPS and frame count.

The report includes checkpoint/policy identities, confidence/device, processed frame
count, resolution, source FPS, derived duration, detection counts per class (summed
across frames, **not unique tracked vehicles**) and Apply/Skip frame counts. Raw
always counts as Skip. A routed operator counts as Apply even if its output happens
to equal its input.

Wall-clock measurements include per-frame decode, preprocessing, JPEG, inference,
and rendering/encoding totals plus mean/p50/p95 milliseconds. Inference includes
copying class IDs to CPU to wait for CUDA results. Model loading/checkpoint
validation and encoder finalization/FFmpeg conversion are reported separately.
`end_to_end_fps` divides completed frames by total service time, including those
startup/finalization costs. Upload, queue wait, Gradio cache copying and browser
playback are outside that interval. These demo measurements are **not** the
study's Ultralytics-internal `speed_ms`; short clips and first-call warmup can
dominate them. Frame latency samples are retained only as bounded scalar values;
video frames are decoded, inferred and written incrementally.

## Limits, files and known behavior

Defaults are 200 MiB/upload, 120 seconds, 7,200 frames, and 8,294,400 pixels/frame.
Change them with `--max-mb`, `--max-seconds`, `--max-frames`, `--max-pixels` (positive
integers). The service checks metadata before processing and enforces decoded
frame/duration/pixel limits as well. Invalid/missing FPS falls back to 30 with an
explicit report warning; duration is then an estimate. Frame count metadata may
be absent, in which case progress shows processed frames without a percentage.
A stream ending substantially before its declared frame count is rejected.

H.264/yuv420p requires even width and height; odd-size inputs are rejected with
instructions to resize the source, rather than silently changing geometry. Output
uses constant source FPS and omits audio. Variable-frame-rate timing, rotation
metadata, live streams, tracking and side-by-side policy comparisons are outside
this demo. Normalize unusual camera files before uploading. CPU can be slow,
especially for full-resolution preprocessing.

Intermediates use private system temporary directories and are removed on success,
failure or cancellation. Completed videos are copied to Gradio's cache before
removing the work directory. Gradio checks every ten minutes for cached uploads
and outputs older than one hour; download results before they expire. Do not use
repository folders for uploads/outputs. No uploaded/generated videos belong in git;
`.venv-demo` and `.venv-study` are ignored. As with any process, a forced kill can
leave system temporary files for OS cleanup.

## Checks

```sh
# Lightweight pipeline, registry, limits and error tests, without a model/GPU.
uv run --locked pytest -q tests/test_inference.py

# Also exercise UI wiring when Gradio is installed.
UV_PROJECT_ENVIRONMENT=.venv-demo uv run --locked --extra demo pytest -q tests/test_inference.py
```

Use a short local clip to check `model=v0, preprocessing=v0`, then
`model=v0, preprocessing=v3`. Both should preserve frame count/FPS/dimensions;
only the latter should show the exploratory warning. Synthetic checks validate
software behavior, not detection quality or scientific conclusions.
