# Local road detection video demo

The Gradio app supports **Vehicle**, **Traffic Sign**, and **Combined** detection on
one uploaded video. Vehicle models (v0–v4) and preprocessing policies (v0–v4) remain
independently selectable. Traffic Sign mode selects Hiếu or Phát with native frames.
Combined mode runs one model from each category and writes one annotated video.
It reads trusted local checkpoints; it does not train, evaluate AP, or modify study artifacts.

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

FFmpeg conversion has a separate default timeout of **600 seconds**. On slower
CPUs or with larger clips, raise it at launch, for example `--encoding-timeout 1800`.
A timeout names this option in its error and removes partial files.

## Workflow

1. Upload an MP4, MOV or AVI video.
2. Select **Detection mode**. Vehicle shows the vehicle model/policy controls;
   Traffic Sign shows the Hiếu/Phát selector and **Native / No additional filtering**;
   Combined shows both. Confidence and device apply to both detectors. A missing
   sign checkpoint disables modes that need it without blocking Vehicle mode.
   Either installed sign model is usable independently; no model substitution occurs.
3. For Vehicle or Combined, select **Model version**. Startup checks dated run directories,
   newest first. It selects the newest valid `train/weights/best.pt` whose saved
   `config.yaml` matches the version's policy. If `train/complete.json` exists,
   its checkpoint hash must also match. The checkpoint must load as a detector
   with exactly `car, truck, bus, motorcycle, bicycle` in that order.
4. Inspect **Checkpoint identities and unavailable versions** for the selected
   path, SHA256, training policy digest, model image size, rejected runs, and
   actionable errors. Unavailable versions are excluded from the model selector;
   the app still opens if none are available. No fallback to another version occurs.
   Restart the app after restoring weights. Only load checkpoints you trust:
   PyTorch checkpoints can contain executable serialized objects.
5. For Vehicle or Combined, select **Preprocessing version** independently. Its current YAML supplies
   the recipe: v0 Raw, v1 Gamma-or-Skip, v2 Local-shadow, v3 Adaptive local contrast,
   v4 Canny-guided detail. Missing/invalid policy YAMLs are excluded and reported.
6. For Traffic Sign or Combined, select **Traffic Sign model**: Hiếu — YOLO26n (832)
   or Phát — YOLO26n (640). Follow the explicit installation below first.
7. Choose detection confidence (default **0.25**) and CPU/GPU. This threshold is
   distinct from the study's AP-evaluation confidence threshold of 0.001.
8. Click **Run inference**. A progress indicator reports decoded/processed frames.
   The original and annotated videos appear on completion, with an MP4 download
   and a JSON report. **Cancel active job** stops processing at the next frame
   boundary; a running model call or final FFmpeg conversion finishes first.

The queue permits one inference job at a time, with at most four waiting requests.
Cancellation leaves queued requests in place. Cross-policy choices visibly display
**exploratory / out-of-distribution pair**, also recorded in the report. Models
were trained on their own policy's outputs; this UI makes no AP-improvement claims.

## Pipeline and metrics

Vehicle frames follow **OpenCV BGR uint8 → selected `apply_recipe` → JPEG quality 95
encode/decode → selected detector**, using the model run's configured training
`imgsz`. Box/label/confidence rendering uses the original decoded frame; the policy
and JPEG stage preserve coordinates. Output dimensions and valid source FPS are
preserved. The final MP4 is checked for readability, dimensions, FPS and frame count.

Traffic Sign frames follow **original OpenCV BGR uint8 → sign YOLO** at the selected
model's image size. No vehicle recipe, JPEG95, or training augmentation is added.
Combined mode fans out independent copies of the same original frame into both
pipelines, then draws `Vehicle / car` and `Sign / No U-turn` labels on one output.
Each detector keeps its own class IDs; there is no cross-model NMS or tracking.

The report includes checkpoint/policy identities, confidence/device, processed frame
count, resolution, source FPS, derived duration, detection counts per class (summed
across frames, **not unique tracked vehicles**) and Apply/Skip frame counts. Raw
always counts as Skip. A routed operator counts as Apply even if its output happens
to equal its input. `detectors` contains each model's identity (source revision,
checkpoint SHA256, taxonomy, image size and preprocessing), separate class counts,
and stage timings. Only Vehicle has Apply/Skip routing. Single-detector reports keep
the existing flat `detections_per_class`; Combined keys that field by model ID to
avoid merging taxonomies. Sign-only reports use `preprocessing_version=native` and
`routing=null`.

Wall-clock measurements include per-frame decode, preprocessing, JPEG, inference,
and rendering/encoding totals plus mean/p50/p95 milliseconds. Inference includes
copying class IDs to CPU to wait for CUDA results. Model loading/checkpoint
validation and encoder finalization/FFmpeg conversion are reported separately.
`end_to_end_fps` divides completed frames by total service time, including those
startup/finalization costs. Upload, queue wait, Gradio cache copying and browser
playback are outside that interval. These demo measurements are **not** the
study's Ultralytics-internal `speed_ms`; short clips and first-call warmup can
dominate them. Frame latency samples are retained only as bounded scalar values;
video frames are decoded, inferred and written incrementally. Combined totals include
both detector calls and rendering; overall FPS is measured from total service time,
not estimated from either detector's internal speed. Native sign preprocessing/JPEG
stages report zero. Per-detector rendering excludes the shared video writer.

## Install the external sign checkpoints explicitly

The committed [external model manifest](../configs/external_models.yaml) pins the
source revision, source checkpoint path, local path, SHA256, task, ordered taxonomy,
native preprocessing and image size for each detector:

- **Hiếu:** [ec2003/VN-traffic-sign-detection at 1bcf75e](https://github.com/ec2003/VN-traffic-sign-detection/tree/1bcf75eebbb3b255f622ade63900cead00abd863),
  `output/working/runs/detect/train/weights/best.pt`, `imgsz=832`.
  The source repository has an Apache-2.0 LICENSE.
- **Phát:** [dhphat/traffic-sign-detection at f3c30d8](https://github.com/dhphat/traffic-sign-detection/tree/f3c30d8be4e31404a73e8621d8fb050a62c90e05),
  `runs/detect/runs/detect/traffic_sign_train/weights/best.pt`, `imgsz=640`.
  The nested `runs/detect/runs/detect` path is intentional. The reviewed repository
  has no visible LICENSE; confirm use and redistribution permission with its owner.

Both checkpoints loaded successfully in the locked demo environment and report
`task=detect` with IDs 0–5 exactly: `No U-turn`, `No going opposite`, `No stopping`,
`No waiting`, `Roundabout ahead`, `Speed limit 60kmh`. Their training recipes and
datasets differ; published AP values are **not a controlled head-to-head benchmark**
and are not comparable to the five-class BDD100K experiment.

Checkpoint bytes remain **local-only** under gitignored `weights/`. Confirm the
checkpoint-specific permissions and trust the publisher before installing. After
obtaining permission, either copy the exact pinned files to the two manifest paths
or explicitly download only those files:

```sh
mkdir -p weights/external
curl --fail --location \
  https://raw.githubusercontent.com/ec2003/VN-traffic-sign-detection/1bcf75eebbb3b255f622ade63900cead00abd863/output/working/runs/detect/train/weights/best.pt \
  -o weights/external/traffic-sign-hieu.pt
curl --fail --location \
  https://raw.githubusercontent.com/dhphat/traffic-sign-detection/f3c30d8be4e31404a73e8621d8fb050a62c90e05/runs/detect/runs/detect/traffic_sign_train/weights/best.pt \
  -o weights/external/traffic-sign-phat.pt
sha256sum --check <<'SHA256'
9e25eab0aa5a3895d5ccf9b37ab730075953c368fd8b57b34f30dd859639150d  weights/external/traffic-sign-hieu.pt
f42c8c553b266c728883236596fb133624094c68df2159864e447ef2ac0a07ef  weights/external/traffic-sign-phat.pt
SHA256

# Validate hashes before loading, then exact task and class order, in the demo environment.
UV_PROJECT_ENVIRONMENT=.venv-demo uv run --locked --extra demo python - <<'PY'
from pathlib import Path
from vehicle_detection.model_registry import discover_sign_models
for key, entry in discover_sign_models(Path.cwd()).items():
    print(key, entry.error or f"Ready: imgsz={entry.imgsz}, sha256={entry.sha256}")
    assert not entry.error, entry.error
PY
```

Restart the app after installation. Startup rejects a missing or mismatched hash
**before deserializing** that checkpoint, then validates actual task and class names.
Every job rechecks hashes and model taxonomy. Unavailable reasons appear in the
Models panel; other installed detectors remain usable. There are no runtime weight
downloads or fallback models. A matching hash identifies bytes; it does not make an
untrusted PyTorch pickle safe.

Do not copy upstream datasets, notebooks, requirements, or credentials. The issue
review identified a hard-coded Roboflow credential in Hiếu's upstream notebook:
the owner should revoke/rotate it and remediate repository history. This integration
does not ingest that notebook or reproduce the credential; owner notification remains
an upstream follow-up. Source repository licensing does not resolve all checkpoint,
dataset, or Ultralytics distribution terms; verify them before redistributing weights.

## Limits, files and known behavior

Defaults are 200 MiB/upload, 120 seconds, 7,200 frames, and 8,294,400 pixels/frame.
Change them with `--max-mb`, `--max-seconds`, `--max-frames`, `--max-pixels` (positive
integers). The service checks metadata before processing and enforces decoded
frame/duration/pixel limits as well. Upload paths, type, size, output paths, device
and video metadata are validated before loading the detector. Invalid/missing FPS
falls back to 30 with an explicit report warning; duration is then an estimate.
Frame count metadata may be absent, in which case progress shows processed frames
without a percentage.
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

# Exercise UI wiring and real H.264 encode/decode (requires FFmpeg/ffprobe).
UV_PROJECT_ENVIRONMENT=.venv-demo uv run --locked --extra demo pytest -q tests/test_inference.py
```

The separate `demo-smoke` CI job installs only the `demo` extra and FFmpeg, rejects
any skipped demo test, and checks runtime imports/app construction. Its CPU-only
synthetic MP4 tests exercise real video IO and H.264 conversion, verify yuv420p,
frame count, dimensions and FPS with ffprobe and full decode, and pass the output
through the Gradio video component. They stub only the detector; no GPU is needed.

Use a short local clip to check `model=v0, preprocessing=v0`, then
`model=v0, preprocessing=v3`. Both should preserve frame count/FPS/dimensions;
only the latter should show the exploratory warning. Synthetic checks validate
software behavior, not detection quality or scientific conclusions. The automated
MP4 tests also cover each sign model separately and Combined mode with each sign
model, using CPU detector stubs and real codecs. They check native input isolation,
distinct combined labels, separate counts/timings, invalid weights, cancellation and
cleanup. Local CPU smoke checks also ran both pinned real sign checkpoints separately
and with the real v0 checkpoint on a three-frame synthetic MP4; these establish
runtime compatibility, not road-scene accuracy.

For manual GPU/browser validation:

1. Launch with `--device 0` and upload a short, trusted road clip containing signs
   (outside the study's sealed final holdout). Run each sign model separately.
2. Check native preprocessing, correct 832/640 identities, legible sign labels,
   original/annotated browser playback, download, and frame count/FPS/dimensions.
3. Run Combined with v0/v0 and then v0/v3, once per sign model. Verify both prefixes,
   separate class counts, Vehicle-only routing, and the exploratory warning for v0/v3.
4. Cancel a running Combined job and rerun; confirm temporary files are removed and
   the next job works. Remove one local sign checkpoint and restart: the other sign
   and Vehicle modes must remain usable, with an actionable unavailable reason.

GPU/browser checks and road-scene detection quality require this manual smoke;
CPU/synthetic checks do not establish them.
