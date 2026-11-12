"""Bounded, streaming demo inference; no training or study artifact writes."""

import math
import re
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import yaml

from .common import digest, file_hash, read_json
from .data import FIVE_CLASSES
from .experiment_data import load_experiment, policy_identity
from .preprocess import apply_recipe, validate_image

VERSIONS = tuple(f"v{i}" for i in range(5))
EXPLORATORY = "exploratory / out-of-distribution pair"


@dataclass(frozen=True)
class ModelEntry:
    version: str
    checkpoint: Path | None = None
    sha256: str = ""
    policy_id: str = ""
    imgsz: int = 640
    error: str = ""
    rejected: tuple[str, ...] = ()

    def identity(self):
        return {
            "version": self.version,
            "checkpoint": str(self.checkpoint) if self.checkpoint else None,
            "sha256": self.sha256,
            "training_policy_id": self.policy_id,
            "imgsz": self.imgsz,
            "unavailable_reason": self.error,
            "rejected_runs": list(self.rejected),
        }


@dataclass(frozen=True)
class VideoLimits:
    max_mb: int = 200
    max_seconds: int = 120
    max_pixels: int = 3840 * 2160
    max_frames: int = 7200
    encoding_timeout_seconds: int = 600

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in vars(self).values()):
            raise ValueError("Video limits must be positive integers")


DEFAULT_LIMITS = VideoLimits()


def load_detector(checkpoint):
    """Only load trusted local checkpoints; never trigger an automatic weight download."""
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise ValueError(f"Missing checkpoint: {checkpoint}; restore its train/weights/best.pt")
    try:
        from ultralytics import YOLO

        model = YOLO(str(checkpoint))
        if model.task != "detect" or model.names != dict(enumerate(FIVE_CLASSES)):
            raise ValueError(f"Expected detection taxonomy {FIVE_CLASSES}")
        return model
    except Exception as exc:
        raise ValueError(f"Cannot load {checkpoint}: {exc}") from exc


def discover_models(root):
    """Newest valid dated run per version, with every rejected candidate made visible."""
    root = Path(root).resolve()
    registry = {}
    for version in VERSIONS:
        rejected = []
        try:
            config = load_experiment(root, version)
            policy_id = digest(policy_identity(config))
            runs = sorted(
                p
                for p in (root / "runs").glob(f"{version}-*")
                if re.fullmatch(rf"{version}-\d{{4}}-\d\d-\d\d", p.name) and p.is_dir()
            )
            for run in reversed(runs):
                try:
                    snapshot = yaml.safe_load((run / "config.yaml").read_text())
                    if digest(policy_identity(snapshot)) != policy_id:
                        raise ValueError("run policy differs from the version YAML")
                    imgsz = snapshot["training"]["imgsz"]
                    if type(imgsz) is not int or not 32 <= imgsz <= 4096:
                        raise ValueError("run training.imgsz must be an integer in [32, 4096]")
                    checkpoint = run / "train/weights/best.pt"
                    sha256 = file_hash(checkpoint)
                    marker = run / "train/complete.json"
                    if marker.is_file() and read_json(marker)["best_sha256"] != sha256:
                        raise ValueError("checkpoint differs from train/complete.json")
                    load_detector(checkpoint)
                    registry[version] = ModelEntry(
                        version, checkpoint, sha256, policy_id, imgsz, rejected=tuple(rejected)
                    )
                    break
                except Exception as exc:
                    rejected.append(f"{run.name}: {exc}")
            else:
                raise ValueError(
                    f"No valid runs/{version}-YYYY-MM-DD/train/weights/best.pt; "
                    "restore a trusted trained run and restart the app"
                )
        except Exception as exc:
            registry[version] = ModelEntry(version, error=str(exc), rejected=tuple(rejected))
    return registry


def validate_device(device):
    if device == "cpu":
        return
    if device != "0":
        raise ValueError("Device must be cpu or 0 (the first CUDA GPU)")
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise ValueError("CUDA GPU 0 is unavailable; select cpu or install a CUDA-enabled PyTorch")


def process_frame(frame, model, policy, *, confidence, imgsz, device):
    """Apply the independently selected policy, JPEG95, then draw on source coordinates."""
    validate_image(frame)
    start = perf_counter()
    processed, applied, _ = apply_recipe(frame, policy["variant"], policy["recipes"])
    if processed.shape != frame.shape or processed.dtype != frame.dtype:
        raise ValueError("Preprocessing changed frame geometry/dtype")
    preprocess_end = perf_counter()
    ok, encoded = cv2.imencode(".jpg", processed, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ValueError("JPEG95 encoding failed")
    processed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if processed is None or processed.shape != frame.shape:
        raise ValueError("JPEG95 decoding failed or changed geometry")
    jpeg_end = perf_counter()
    result = model.predict(
        source=processed, conf=confidence, imgsz=imgsz, device=device, verbose=False, save=False
    )[0]
    # Copy detections to CPU here so inference timing includes asynchronous CUDA work.
    classes = result.boxes.cls.cpu().numpy().astype(int).tolist()
    inference_end = perf_counter()
    if any(c < 0 or c >= len(FIVE_CLASSES) for c in classes):
        raise ValueError("Detector returned a class outside the five-class taxonomy")
    annotated = result.plot(img=frame.copy(), conf=True, labels=True, line_width=2)
    if annotated.shape != frame.shape:
        raise ValueError("Detector rendering changed frame geometry")
    return (
        annotated,
        bool(applied),
        Counter(FIVE_CLASSES[c] for c in classes),
        {
            "preprocessing": preprocess_end - start,
            "jpeg": jpeg_end - preprocess_end,
            "model_inference": inference_end - jpeg_end,
            "rendering_encoding": perf_counter() - inference_end,
        },
    )


def open_writer(path, fps, size):
    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            if codec == "mp4v" and not shutil.which("ffmpeg"):
                writer.release()
                raise ValueError(
                    "OpenCV H.264 is unavailable; install FFmpeg with libx264 and restart"
                )
            return writer, codec
        writer.release()
    raise ValueError("Cannot open an MP4 encoder; install OpenCV video codecs and FFmpeg")


def finish_video(
    intermediate,
    output,
    codec,
    frames,
    size,
    fps,
    *,
    timeout=DEFAULT_LIMITS.encoding_timeout_seconds,
):
    if codec == "avc1":
        intermediate.replace(output)
    else:
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-y",
                    "-i",
                    str(intermediate),
                    "-an",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(output),
                ],
                check=True,
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise ValueError(
                f"FFmpeg conversion exceeded {timeout} seconds; "
                "increase --encoding-timeout for this CPU/video"
            ) from exc
        except (OSError, subprocess.SubprocessError) as exc:
            detail = (getattr(exc, "stderr", b"") or b"").decode(errors="replace")[-1000:]
            raise ValueError(
                f"FFmpeg H.264 conversion failed; check libx264 support. {detail}"
            ) from exc
        intermediate.unlink()
    check = cv2.VideoCapture(str(output))
    try:
        ok, first = check.read()
        if not ok:
            raise ValueError("Output MP4 is unreadable; check OpenCV/FFmpeg codec support")
        if (
            first.shape[1::-1] != size
            or check.get(cv2.CAP_PROP_FRAME_COUNT) != frames
            or not math.isclose(check.get(cv2.CAP_PROP_FPS), fps, rel_tol=1e-4, abs_tol=0.01)
        ):
            raise ValueError(
                "MP4 encoding changed frame count, dimensions or FPS; check disk/codecs"
            )
    finally:
        check.release()


def infer_video(
    source,
    output,
    entry,
    policy,
    *,
    confidence=0.25,
    device="cpu",
    limits=DEFAULT_LIMITS,
    progress=None,
    cancelled=None,
):
    """Process one frame at a time; caller owns output lifetime. Remove partial files on failure."""
    start = perf_counter()
    if entry.error or entry.checkpoint is None:
        raise ValueError(entry.error or "Select an available model version")
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Confidence must be in [0, 1]")
    source, output = Path(source), Path(output)
    if source.suffix.lower() not in {".mp4", ".mov", ".avi"} or not source.is_file():
        raise ValueError("Upload a local MP4, MOV or AVI file")
    if not 0 < source.stat().st_size <= limits.max_mb * 1024**2:
        raise ValueError(f"Video must be nonempty and at most {limits.max_mb} MiB")
    if output.suffix.lower() != ".mp4" or output.exists() or output.resolve() == source.resolve():
        raise ValueError("Output must be a new .mp4 path, separate from the source")
    intermediate = output.with_name(output.stem + ".encoding.mp4")
    if intermediate.exists() or intermediate.resolve() == source.resolve():
        raise ValueError("Intermediate output already exists; use a fresh output directory")
    validate_device(device)
    capture = cv2.VideoCapture(str(source))
    writer = None
    timings = {
        stage: []
        for stage in ("decode", "preprocessing", "jpeg", "model_inference", "rendering_encoding")
    }
    counts = Counter(dict.fromkeys(FIVE_CLASSES, 0))
    applied_count = frames = 0
    warnings = []
    try:
        if not capture.isOpened():
            raise ValueError("Cannot decode video; check the file and installed OpenCV codecs")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            fps = 30.0
            warnings.append("Source FPS is missing/invalid; output and duration use 30 FPS")
        declared = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        declared = int(declared) if math.isfinite(declared) and declared > 0 else 0
        if declared > limits.max_frames or declared / fps > limits.max_seconds:
            raise ValueError(
                f"Video exceeds {limits.max_seconds} seconds/{limits.max_frames} frames"
            )
        width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if width * height > limits.max_pixels:
            raise ValueError(f"Video exceeds the {limits.max_pixels} pixel frame limit")
        loading_at = perf_counter()
        if file_hash(entry.checkpoint) != entry.sha256:
            raise ValueError("Checkpoint changed since startup; restart to validate the registry")
        model = load_detector(entry.checkpoint)
        model_load_seconds = perf_counter() - loading_at
        while True:
            if cancelled is not None and cancelled():
                raise ValueError("Inference cancelled; partial output removed")
            decoded_at = perf_counter()
            ok, frame = capture.read()
            decode_seconds = perf_counter() - decoded_at
            if not ok:
                break
            validate_image(frame)
            h, w = frame.shape[:2]
            if w * h > limits.max_pixels or w % 2 or h % 2:
                raise ValueError(
                    f"H.264 requires even dimensions and at most {limits.max_pixels} pixels; "
                    "resize the source before uploading"
                )
            if frames == 0:
                size = (w, h)
                writer, codec = open_writer(intermediate, fps, size)
            elif (w, h) != size:
                raise ValueError("Video resolution changes between frames; use a fixed-size source")
            if frames + 1 > limits.max_frames or (frames + 1) / fps > limits.max_seconds:
                raise ValueError("Decoded video exceeds the configured duration/frame limit")
            annotated, applied, detections, elapsed = process_frame(
                frame, model, policy, confidence=confidence, imgsz=entry.imgsz, device=device
            )
            encoded_at = perf_counter()
            writer.write(annotated)
            elapsed["rendering_encoding"] += perf_counter() - encoded_at
            elapsed["decode"] = decode_seconds
            for stage, seconds in elapsed.items():
                timings[stage].append(seconds)
            frames += 1
            applied_count += applied
            counts.update(detections)
            if progress is not None:
                progress(frames, declared)
        if not frames:
            raise ValueError("Video contains no decodable frames")
        if declared and frames < declared - 1:
            raise ValueError(f"Video ended early ({frames}/{declared} frames); file may be corrupt")
        encoding_at = perf_counter()
        writer.release()
        writer = None
        if cancelled is not None and cancelled():
            raise ValueError("Inference cancelled; partial output removed")
        finish_video(
            intermediate,
            output,
            codec,
            frames,
            size,
            fps,
            timeout=limits.encoding_timeout_seconds,
        )
        if cancelled is not None and cancelled():
            raise ValueError("Inference cancelled; partial output removed")
        final_encoding_seconds = perf_counter() - encoding_at
        total_seconds = perf_counter() - start
        return {
            "model": entry.identity(),
            "preprocessing_version": policy["id"],
            "preprocessing_policy_id": digest(policy_identity(policy)),
            "pair": EXPLORATORY if entry.version != policy["id"] else "matched model/policy",
            "confidence": confidence,
            "device": device,
            "frames": frames,
            "resolution": list(size),
            "source_fps": fps,
            "duration_seconds": frames / fps,
            "detections_per_class": dict(counts),
            "routing": {"Apply": applied_count, "Skip": frames - applied_count},
            "total_seconds": total_seconds,
            "end_to_end_fps": frames / total_seconds,
            "model_load_seconds": model_load_seconds,
            "final_encoding_seconds": final_encoding_seconds,
            "stages": {
                stage: {
                    "total_seconds": sum(samples),
                    "mean_ms": 1000 * np.mean(samples),
                    "p50_ms": 1000 * np.percentile(samples, 50),
                    "p95_ms": 1000 * np.percentile(samples, 95),
                }
                for stage, samples in timings.items()
            },
            "timing_note": (
                "Wall-clock demo timings include JPEG95 and video work; not study speed_ms"
            ),
            "warnings": warnings,
        }
    except BaseException:
        if writer is not None:
            writer.release()
            writer = None
        intermediate.unlink(missing_ok=True)
        output.unlink(missing_ok=True)
        raise
    finally:
        capture.release()
        if writer is not None:
            writer.release()
