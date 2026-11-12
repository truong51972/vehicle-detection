"""Demo contracts with detector stubs and both simulated and real video IO."""

import json
import runpy
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
import yaml

from vehicle_detection import inference, model_registry
from vehicle_detection.common import file_hash
from vehicle_detection.data import FIVE_CLASSES
from vehicle_detection.experiment_data import config_path, load_experiment
from vehicle_detection.inference import finish_video
from vehicle_detection.model_registry import load_detector as load_checkpoint
from vehicle_detection.preprocess import apply_recipe

REPO = Path(__file__).resolve().parents[1]


class Tensor:
    def __init__(self, classes=(0, 4, 0)):
        self.classes = classes

    def cpu(self):
        return self

    def numpy(self):
        return np.array(self.classes)


class Detector:
    task = "detect"
    names = dict(enumerate(FIVE_CLASSES))

    def __init__(self, names=FIVE_CLASSES, classes=(0, 4, 0), marker=None):
        self.inputs = []
        self.rendered = []
        self.names = dict(enumerate(names))
        self.classes = classes
        self.marker = marker

    def predict(self, **kwargs):
        self.inputs.append(kwargs)

        def plot(**settings):
            self.rendered.append(settings | {"names": result.names.copy()})
            if self.marker is not None:
                settings["img"][0, self.marker] = 255
            return settings["img"]

        result = SimpleNamespace(
            boxes=SimpleNamespace(cls=Tensor(self.classes)), plot=plot, names=self.names
        )
        return [result]


@pytest.fixture
def registry_root(tmp_path, monkeypatch):
    root = tmp_path / "project"
    for version in inference.VERSIONS:
        path = config_path(root, version)
        path.parent.mkdir(parents=True)
        shutil.copyfile(config_path(REPO, version), path)
    monkeypatch.setattr(inference, "load_detector", lambda path, **kwargs: Detector())
    return root


def add_run(root, version="v0", date="2026-11-09"):
    run = root / f"runs/{version}-{date}"
    weights = run / "train/weights/best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"test checkpoint")
    shutil.copyfile(config_path(root, version), run / "config.yaml")
    return weights


def add_signs(root, monkeypatch):
    manifest = yaml.safe_load((REPO / "configs/external_models.yaml").read_text())
    for config in manifest.values():
        checkpoint = root / config["checkpoint"]
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"trusted test sign checkpoint")
        config["sha256"] = file_hash(checkpoint)
    (root / "configs").mkdir(exist_ok=True)
    (root / "configs/external_models.yaml").write_text(yaml.safe_dump(manifest))
    monkeypatch.setattr(
        model_registry, "load_detector", lambda path, *, names: Detector(names, (0, 5))
    )
    return model_registry.discover_sign_models(root)


def test_registry_latest_valid_with_rejection_reasons_and_missing_versions(registry_root):
    older = add_run(registry_root)
    newer = add_run(registry_root, date="2026-11-10")
    (newer.parents[1] / "complete.json").write_text(json.dumps({"best_sha256": "wrong"}))
    registry = inference.discover_models(registry_root)
    assert registry["v0"].checkpoint == older
    assert registry["v0"].sha256 == file_hash(older)
    assert "complete.json" in registry["v0"].rejected[0]
    assert registry["v0"].policy_id and registry["v0"].imgsz == 640
    assert registry["v1"].checkpoint is None and "restore" in registry["v1"].error


def test_registry_rejects_wrong_run_policy_and_corrupt_checkpoints(registry_root, monkeypatch):
    checkpoint = add_run(registry_root)
    config = yaml.safe_load((checkpoint.parents[2] / "config.yaml").read_text())
    config["id"] = "v3"
    (checkpoint.parents[2] / "config.yaml").write_text(yaml.safe_dump(config))
    assert "run policy differs" in inference.discover_models(registry_root)["v0"].rejected[0]
    shutil.copyfile(config_path(registry_root, "v0"), checkpoint.parents[2] / "config.yaml")

    def corrupt(path):
        raise ValueError("corrupt checkpoint")

    monkeypatch.setattr(inference, "load_detector", corrupt)
    assert "corrupt checkpoint" in inference.discover_models(registry_root)["v0"].rejected[0]


@pytest.mark.parametrize("invalid", ["taxonomy", "task", "corrupt", "missing"])
def test_checkpoint_validation(tmp_path, monkeypatch, invalid):
    checkpoint = tmp_path / "best.pt"
    if invalid != "missing":
        checkpoint.write_bytes(b"test")
    detector = Detector()
    if invalid == "taxonomy":
        detector.names = {0: "person"}
    if invalid == "task":
        detector.task = "segment"

    def load(path):
        if invalid == "corrupt":
            raise RuntimeError("invalid pickle")
        return detector

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=load))
    with pytest.raises(ValueError, match="taxonomy|pickle|Missing"):
        inference.load_detector(checkpoint)


@pytest.mark.parametrize("version", inference.VERSIONS)
def test_frame_policy_and_jpeg95_are_deterministic_and_preserve_source(version):
    policy = load_experiment(REPO, version)
    frame = np.full((64, 80, 3), (35, 40, 45), np.uint8)
    frame[:, 40:] += 5
    original = frame.copy()
    processed, applied, _ = apply_recipe(frame, policy["variant"], policy["recipes"])
    ok, encoded = cv2.imencode(".jpg", processed, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    expected = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    model = Detector()
    for _ in range(2):
        annotated, routed, counts, timing = inference.process_frame(
            frame, model, policy, confidence=0.25, imgsz=96, device="cpu"
        )
        assert np.array_equal(model.inputs[-1]["source"], expected)
        assert model.inputs[-1]["imgsz"] == 96 and model.inputs[-1]["conf"] == 0.25
        assert np.array_equal(annotated, original) and np.array_equal(frame, original)
        assert model.rendered[-1]["labels"] and model.rendered[-1]["conf"]
        assert routed == bool(applied)
        assert counts == {"car": 2, "bicycle": 1}
        assert all(value >= 0 for value in timing.values())


@pytest.fixture
def video_job(registry_root, tmp_path, monkeypatch):
    add_run(registry_root)
    entry = inference.discover_models(registry_root)["v0"]
    source, output = tmp_path / "source.mp4", tmp_path / "output.mp4"
    source.write_bytes(b"synthetic upload")
    model = Detector()
    monkeypatch.setattr(inference, "load_detector", lambda path, **kwargs: model)

    class Capture:
        fps = 10
        declared = 3
        total = 3
        read_count = 0
        released = False
        opened = True

        def isOpened(self):
            return self.opened

        def get(self, prop):
            return {
                cv2.CAP_PROP_FPS: self.fps,
                cv2.CAP_PROP_FRAME_COUNT: self.declared,
                cv2.CAP_PROP_FRAME_WIDTH: 80,
                cv2.CAP_PROP_FRAME_HEIGHT: 64,
            }[prop]

        def read(self):
            self.read_count += 1
            if self.read_count > self.total:
                return False, None
            return True, np.full((64, 80, 3), 40, np.uint8)

        def release(self):
            self.released = True

    capture = Capture()
    monkeypatch.setattr(inference.cv2, "VideoCapture", lambda path: capture)
    writer = SimpleNamespace(written=0, released=False, codec="avc1")

    def write(frame):
        # A frame must be written before the next one is decoded.
        assert capture.read_count == writer.written + 1
        writer.written += 1

    def release():
        writer.released = True

    writer.write, writer.release = write, release

    def open_writer(path, fps, size):
        path.write_bytes(b"partial encoded video")
        assert fps > 0 and size == (80, 64)
        return writer, writer.codec

    monkeypatch.setattr(inference, "open_writer", open_writer)
    monkeypatch.setattr(
        inference, "finish_video", lambda src, dst, *args, **kwargs: src.replace(dst)
    )
    return SimpleNamespace(
        source=source, output=output, entry=entry, model=model, capture=capture, writer=writer
    )


@pytest.mark.parametrize("version", ["v0", "v3"])
def test_video_summary_uses_policy_independent_of_model(video_job, version):
    job = video_job
    progress = []
    summary = inference.infer_video(
        job.source,
        job.output,
        job.entry,
        load_experiment(REPO, version),
        progress=lambda done, total: progress.append((done, total)),
    )
    assert job.output.is_file() and job.capture.released and job.writer.released
    assert summary["frames"] == 3 and summary["resolution"] == [80, 64]
    assert summary["source_fps"] == 10 and summary["duration_seconds"] == 0.3
    assert summary["routing"] == {
        "Apply": 3 if version == "v3" else 0,
        "Skip": 0 if version == "v3" else 3,
    }
    assert summary["detections_per_class"] == dict.fromkeys(FIVE_CLASSES, 0) | {
        "car": 6,
        "bicycle": 3,
    }
    assert summary["end_to_end_fps"] == 3 / summary["total_seconds"]
    assert summary["pair"] == (inference.EXPLORATORY if version == "v3" else "matched model/policy")
    assert set(summary["stages"]) == {
        "decode",
        "preprocessing",
        "jpeg",
        "model_inference",
        "rendering_encoding",
    }
    assert all(stage["p95_ms"] >= stage["p50_ms"] for stage in summary["stages"].values())
    assert progress == [(1, 3), (2, 3), (3, 3)]


@pytest.mark.parametrize("fps", [0, float("nan")])
def test_missing_fps_and_frame_count_use_safe_fallback(video_job, fps):
    job = video_job
    job.capture.fps, job.capture.declared = fps, 0
    result = inference.infer_video(job.source, job.output, job.entry, load_experiment(REPO, "v0"))
    assert result["source_fps"] == 30 and result["warnings"]


@pytest.mark.parametrize("failure", ["empty", "unreadable", "truncated", "limit", "cancel"])
def test_video_failure_releases_io_and_removes_partial_output(video_job, failure):
    job = video_job
    if failure == "empty":
        job.capture.total = 0
    elif failure == "unreadable":
        job.capture.opened = False
    elif failure == "truncated":
        job.capture.total = 1
    elif failure == "limit":
        job.capture.declared, job.capture.total = 0, 4
    with pytest.raises(ValueError):
        inference.infer_video(
            job.source,
            job.output,
            job.entry,
            load_experiment(REPO, "v0"),
            limits=inference.VideoLimits(max_frames=3),
            cancelled=lambda: failure == "cancel" and job.writer.written == 1,
        )
    assert job.capture.released
    if job.writer.written:
        assert job.writer.released
    assert not job.output.exists() and not job.output.with_suffix(".encoding.mp4").exists()


def test_checkpoint_changes_and_existing_outputs_rejected(video_job):
    job = video_job
    job.output.write_bytes(b"keep existing output")
    with pytest.raises(ValueError, match="new .mp4"):
        inference.infer_video(job.source, job.output, job.entry, load_experiment(REPO, "v0"))
    assert job.output.read_bytes() == b"keep existing output"
    job.output.unlink()
    job.entry.checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Checkpoint changed"):
        inference.infer_video(job.source, job.output, job.entry, load_experiment(REPO, "v0"))


def test_encoder_failure_is_actionable(tmp_path, monkeypatch):
    releases = []
    monkeypatch.setattr(
        inference.cv2,
        "VideoWriter",
        lambda *args: SimpleNamespace(
            isOpened=lambda: False, release=lambda: releases.append(True)
        ),
    )
    with pytest.raises(ValueError, match="MP4 encoder"):
        inference.open_writer(tmp_path / "out.mp4", 30, (80, 64))
    assert len(releases) == 2


def test_mp4v_requires_ffmpeg(tmp_path, monkeypatch):
    available = iter([False, True])
    monkeypatch.setattr(
        inference.cv2,
        "VideoWriter",
        lambda *args: SimpleNamespace(isOpened=lambda: next(available), release=lambda: None),
    )
    monkeypatch.setattr(inference.shutil, "which", lambda name: None)
    with pytest.raises(ValueError, match="install FFmpeg"):
        inference.open_writer(tmp_path / "out.mp4", 30, (80, 64))


def test_gpu_unavailable_and_invalid_limits(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    )
    with pytest.raises(ValueError, match="unavailable"):
        inference.validate_device("0")
    with pytest.raises(ValueError, match="positive integers"):
        inference.VideoLimits(max_seconds=0)
    with pytest.raises(ValueError, match="positive integers"):
        inference.VideoLimits(encoding_timeout_seconds=0)


@pytest.mark.parametrize(
    "failure",
    [
        "suffix",
        "missing",
        "empty",
        "size",
        "confidence",
        "pixels",
        "output",
        "output_suffix",
        "intermediate",
        "duration",
        "device",
    ],
)
def test_video_input_boundaries(video_job, monkeypatch, failure):
    job = video_job
    loader = Mock(side_effect=AssertionError("Detector loaded for an invalid upload"))
    monkeypatch.setattr(inference, "load_detector", loader)
    confidence = float("nan") if failure == "confidence" else 0.25
    limits = inference.VideoLimits(max_pixels=100 if failure == "pixels" else 80 * 64)
    if failure == "suffix":
        job.source = job.source.rename(job.source.with_suffix(".txt"))
    elif failure == "missing":
        job.source.unlink()
    elif failure == "empty":
        job.source.write_bytes(b"")
    elif failure == "size":
        with job.source.open("wb") as stream:
            stream.truncate(201 * 1024**2)
    elif failure == "output":
        job.output.write_bytes(b"keep")
    elif failure == "output_suffix":
        job.output = job.output.with_suffix(".avi")
    elif failure == "intermediate":
        job.output.with_suffix(".encoding.mp4").write_bytes(b"keep")
    elif failure == "duration":
        job.capture.declared = 7201
    with pytest.raises(ValueError):
        inference.infer_video(
            job.source,
            job.output,
            job.entry,
            load_experiment(REPO, "v0"),
            confidence=confidence,
            limits=limits,
            device="invalid" if failure == "device" else "cpu",
        )
    loader.assert_not_called()
    assert job.output.exists() == (failure == "output")


def test_conversion_failure_removes_partial_files(video_job, monkeypatch):
    job = video_job

    def failed(src, dst, *args, **kwargs):
        dst.write_bytes(b"partial conversion")
        raise ValueError("FFmpeg conversion failed")

    monkeypatch.setattr(inference, "finish_video", failed)
    with pytest.raises(ValueError, match="FFmpeg"):
        inference.infer_video(job.source, job.output, job.entry, load_experiment(REPO, "v0"))
    assert job.writer.released and job.capture.released
    assert not job.output.exists() and not job.output.with_suffix(".encoding.mp4").exists()


def test_conversion_timeout_can_exceed_two_minutes(tmp_path, monkeypatch):
    intermediate, output = tmp_path / "encoding.mp4", tmp_path / "out.mp4"
    intermediate.write_bytes(b"encoded")

    def convert(command, **kwargs):
        assert kwargs["timeout"] == 900
        simulated_seconds = 121
        if kwargs["timeout"] < simulated_seconds:
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        output.write_bytes(b"converted")

    monkeypatch.setattr(inference.subprocess, "run", convert)
    monkeypatch.setattr(
        inference.cv2,
        "VideoCapture",
        lambda path: SimpleNamespace(
            read=lambda: (True, np.zeros((64, 80, 3), np.uint8)),
            get=lambda prop: {cv2.CAP_PROP_FRAME_COUNT: 3, cv2.CAP_PROP_FPS: 10}[prop],
            release=lambda: None,
        ),
    )
    finish_video(intermediate, output, "mp4v", 3, (80, 64), 10, timeout=900)
    assert output.exists() and not intermediate.exists()


@pytest.mark.parametrize("failure", ["timeout", "codec"])
def test_ffmpeg_errors_report_cause_and_clean_up(video_job, monkeypatch, failure):
    job = video_job
    job.writer.codec = "mp4v"
    monkeypatch.setattr(inference, "finish_video", finish_video)

    def convert(command, **kwargs):
        assert kwargs["timeout"] == 17
        job.output.write_bytes(b"partial conversion")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        raise subprocess.CalledProcessError(1, command, stderr=b"encoder unavailable")

    monkeypatch.setattr(inference.subprocess, "run", convert)
    message = "17 seconds.*--encoding-timeout" if failure == "timeout" else "encoder unavailable"
    with pytest.raises(ValueError, match=message):
        inference.infer_video(
            job.source,
            job.output,
            job.entry,
            load_experiment(REPO, "v0"),
            limits=inference.VideoLimits(encoding_timeout_seconds=17),
        )
    assert job.capture.released and job.writer.released
    assert not job.output.exists() and not job.output.with_suffix(".encoding.mp4").exists()


@pytest.mark.parametrize(
    ("version", "sign_id"),
    [("v0", None), ("v3", None)]
    + [(version, key) for key in model_registry.SIGN_MODELS for version in (None, "v3")],
)
def test_real_mp4_is_h264_decodable_and_accepted_by_gradio(
    registry_root, tmp_path, monkeypatch, version, sign_id
):
    gr = pytest.importorskip("gradio")
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("Real MP4 smoke test requires FFmpeg and ffprobe")
    add_run(registry_root)
    vehicle = inference.discover_models(registry_root)["v0"]
    signs = add_signs(registry_root, monkeypatch)
    monkeypatch.setattr(inference, "load_detector", lambda path, *, names: Detector(names))
    entry = vehicle if version else signs[sign_id]
    source, output = tmp_path / "source.mp4", tmp_path / "out.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (80, 64))
    assert writer.isOpened()
    try:
        for i in range(4):
            writer.write(np.full((64, 80, 3), 40 + i, np.uint8))
    finally:
        writer.release()
    summary = inference.infer_video(
        source,
        output,
        entry,
        load_experiment(REPO, version) if version else None,
        sign_entry=signs[sign_id] if sign_id and version else None,
        device="cpu",
    )
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,pix_fmt,width,height,r_frame_rate,nb_frames",
            "-of",
            "json",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    assert stream == {
        "codec_name": "h264",
        "pix_fmt": "yuv420p",
        "width": 80,
        "height": 64,
        "r_frame_rate": "10/1",
        "nb_frames": "4",
    }
    capture = cv2.VideoCapture(str(output))
    decoded = 0
    try:
        assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(10)
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            assert frame.shape == (64, 80, 3)
            decoded += 1
    finally:
        capture.release()
    assert decoded == summary["frames"] == 4
    if sign_id:
        assert summary["detectors"][-1]["model"]["model_id"] == sign_id
        assert summary["detectors"][-1]["stages"]["jpeg"]["total_seconds"] == 0
    video = gr.Video(interactive=False)
    assert video.postprocess(str(output)).path == str(output)


def test_demo_ui_wiring_and_cleanup(registry_root, monkeypatch):
    gr = pytest.importorskip("gradio")
    add_run(registry_root)
    build_app = runpy.run_path(str(REPO / "apps/gradio_demo.py"))["build_app"]
    app = build_app(registry_root)
    try:
        dropdowns = [block for block in app.blocks.values() if isinstance(block, gr.Dropdown)]
        assert dropdowns[0].choices == [("v0", "v0")]
        assert [value for label, value in dropdowns[1].choices] == list(inference.VERSIONS)
        run = next(fn for fn in app.fns.values() if fn.fn.__name__ == "run_job")
        assert run.concurrency_limit == 1 and run.concurrency_id == "inference"
        captured = {}

        def infer(source, output, entry, policy, **kwargs):
            captured.update(entry=entry, policy=policy, output=output)
            output.write_bytes(b"encoded test video")
            return {"pair": inference.EXPLORATORY}

        monkeypatch.setitem(run.fn.__globals__, "infer_video", infer)
        source, annotated, download, report = run.fn(
            "source.mp4", "v0", "v3", 0.25, "cpu", {"requested": False}
        )
        assert captured["entry"].version == "v0" and captured["policy"]["id"] == "v3"
        assert not captured["output"].exists()  # Work directory removed after cache copy.
        assert Path(annotated).read_bytes() == b"encoded test video" and download == annotated
        assert source == "source.mp4" and report["pair"] == inference.EXPLORATORY
        Path(annotated).unlink()
        warnings = next(fn.fn for fn in app.fns.values() if fn.fn.__name__ == "pair_warning")
        assert inference.EXPLORATORY in warnings("v0", "v3")
        state = {"requested": False}
        next(fn.fn for fn in app.fns.values() if fn.fn.__name__ == "cancel_job")(state)
        assert state["requested"]
    finally:
        app.close()


def test_demo_launches_without_weights(registry_root):
    gr = pytest.importorskip("gradio")
    build_app = runpy.run_path(str(REPO / "apps/gradio_demo.py"))["build_app"]
    app = build_app(registry_root)
    try:
        run = next(
            block
            for block in app.blocks.values()
            if isinstance(block, gr.Button) and block.value == "Run inference"
        )
        assert run.interactive is False
        models = next(
            block
            for block in app.blocks.values()
            if isinstance(block, gr.JSON) and block.label == "Models"
        )
        assert all(entry["unavailable_reason"] for entry in models.value.values())
    finally:
        app.close()


def test_sign_registry_pins_identity_and_independent_taxonomy(registry_root, monkeypatch):
    signs = add_signs(registry_root, monkeypatch)
    assert list(signs) == list(model_registry.SIGN_MODELS)
    for key, spec in signs.items():
        assert not spec.error and spec.category == "traffic-sign"
        assert spec.names == (
            "No U-turn",
            "No going opposite",
            "No stopping",
            "No waiting",
            "Roundabout ahead",
            "Speed limit 60kmh",
        )
        assert spec.imgsz == (832 if key.endswith("hieu") else 640)
        assert spec.preprocessing == "native" and len(spec.source_revision) == 40
        assert spec.sha256 == file_hash(spec.checkpoint)
        assert spec.identity()["model_id"] == key
    assert "runs/detect/runs/detect/" in signs["traffic-sign-phat"].source_checkpoint


@pytest.mark.parametrize(
    "failure", ["missing", "hash", "task", "names", "imgsz", "path", "revision"]
)
def test_bad_sign_registry_entry_does_not_block_other_models(registry_root, monkeypatch, failure):
    signs = add_signs(registry_root, monkeypatch)
    key = "traffic-sign-hieu"
    manifest_path = registry_root / "configs/external_models.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    if failure == "missing":
        signs[key].checkpoint.unlink()
    elif failure == "hash":
        signs[key].checkpoint.write_bytes(b"corrupt")
    else:
        field, value = {
            "task": ("task", "segment"),
            "names": ("names", ["wrong"]),
            "imgsz": ("imgsz", 0),
            "path": ("checkpoint", "../outside.pt"),
            "revision": ("source_revision", "main"),
        }[failure]
        manifest[key][field] = value
        manifest_path.write_text(yaml.safe_dump(manifest))
    loader = Mock(return_value=Detector())
    monkeypatch.setattr(model_registry, "load_detector", loader)
    registry = model_registry.discover_sign_models(registry_root)
    assert "docs/demo.md" in registry[key].error
    assert not registry["traffic-sign-phat"].error
    assert loader.call_count == 1  # Rejected before deserializing Hiếu's bytes.
    add_run(registry_root)
    assert not inference.discover_models(registry_root)["v0"].error


@pytest.mark.parametrize("failure", ["taxonomy", "order", "task", "corrupt"])
def test_sign_checkpoint_loader_validates_actual_model(registry_root, monkeypatch, failure):
    spec = add_signs(registry_root, monkeypatch)["traffic-sign-hieu"]
    detector = Detector(spec.names)
    if failure == "taxonomy":
        detector.names = dict(enumerate(FIVE_CLASSES))
    elif failure == "order":
        detector.names = dict(enumerate(reversed(spec.names)))
    elif failure == "task":
        detector.task = "segment"

    def load(path):
        if failure == "corrupt":
            raise RuntimeError("corrupt checkpoint")
        return detector

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=load))
    with pytest.raises(ValueError, match="taxonomy|corrupt"):
        load_checkpoint(spec.checkpoint, names=spec.names)


@pytest.mark.parametrize("key", model_registry.SIGN_MODELS)
def test_sign_video_uses_native_frames_and_six_class_counts(video_job, monkeypatch, key):
    job = video_job
    spec = add_signs(job.entry.checkpoint.parents[4], monkeypatch)[key]
    model = Detector(spec.names, classes=(0, 5))
    monkeypatch.setattr(inference, "load_detector", lambda path, **kwargs: model)
    monkeypatch.setattr(
        inference, "apply_recipe", Mock(side_effect=AssertionError("Vehicle policy"))
    )
    monkeypatch.setattr(inference.cv2, "imencode", Mock(side_effect=AssertionError("JPEG")))
    summary = inference.infer_video(job.source, job.output, spec, None)
    assert summary["mode"] == "Traffic Sign" and summary["routing"] is None
    assert summary["preprocessing_version"] == "native"
    assert summary["detections_per_class"] == dict.fromkeys(spec.names, 0) | {
        spec.names[0]: 3,
        spec.names[5]: 3,
    }
    assert summary["stages"]["jpeg"]["total_seconds"] == 0
    assert summary["stages"]["preprocessing"]["total_seconds"] == 0
    for settings in model.inputs:
        assert np.all(settings["source"] == 40) and settings["imgsz"] == spec.imgsz
        assert settings["device"] == "cpu" and settings["conf"] == 0.25


@pytest.mark.parametrize("key", model_registry.SIGN_MODELS)
def test_combined_inputs_overlays_counts_and_timings_are_separate(video_job, monkeypatch, key):
    job = video_job
    spec = add_signs(job.entry.checkpoint.parents[4], monkeypatch)[key]
    sign = Detector(spec.names, classes=(0, 5), marker=1)
    job.model.marker = 0
    monkeypatch.setattr(
        inference,
        "load_detector",
        lambda path, **kwargs: job.model if path == job.entry.checkpoint else sign,
    )
    write = job.writer.write

    def check_overlay(frame):
        assert np.all(frame[0, :2] == 255)
        write(frame)

    job.writer.write = check_overlay
    policy = load_experiment(REPO, "v3")
    summary = inference.infer_video(job.source, job.output, job.entry, policy, sign_entry=spec)
    assert summary["mode"] == "Combined" and summary["frames"] == 3
    assert summary["routing"] == {"Apply": 3, "Skip": 0}
    assert len(summary["detectors"]) == 2
    vehicle_report, sign_report = summary["detectors"]
    assert vehicle_report["detections_per_class"]["car"] == 6
    assert sign_report["detections_per_class"][spec.names[5]] == 3
    assert "car" not in sign_report["detections_per_class"]
    assert "routing" not in sign_report and sign_report["stages"]["jpeg"]["total_seconds"] == 0
    assert summary["detections_per_class"][spec.version] == sign_report["detections_per_class"]
    for vehicle_input, sign_input in zip(job.model.inputs, sign.inputs, strict=True):
        assert np.all(sign_input["source"] == 40)
        assert not np.array_equal(vehicle_input["source"], sign_input["source"])
        assert sign_input["imgsz"] == spec.imgsz and vehicle_input["imgsz"] == job.entry.imgsz
    assert job.model.rendered[0]["names"][0] == "Vehicle / car"
    assert sign.rendered[0]["names"][0] == "Sign / No U-turn"
    assert sign.names[0] == "No U-turn" and job.model.names[0] == "car"
    total = sum(item["stages"]["model_inference"]["total_seconds"] for item in summary["detectors"])
    assert summary["stages"]["model_inference"]["total_seconds"] == pytest.approx(total)
    assert summary["end_to_end_fps"] == summary["frames"] / summary["total_seconds"]


@pytest.mark.parametrize("failure", ["cancel", "inference", "changed", "unavailable", "taxonomy"])
def test_combined_failure_cleans_up_and_never_substitutes(video_job, monkeypatch, failure):
    job = video_job
    spec = add_signs(job.entry.checkpoint.parents[4], monkeypatch)["traffic-sign-hieu"]
    sign = Detector(spec.names, classes=(6,) if failure == "taxonomy" else (0, 5))
    if failure == "inference":
        sign.predict = Mock(side_effect=RuntimeError("sign inference failed"))
    if failure == "changed":
        spec.checkpoint.write_bytes(b"changed")
    if failure == "unavailable":
        spec = replace(spec, error="Missing trusted checkpoint")
    monkeypatch.setattr(
        inference,
        "load_detector",
        lambda path, **kwargs: job.model if path == job.entry.checkpoint else sign,
    )
    with pytest.raises((ValueError, RuntimeError)):
        inference.infer_video(
            job.source,
            job.output,
            job.entry,
            load_experiment(REPO, "v0"),
            sign_entry=spec,
            cancelled=lambda: failure == "cancel" and bool(job.model.inputs),
        )
    assert not job.output.exists() and not job.output.with_suffix(".encoding.mp4").exists()
    if failure != "unavailable":
        assert job.capture.released
    if failure in {"cancel", "changed", "unavailable"}:
        assert not sign.inputs


@pytest.mark.parametrize("class_id", [-1, 5, 0.5, float("nan")])
def test_vehicle_class_validation_stays_strict(class_id):
    with pytest.raises(ValueError, match="taxonomy"):
        inference.process_frame(
            np.full((64, 80, 3), 40, np.uint8),
            Detector(classes=(class_id,)),
            load_experiment(REPO, "v0"),
            confidence=0.25,
            imgsz=640,
            device="cpu",
        )


@pytest.mark.parametrize("mode", ["Traffic Sign", "Combined"])
@pytest.mark.parametrize("key", model_registry.SIGN_MODELS)
def test_demo_routes_modes_and_caches_output(registry_root, monkeypatch, mode, key):
    gr = pytest.importorskip("gradio")
    add_run(registry_root)
    add_signs(registry_root, monkeypatch)
    build_app = runpy.run_path(str(REPO / "apps/gradio_demo.py"))["build_app"]
    app = build_app(registry_root)
    try:
        run = next(fn.fn for fn in app.fns.values() if fn.fn.__name__ == "run_job")
        change = next(fn.fn for fn in app.fns.values() if fn.fn.__name__ == "change_mode")
        vehicle_controls, sign_controls, button, warning = change(mode, "v0", "v3")
        assert vehicle_controls["visible"] == (mode == "Combined")
        assert sign_controls["visible"] and button["interactive"]
        assert (inference.EXPLORATORY in warning) == (mode == "Combined")
        captured = {}

        def infer(source, output, entry, policy, **kwargs):
            captured.update(entry=entry, policy=policy, output=output, **kwargs)
            output.write_bytes(b"video")
            return {"mode": mode}

        monkeypatch.setitem(run.__globals__, "infer_video", infer)
        _, annotated, download, report = run(
            "source.mp4", "v0", "v3", 0.25, "cpu", {"requested": False}, mode, key
        )
        assert report["mode"] == mode and annotated == download
        assert Path(annotated).read_bytes() == b"video" and not captured["output"].exists()
        Path(annotated).unlink()
        if mode == "Combined":
            assert captured["entry"].version == "v0" and captured["policy"]["id"] == "v3"
            assert captured["sign_entry"].version == key
        else:
            assert captured["entry"].version == key and captured["policy"] is None
            assert captured["sign_entry"] is None
        gr.JSON().postprocess({"model": captured["entry"].identity()})
    finally:
        app.close()


def test_demo_sign_only_install_and_missing_sign_modes(registry_root, monkeypatch):
    pytest.importorskip("gradio")
    signs = add_signs(registry_root, monkeypatch)
    build_app = runpy.run_path(str(REPO / "apps/gradio_demo.py"))["build_app"]
    app = build_app(registry_root)
    try:
        change = next(fn.fn for fn in app.fns.values() if fn.fn.__name__ == "change_mode")
        assert change("Traffic Sign", None, None)[2]["interactive"]
        assert not change("Combined", None, None)[2]["interactive"]
    finally:
        app.close()
    for spec in signs.values():
        spec.checkpoint.unlink()
    add_run(registry_root)
    app = build_app(registry_root)
    try:
        change = next(fn.fn for fn in app.fns.values() if fn.fn.__name__ == "change_mode")
        assert change("Vehicle", "v0", "v0")[2]["interactive"]
        assert not change("Traffic Sign", "v0", "v0")[2]["interactive"]
    finally:
        app.close()
