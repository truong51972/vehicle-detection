"""Demo contracts with lightweight detectors and incremental video IO stubs."""

import json
import runpy
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

from vehicle_detection import inference
from vehicle_detection.common import file_hash
from vehicle_detection.data import FIVE_CLASSES
from vehicle_detection.experiment_data import config_path, load_experiment
from vehicle_detection.preprocess import apply_recipe

REPO = Path(__file__).resolve().parents[1]


class Tensor:
    def cpu(self):
        return self

    def numpy(self):
        return np.array([0, 4, 0])


class Detector:
    task = "detect"
    names = dict(enumerate(FIVE_CLASSES))

    def __init__(self):
        self.inputs = []
        self.rendered = []

    def predict(self, **kwargs):
        self.inputs.append(kwargs)

        def plot(**settings):
            self.rendered.append(settings)
            return settings["img"]

        return [SimpleNamespace(boxes=SimpleNamespace(cls=Tensor()), plot=plot)]


@pytest.fixture
def registry_root(tmp_path, monkeypatch):
    root = tmp_path / "project"
    for version in inference.VERSIONS:
        path = config_path(root, version)
        path.parent.mkdir(parents=True)
        shutil.copyfile(config_path(REPO, version), path)
    monkeypatch.setattr(inference, "load_detector", lambda path: Detector())
    return root


def add_run(root, version="v0", date="2026-11-09"):
    run = root / f"runs/{version}-{date}"
    weights = run / "train/weights/best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"test checkpoint")
    shutil.copyfile(config_path(root, version), run / "config.yaml")
    return weights


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
    monkeypatch.setattr(inference, "load_detector", lambda path: model)

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
    writer = SimpleNamespace(written=0, released=False)

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
        return writer, "avc1"

    monkeypatch.setattr(inference, "open_writer", open_writer)
    monkeypatch.setattr(inference, "finish_video", lambda src, dst, *args: src.replace(dst))
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


@pytest.mark.parametrize("failure", ["suffix", "empty", "size", "confidence", "pixels"])
def test_video_input_boundaries(video_job, failure):
    job = video_job
    confidence = float("nan") if failure == "confidence" else 0.25
    limits = inference.VideoLimits(max_pixels=100 if failure == "pixels" else 80 * 64)
    if failure == "suffix":
        job.source = job.source.rename(job.source.with_suffix(".txt"))
    elif failure == "empty":
        job.source.write_bytes(b"")
    elif failure == "size":
        with job.source.open("wb") as stream:
            stream.truncate(201 * 1024**2)
    with pytest.raises(ValueError):
        inference.infer_video(
            job.source,
            job.output,
            job.entry,
            load_experiment(REPO, "v0"),
            confidence=confidence,
            limits=limits,
        )
    assert not job.model.inputs and not job.output.exists()


def test_conversion_failure_removes_partial_files(video_job, monkeypatch):
    job = video_job

    def failed(src, dst, *args):
        dst.write_bytes(b"partial conversion")
        raise ValueError("FFmpeg conversion failed")

    monkeypatch.setattr(inference, "finish_video", failed)
    with pytest.raises(ValueError, match="FFmpeg"):
        inference.infer_video(job.source, job.output, job.entry, load_experiment(REPO, "v0"))
    assert job.writer.released and job.capture.released
    assert not job.output.exists() and not job.output.with_suffix(".encoding.mp4").exists()


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
