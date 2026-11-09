"""New post-hoc safeguards; prior frozen policy/tests remain unchanged."""

import hashlib
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from vehicle_detection.common import file_hash, read_json, write_json
from vehicle_detection.posthoc_training import (
    ADMIN_ARGS,
    STUDY,
    compare_args,
    contrasts,
    evaluate,
    export_row,
    prepare,
    read_report,
    train,
)
from vehicle_detection.v3_adaptive import apply


def test_training_recipe_allows_only_data_and_output_path_changes():
    reference = yaml.safe_load(
        Path("docs/issue47-s20-resolution-20e/train960_args.yaml").read_text()
    )
    actual = {**reference, **dict.fromkeys(ADMIN_ARGS, "owned-posthoc-path")}
    compare_args(actual, reference)
    for key, value in (("seed", 7), ("imgsz", 640), ("epochs", 19), ("hsv_v", 0.1), ("amp", True)):
        with pytest.raises(ValueError, match="arguments differ"):
            compare_args({**actual, key: value}, reference)


@pytest.mark.parametrize("level", [40, 255])
def test_exports_start_from_original_pixels_keep_exact_labels_and_raw_bytes(tmp_path, level):
    source = tmp_path / "sources"
    source.mkdir()
    image = np.full((64, 96, 3), level, np.uint8)
    assert cv2.imwrite(str(source / "one.jpg"), image)
    original = cv2.imread(str(source / "one.jpg"))
    raw = tmp_path / "historical"
    output = tmp_path / "owned"
    for domain in ("raw", "adaptive"):
        for kind in ("images", "labels"):
            (output / f"data/{domain}/{kind}/val").mkdir(parents=True)
    for kind in ("images", "labels"):
        (raw / f"{kind}/val").mkdir(parents=True)
    assert cv2.imwrite(str(raw / "images/val/one.jpg"), original, [cv2.IMWRITE_JPEG_QUALITY, 95])
    (raw / "labels/val/one.txt").write_bytes(b"")
    row = {
        "id": "one",
        "source_path": "one.jpg",
        "sha256": file_hash(source / "one.jpg"),
        "width": 96,
        "height": 64,
        "boxes": [],
    }
    expected = (file_hash(raw / "images/val/one.jpg"), file_hash(raw / "labels/val/one.txt"))
    record, _ = export_row(row, "val", source, raw, output, expected)
    ok, encoded = cv2.imencode(
        ".jpg", apply(original, "adaptive_local"), [cv2.IMWRITE_JPEG_QUALITY, 95]
    )
    assert ok and record[3] == hashlib.sha256(encoded.tobytes()).hexdigest()
    assert file_hash(output / "data/raw/images/val/one.jpg") == expected[0]
    assert (output / "data/adaptive/labels/val/one.txt").read_bytes() == b""
    assert record[5] == (level == 40)
    if level == 255:
        assert record[2] == record[3]
    with pytest.raises(ValueError, match="Raw export changed"):
        export_row(row, "val", source, raw, output, ("bad", expected[1]))
    with pytest.raises(ValueError, match="source path/bytes"):
        export_row({**row, "sha256": "bad"}, "val", source, raw, output, expected)


def test_fixed_matrix_contrasts_do_not_confuse_joint_pipeline_with_training():
    metrics = {
        k: {"all": {"mAP50_95": v}}
        for k, v in zip(
            ("raw/raw", "raw/adaptive", "adaptive/raw", "adaptive/adaptive"),
            (0.1, 0.12, 0.11, 0.15),
            strict=True,
        )
    }
    result = contrasts(metrics)
    assert result["adaptive_training_on_adaptive"] == pytest.approx(0.03)
    assert result["adaptive_training_on_raw"] == pytest.approx(0.01)
    assert result["joint_pipeline"] == pytest.approx(0.05)
    assert result["interaction"] == pytest.approx(0.02)


def test_defaults_are_read_only_and_evidence_tampering_is_rejected(tmp_path):
    assert read_report(tmp_path)["status"] == "not acquired"
    assert prepare(tmp_path) is train(tmp_path) is evaluate(tmp_path) is None
    report = tmp_path / f"docs/{STUDY}"
    write_json(report / "result.json", {"status": "completed", "scope": "post-hoc"})
    write_json(report / "evidence.json", {"result.json": file_hash(report / "result.json")})
    assert read_report(tmp_path)["status"] == "completed"
    write_json(report / "result.json", {})
    with pytest.raises(ValueError, match="evidence changed"):
        read_report(tmp_path)


def test_new_notebook_default_in_a_fresh_kernel_cannot_acquire(tmp_path):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    from vehicle_detection.posthoc_training import NOTEBOOK, PLAN

    root = tmp_path / "project"
    (root / "docs").mkdir(parents=True)
    (root / "pyproject.toml").touch()
    shutil.copyfile(PLAN, root / PLAN)
    if Path(f"docs/{STUDY}/evidence.json").exists():
        shutil.copytree(f"docs/{STUDY}", root / f"docs/{STUDY}")
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    notebook.cells.insert(
        1,
        nbformat.v4.new_code_cell("""import sys
from unittest.mock import patch
sys.modules["torch"] = None
sys.modules["ultralytics"] = None
for name in ("frozen_inputs", "prepared_inputs", "export_row"):
    patch("vehicle_detection.posthoc_training." + name,
          side_effect=AssertionError("Acquisition started")).start()
"""),
    )
    nbclient.NotebookClient(
        notebook, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(root)}}
    ).execute()
    assert all(c.execution_count is not None for c in notebook.cells if c.cell_type == "code")
    assert not any((root / name).exists() for name in ("runs", "data", "weights"))


def test_training_uses_pinned_initialization_native_recipe_and_saves_failed_cost(
    tmp_path, monkeypatch
):
    torch = pytest.importorskip("torch")
    ultralytics = pytest.importorskip("ultralytics")
    import vehicle_detection.posthoc_training as study
    from vehicle_detection.baseline import COCO_TO_VEHICLE
    from vehicle_detection.data import CLASSES

    checkpoint = tmp_path / "weights/yolo26n.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"synthetic initialization identity")
    monkeypatch.setattr("vehicle_detection.baseline.PRETRAINED_SHA256", file_hash(checkpoint))
    monkeypatch.setattr(
        study, "prepared_inputs", lambda root: {"train": [{"id": "train"}], "val": [{"id": "val"}]}
    )
    monkeypatch.setattr(study, "install_selection", lambda *args: {})
    monkeypatch.setattr("vehicle_detection.finetuning.backbone_boundary", lambda model: 11)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "fixture")
    for name in (
        "empty_cache",
        "reset_peak_memory_stats",
        "max_memory_allocated",
        "max_memory_reserved",
    ):
        monkeypatch.setattr(torch.cuda, name, lambda: 0)
    calls = []

    class Model:
        def __init__(self, path):
            calls.append(path)
            self.names = dict.fromkeys(range(80), "other")
            self.names.update({c: CLASSES[b] for c, b in COCO_TO_VEHICLE.items()})
            self.model = None

        def train(self, **kwargs):
            calls.append(kwargs)
            raise RuntimeError("fixture stops before optimization")

    monkeypatch.setattr(ultralytics, "YOLO", Model)
    with pytest.raises(RuntimeError, match="fixture stops"):
        train(tmp_path, allow_training=True)
    assert calls[0] == str(checkpoint)
    assert {
        k: calls[1][k] for k in ("epochs", "imgsz", "batch", "seed", "freeze", "amp", "hsv_v")
    } == {
        "epochs": 20,
        "imgsz": 960,
        "batch": 16,
        "seed": 2026,
        "freeze": 11,
        "amp": False,
        "hsv_v": 0,
    }
    assert read_json(tmp_path / f"docs/{STUDY}/training_measurement.json")["status"] == "incomplete"
    with pytest.raises(FileExistsError, match="Attempt exists"):
        train(tmp_path, allow_training=True)


def test_four_cell_native_matrix_on_synthetic_cpu_data(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("ultralytics")
    from ultralytics.nn.tasks import DetectionModel, yaml_model_load

    import vehicle_detection.posthoc_training as study
    from vehicle_detection.baseline import VAL_SETTINGS
    from vehicle_detection.five_class import FIVE_CLASSES
    from vehicle_detection.method_discovery import cohorts, outcome
    from vehicle_detection.preprocess import quality

    output = tmp_path / f"runs/{STUDY}"
    report = tmp_path / f"docs/{STUDY}"
    old_report = tmp_path / f"docs/{study.BASELINE}"
    rows, features = [], {}
    for i, level in enumerate((20, 120)):
        image = np.full((32, 64, 3), level, np.uint8)
        row = {
            "id": f"image{i}",
            "group": f"g{i}",
            "source_group": f"g{i}",
            "width": 64,
            "height": 32,
            "boxes": [[c, 4 + c * 10, 8, 12 + c * 10, 20] for c in range(5)],
        }
        rows.append(row)
        features[row["id"]] = quality(image)
        for domain in ("raw", "adaptive"):
            directory = output / f"data/{domain}"
            for kind in ("images", "labels"):
                (directory / f"{kind}/val").mkdir(parents=True, exist_ok=True)
            assert cv2.imwrite(str(directory / f"images/val/{row['id']}.jpg"), image)
            (directory / f"labels/val/{row['id']}.txt").write_text(
                "".join(
                    f"{c} {(x1 + x2) / 128:g} {(y1 + y2) / 64:g} "
                    f"{(x2 - x1) / 64:g} {(y2 - y1) / 32:g}\n"
                    for c, x1, y1, x2, y2 in row["boxes"]
                )
            )
            (directory / "data.yaml").write_text(
                yaml.safe_dump(
                    {
                        "path": str(directory),
                        "train": "images/val",
                        "val": "images/val",
                        "names": dict(enumerate(FIVE_CLASSES)),
                    }
                )
            )
    raw = tmp_path / f"runs/{study.BASELINE}/train960/weights/best.pt"
    raw.parent.mkdir(parents=True)
    model = DetectionModel(yaml_model_load("yolo26n.yaml"), nc=5, verbose=False)
    model.names = dict(enumerate(FIVE_CLASSES))
    torch.save({"model": model, "train_args": {"task": "detect"}}, raw)
    adaptive = output / "train/weights/best.pt"
    adaptive.parent.mkdir(parents=True)
    shutil.copyfile(raw, adaptive)
    monkeypatch.setattr(study, "CHECKPOINT", file_hash(raw))
    monkeypatch.setattr(study, "prepared_inputs", lambda root: {"train": rows, "val": rows})
    monkeypatch.setattr(study, "preservation", lambda root: {})
    monkeypatch.setitem(VAL_SETTINGS, "device", "cpu")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    write_json(
        report / "checkpoint.json",
        {"selected_epoch": 1, "checkpoint_hashes": {"best": file_hash(adaptive)}},
    )
    write_json(
        output / "trained.json",
        {"checkpoint_receipt_sha256": file_hash(report / "checkpoint.json")},
    )
    write_json(output / "features.json", features)
    write_json(report / "data.json", {})
    write_json(report / "training_measurement.json", {"wall_seconds": 1})
    write_json(old_report / "train960_measurement.json", {"wall_seconds": 1})
    empty = {
        r["id"]: {
            "target_cls": np.arange(5),
            "pred_cls": np.empty(0),
            "conf": np.empty(0),
            "tp": np.zeros((0, 10), dtype=bool),
        }
        for r in rows
    }
    reference = {
        k: outcome(empty, rs)
        for k, rs in cohorts(rows, features).items()
        if k in ("all", "dark", "bright", "small_two_wheeler")
    }
    write_json(old_report / "eval960_960_metrics.json", reference)
    result = evaluate(tmp_path, allow_inference=True)
    assert result["status"] == "completed" and "Post-hoc" in result["scope"]
    assert list(result["metrics"]) == [
        "raw/raw",
        "raw/adaptive",
        "adaptive/raw",
        "adaptive/adaptive",
    ]
    assert all(v == 0 for v in result["contrasts"].values())
    assert not result["confirmation_accessed"] and not result["final_test_accessed"]
    assert read_report(tmp_path)["status"] == "completed"
