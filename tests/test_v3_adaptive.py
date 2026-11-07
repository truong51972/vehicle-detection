"""Prospective v3 controls and gates; synthetic checks are not research evidence."""

import json
import subprocess
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from vehicle_detection.common import file_hash, read_json, write_json
from vehicle_detection.five_class import FIVE_CLASSES
from vehicle_detection.v3_adaptive import (
    ARMS,
    EXPOSURES,
    STUDY,
    apply,
    assess,
    local_contrast,
    population,
    read_report,
    route,
    run,
)


def test_bounded_achromatic_operator_and_pixel_only_route():
    image = np.random.default_rng(2026).integers(40, 210, (72, 128, 3), dtype=np.uint8)
    original = image.copy()
    output = local_contrast(image)
    difference = output.astype(int) - image.astype(int)
    assert output.shape == image.shape and output.dtype == np.uint8
    assert np.array_equal(image, original)
    assert np.abs(difference).max() <= 12
    assert np.array_equal(difference[:, :, 0], difference[:, :, 1])
    assert np.array_equal(difference[:, :, 0], difference[:, :, 2])
    for value in (0, 255):
        extreme = np.full_like(image, value)
        assert np.array_equal(local_contrast(extreme), extreme)
        assert np.array_equal(apply(extreme, "adaptive_local"), extreme)
    features = dict(contrast=0.1, brightness=0.2, noise=0, black_fraction=0, white_fraction=0)
    assert route(features)
    for key, value in (
        ("contrast", 0.2),
        ("brightness", 0.04),
        ("brightness", 0.8),
        ("noise", 0.010001),
        ("black_fraction", 0.3),
        ("white_fraction", 0.1),
    ):
        assert not route({**features, key: value})
    with pytest.raises(ValueError, match="BGR uint8"):
        apply(image.astype(float), "adaptive_local")
    with pytest.raises(ValueError, match="Unknown"):
        apply(image, "unplanned")


def test_fresh_population_excludes_older_exposure_and_commits_exact_membership():
    root = Path.cwd()
    rows = population(root)
    declared = pd.read_csv(root / f"docs/{STUDY}/membership.csv.gz", keep_default_na=False)
    assert rows == declared.to_dict("records")
    assert len(rows) == 1479 and len({r["group"] for r in rows}) == 926
    ids = {r["id"] for r in rows}
    # These older sample groups were omitted from the recent pilots' exclusions.
    for name in EXPOSURES:
        if name.endswith(".json"):
            prior = set(read_json(name)["calibration_ids"])
        else:
            prior = set(pd.read_csv(name).id)
        assert not ids & prior
    manifest = read_json("docs/round2-evaluation-v2/evaluation_sets.json")
    assert ids <= set(manifest["sets"]["calibration_exploration"])
    assert not ids & set(manifest["sets"]["calibration_confirmation"])


def passing_evidence():
    record = {
        "mAP50_95": 0.2,
        "mAP50": 0.3,
        "classes": {c: {"AP50_95": 0.2, "source_groups": 30} for c in FIVE_CLASSES},
        "counts": {c: {"FN50": 100} for c in FIVE_CLASSES},
    }
    metrics = {
        a: {s: deepcopy(record) for s in ("all", "dark", "bright", "apply", "skip")} for a in ARMS
    }
    for arm in ("adaptive_local", "always_local"):
        for cohort in ("all", "dark", "bright", "apply"):
            metrics[arm][cohort]["mAP50_95"] = 0.21
            for r in metrics[arm][cohort]["classes"].values():
                r["AP50_95"] = 0.21
    interval = {
        "estimand": {"class_ids": list(range(5))},
        "valid_draws": 1000,
        "recovery_ci_low": 0.005,
        "per_class": [
            {"class": c, "source_groups": 30, "recovery": 0.01, "ci_low": 0.002}
            for c in FIVE_CLASSES
        ],
    }
    return (
        metrics,
        {"adaptive_local": {"mean_ms": 5, "p95_ms": 10}},
        {s: deepcopy(interval) for s in ("all", "dark", "bright")},
    )


def test_every_failed_gate_survives_and_sparse_routes_do_not_require_all_rare_classes():
    metrics, runtime, intervals = passing_evidence()
    metrics["raw"]["skip"]["classes"].pop("motorcycle")
    metrics["adaptive_local"]["skip"]["classes"].pop("motorcycle")
    assert assess(metrics, runtime, intervals)["advance"]
    metrics["adaptive_local"]["all"]["mAP50_95"] = 0.2
    metrics["adaptive_local"]["all"]["mAP50"] = 0.29
    metrics["adaptive_local"]["dark"]["classes"]["bus"]["AP50_95"] = 0.18
    metrics["adaptive_local"]["dark"]["counts"]["bus"]["FN50"] = 110
    intervals["dark"]["per_class"][2]["ci_low"] = -0.01
    intervals["all"]["recovery_ci_low"] = -0.001
    intervals["all"]["per_class"][3].update(recovery=-0.01, ci_low=-0.02)
    metrics["always_local"]["apply"]["classes"]["car"]["AP50_95"] = 0.1
    runtime["adaptive_local"]["mean_ms"] = 11
    result = assess(metrics, runtime, intervals)
    assert not result["advance"]
    for label in (
        "Primary gain",
        "mAP50",
        "uncertainty",
        "two-wheeler",
        "regression",
        "contradiction",
        "predict",
        "always-on",
        "CPU",
    ):
        assert any(label in reason for reason in result["reasons"]), label


def test_default_only_verifies_evidence_and_attempts_cannot_be_overwritten(tmp_path):
    report = tmp_path / f"docs/{STUDY}"
    assert read_report(tmp_path)["status"] == "not acquired"
    write_json(report / "result.json", {"inputs": {}, "policy_frozen": False})
    write_json(report / "evidence.json", {"result.json": file_hash(report / "result.json")})
    assert not read_report(tmp_path)["policy_frozen"]
    with pytest.raises(FileExistsError, match="do not retry"):
        run(tmp_path, allow_inference=True)
    write_json(report / "result.json", {})
    with pytest.raises(ValueError, match="evidence changed"):
        read_report(tmp_path)


def test_actual_native_acquisition_on_synthetic_cpu_images(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("ultralytics")
    from ultralytics.nn.tasks import DetectionModel, yaml_model_load

    import vehicle_detection.v3_adaptive as study
    from vehicle_detection.baseline import VAL_SETTINGS
    from vehicle_detection.evaluation_stats import paired_interval

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=fixture",
            "-c",
            "user.email=fixture@example.test",
            "commit",
            "--allow-empty",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    monkeypatch.setitem(VAL_SETTINGS, "device", "cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "synthetic CPU fixture")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    preservation = {"tracked": {}, "local": {}}
    monkeypatch.setattr(study, "inventory", lambda root: preservation)

    def small_interval(*args, **kwargs):
        return paired_interval(*args, **{**kwargs, "resamples": 4, "min_class_groups": 1})

    monkeypatch.setattr("vehicle_detection.evaluation_stats.paired_interval", small_interval)
    checkpoint = tmp_path / "fixture.pt"
    model = DetectionModel(yaml_model_load("yolo26n.yaml"), nc=5, verbose=False)
    model.names = dict(enumerate(FIVE_CLASSES))
    torch.save({"model": model, "train_args": {"task": "detect"}}, checkpoint)
    monkeypatch.setattr(study, "CHECKPOINT", file_hash(checkpoint))
    data = tmp_path / "sources"
    data.mkdir()
    rows = []
    for i, value in enumerate((20, 120)):
        source = data / f"fixture{i}.jpg"
        assert cv2.imwrite(str(source), np.full((32, 64, 3), value, dtype=np.uint8))
        rows.append(
            {
                "id": f"fixture{i}",
                "source_path": source.name,
                "sha256": file_hash(source),
                "width": 64,
                "height": 32,
                "source_group": f"group{i}",
                "group": f"group{i}",
                "boxes": [[c, 4 + c * 10, 8, 12 + c * 10, 20] for c in range(5)],
            }
        )
    result = study.acquire(tmp_path, rows, {"data_root": "sources"}, checkpoint, {}, preservation)
    assert result["inference_image_passes"] == 6
    assert not result["decision"]["advance"]
    assert set(result["statistics"]) == set(ARMS)
    assert all(v["tensor_shapes"] for v in result["measurements"].values())
    assert read_report(tmp_path) == json.loads(json.dumps(result))
