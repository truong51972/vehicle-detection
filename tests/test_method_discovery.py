"""Discovery counts preserve GT and distinguish image cohorts from box-size AP."""

from copy import deepcopy

import numpy as np
import pytest

from vehicle_detection.common import file_hash, write_json
from vehicle_detection.method_discovery import (
    assess_resolution,
    cohorts,
    outcome,
    read_report,
    resolution,
    size_support,
)


def population():
    rows = [
        {
            "id": "a",
            "source_group": "g",
            "width": 1280,
            "height": 720,
            "boxes": [[3, 0, 0, 20, 20], [3, 0, 0, 80, 80]],
        }
    ]
    features = {
        "a": {"brightness": 0.2, "contrast": 0.3, "noise": 0, "sharpness": 200, "black_fraction": 0}
    }
    return rows, features


def test_size_support_and_cohorts_keep_all_original_gt():
    rows, features = population()
    groups = cohorts(rows, features)
    assert groups["small_two_wheeler"] == rows
    assert len(groups["small_two_wheeler"][0]["boxes"]) == 2
    assert groups["dark"] == [] and groups["bright"] == rows
    record = size_support(rows, features)
    assert record["motorcycle"]["median_sqrt_area_at_640"] == 25
    assert record["motorcycle"]["below_16_pixels_at_640"] == 1
    assert record["motorcycle"]["routes"] == {"skip": 1}
    assert record["bicycle"]["median_sqrt_area_at_640"] is None


def test_native_class_counts_do_not_assign_misses_to_individual_box_sizes():
    pytest.importorskip("ultralytics")
    rows, _ = population()
    stats = {
        "a": {
            "tp": np.zeros((4, 10), dtype=bool),
            "conf": np.array([0.9, 0.8, 0.1, 0.9]),
            "pred_cls": np.array([3, 3, 3, 4]),
            "target_cls": np.array([3, 3]),
        }
    }
    stats["a"]["tp"][0] = True
    result = outcome(stats, rows)
    assert result["counts"]["motorcycle"] == {"GT": 2, "TP50": 1, "FP50": 1, "FN50": 1}
    assert result["counts"]["bicycle"] == {"GT": 0, "TP50": 0, "FP50": 1, "FN50": 0}
    assert "bicycle" not in result["classes"]
    stats["a"]["target_cls"] = np.array([3])
    with pytest.raises(ValueError, match="GT and native class support differ"):
        outcome(stats, rows)


def test_default_reads_only_verified_evidence_and_never_repeats_an_attempt(tmp_path):
    report = tmp_path / "docs/issue47-discovery"
    write_json(report / "diagnosis.json", {"individual_missed_gt_box_sizes_available": False})
    write_json(report / "resolution.json", {"policy_selected": False})
    write_json(
        report / "evidence.json",
        {name: file_hash(report / name) for name in ("diagnosis.json", "resolution.json")},
    )
    assert not read_report(tmp_path)["individual_missed_gt_box_sizes_available"]
    assert not resolution(tmp_path)["policy_selected"]
    with pytest.raises(FileExistsError, match="attempt exists"):
        resolution(tmp_path, allow_inference=True)
    write_json(report / "resolution.json", {})
    with pytest.raises(ValueError, match="evidence changed"):
        read_report(tmp_path)


def test_resolution_gates_preserve_support_regression_uncertainty_and_cost():
    record = {
        "mAP50_95": 0.2,
        "classes": {
            c: {"AP50_95": 0.2, "source_groups": 5}
            for c in ("car", "truck", "bus", "motorcycle", "bicycle")
        },
    }
    metrics = {
        size: {s: deepcopy(record) for s in ("all", "dark", "bright")} for size in ("640", "960")
    }
    for r in metrics["960"].values():
        r["mAP50_95"] = 0.21
        for cls in r["classes"].values():
            cls["AP50_95"] = 0.21
    result = {
        "metrics": metrics,
        "baseline_reproduced": True,
        "paired_interval": {"estimand": {"class_ids": list(range(5))}, "recovery_ci_low": 0.001},
        "timings": {s: {"native_ms_per_image": {"inference": 1}} for s in ("640", "960")},
    }
    assert assess_resolution(result)["actionable_resolution_signal"]
    result["baseline_reproduced"] = False
    result["metrics"]["960"]["dark"]["classes"]["truck"]["AP50_95"] = 0.18
    result["paired_interval"]["recovery_ci_low"] = 0
    result["timings"]["960"]["native_ms_per_image"]["inference"] = 4
    decision = assess_resolution(result)
    assert not decision["actionable_resolution_signal"]
    assert not decision["preprocessing_policy_selected"]
    assert len(decision["reasons"]) == 4
