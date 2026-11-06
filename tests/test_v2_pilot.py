"""Pilot safety and scientific gates; synthetic examples are not research evidence."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vehicle_detection.common import digest, file_hash, read_json, write_json
from vehicle_detection.experiment_data import shared_splits
from vehicle_detection.five_class import FIVE_CLASSES
from vehicle_detection.v2_pilot import (
    ARMS,
    assess,
    local_shadow,
    require_dark_support,
    run,
    select_sample,
)


def test_local_shadow_is_bounded_deterministic_and_preserves_bright_pixels():
    image = np.full((60, 90, 3), 16, dtype=np.uint8)
    original = image.copy()
    result = local_shadow(image)
    assert np.array_equal(image, original)
    assert result.shape == image.shape and result.dtype == image.dtype
    assert np.array_equal(result, local_shadow(image))
    assert np.all(result == 31)
    for value in (0, 128, 255):
        flat = np.full_like(image, value)
        assert np.array_equal(local_shadow(flat), flat)
    with pytest.raises(ValueError, match="BGR uint8"):
        local_shadow(image.astype(float))


def test_sample_is_whole_group_and_disjoint_from_prior_and_confirmation():
    _, rows = shared_splits(Path.cwd())
    manifest = read_json("docs/round2-evaluation-v2/evaluation_sets.json")
    consumed = pd.read_csv("docs/round2-research-v3/membership.csv.gz")
    selected = select_sample(rows, manifest, consumed)
    assert len(selected) == 256
    ids = {r["id"] for r in selected}
    groups = {r["group"] for r in selected}
    assert not ids & set(consumed.id)
    assert ids <= set(manifest["sets"]["calibration_exploration"])
    assert ids == {r["id"] for r in rows if r["group"] in groups}
    bad = deepcopy(manifest)
    bad["seed"] = 1
    with pytest.raises(ValueError, match="manifest identity"):
        select_sample(rows, bad, consumed)
    confirmation_ids = set(manifest["sets"]["calibration_confirmation"])
    confirmation = next(r for r in rows if r["id"] in confirmation_ids)
    selected[0]["source_group"] = confirmation["source_group"]
    with pytest.raises(ValueError, match="leaks confirmation"):
        select_sample(rows, manifest, consumed)


def gate_inputs():
    record = {
        "mAP50_95": 0.2,
        "classes": {c: {"AP50_95": 0.2, "source_groups": 10} for c in FIVE_CLASSES},
    }
    metrics = {name: {key: deepcopy(record) for key in ("all", "dark", "bright")} for name in ARMS}
    for value in metrics["local_shadow"].values():
        value["mAP50_95"] = 0.21
        for cls in value["classes"].values():
            cls["AP50_95"] = 0.21
    runtime = {"local_shadow": {"mean_ms": 5, "p95_ms": 10}}
    intervals = {
        name: {"recovery_ci_low": 0.001, "estimand": {"class_ids": list(range(5))}}
        for name in ARMS[:2]
    }
    return metrics, runtime, intervals


def test_advancement_requires_gain_support_regression_cost_and_uncertainty():
    metrics, runtime, intervals = gate_inputs()
    assert assess(metrics, runtime, intervals)["advance"]
    metrics["local_shadow"]["all"]["mAP50_95"] = 0.201
    metrics["local_shadow"]["dark"]["classes"]["motorcycle"]["source_groups"] = 1
    metrics["local_shadow"]["bright"]["classes"]["car"]["AP50_95"] = 0.18
    runtime["local_shadow"]["p95_ms"] = 26
    intervals["raw"]["recovery_ci_low"] = 0
    result = assess(metrics, runtime, intervals)
    assert not result["advance"] and not result["v2_trained"]
    for reason in (
        "Primary gain",
        "Insufficient five-class",
        "Class regression",
        "cost exceeds",
        "interval includes zero",
    ):
        assert any(reason in r for r in result["reasons"])


def test_default_only_reads_hash_verified_evidence_and_opt_in_never_overwrites(
    tmp_path, monkeypatch
):
    import builtins

    report = tmp_path / "docs/issue45"
    write_json(report / "summary.json", {"decision": {"advance": False}})
    write_json(report / "evidence.json", {"summary.json": file_hash(report / "summary.json")})
    assert run(tmp_path)["decision"]["advance"] is False
    write_json(report / "summary.json", {"decision": {"advance": True}})
    with pytest.raises(ValueError, match="evidence changed"):
        run(tmp_path)
    (tmp_path / "runs/issue45/pilot").mkdir(parents=True)
    original_import = builtins.__import__

    def lightweight_import(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "ultralytics"}:
            raise ModuleNotFoundError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", lightweight_import)
    with pytest.raises(FileExistsError, match="output exists"):
        run(tmp_path, allow_inference=True)


def test_future_pilot_rejects_untestable_dark_gate_before_inference():
    rows = [
        {"id": str(i), "source_group": str(i), "boxes": [[c, 0, 0, 10, 10] for c in range(5)]}
        for i in range(5)
    ]
    features = {r["id"]: {"brightness": 0.1} for r in rows}
    assert set(require_dark_support(rows, features).values()) == {5}
    rows[0]["boxes"].pop()
    with pytest.raises(ValueError, match="Insufficient dark class support before inference"):
        require_dark_support(rows, features)


def test_sample_rejects_a_rehashed_cross_role_partition():
    rows = [{"role": "development_validation", "id": str(i)} for i in range(256)]
    manifest = {
        "seed": 2026,
        "sets": {"calibration_exploration": ["0"], "calibration_confirmation": []},
    }
    manifest["manifest_id"] = digest(manifest)
    with pytest.raises(ValueError, match="role partition"):
        select_sample(rows, manifest, pd.DataFrame(columns=["id", "group", "source_group"]))
