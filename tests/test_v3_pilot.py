"""Fixed-route and leakage checks; fixtures are not scientific evidence."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vehicle_detection.common import file_hash, read_json, write_json
from vehicle_detection.experiment_data import shared_splits
from vehicle_detection.five_class import FIVE_CLASSES
from vehicle_detection.preprocess import gamma, quality, sharpen
from vehicle_detection.v3_pilot import ARMS, apply, assess, route, run, sample


def test_original_pixel_routes_and_strict_boundaries():
    features = dict(brightness=0.1, contrast=0.1, black_fraction=0, noise=0, sharpness=100)
    assert route(features) == "both"
    for key, value in (("brightness", 0.18), ("contrast", 0.20), ("black_fraction", 0.10)):
        assert route({**features, key: value}) == "detail"
    assert route({**features, "noise": 0.010001}) == "skip"
    assert route({**features, "noise": 0.01}) == "both"
    assert route({**features, "sharpness": 49.999}) == "exposure"
    assert route({**features, "sharpness": 50}) == "both"
    assert route({**features, "sharpness": 150}) == "exposure"


def test_components_use_same_routes_and_order_preserving_geometry_and_source():
    rng = np.random.default_rng(2026)
    image = np.clip(rng.normal(30, 3, (64, 64, 3)), 0, 255).astype(np.uint8)
    original = image.copy()
    assert route(quality(image)) == "both"
    assert np.array_equal(apply(image, "exposure"), gamma(image, 0.85))
    assert np.array_equal(apply(image, "detail"), sharpen(image, 0.15, 1))
    result = apply(image, "routed")
    assert np.array_equal(result, sharpen(gamma(image, 0.85), 0.15, 1))
    assert result.shape == image.shape and result.dtype == image.dtype
    assert np.array_equal(image, original)
    bright = np.full_like(image, 220)
    assert all(np.array_equal(apply(bright, a), bright) for a in ARMS)
    with pytest.raises(ValueError, match="BGR uint8"):
        apply(image.astype(float), "routed")
    with pytest.raises(ValueError, match="Unknown pilot arm"):
        apply(image, "other")


def test_next_four_whole_group_prefixes_exclude_both_consumed_rounds():
    _, rows = shared_splits(Path.cwd())
    manifest = read_json("docs/round2-evaluation-v2/evaluation_sets.json")
    consumed = pd.concat(
        [
            pd.read_csv("docs/round2-research-v3/membership.csv.gz"),
            pd.read_csv("docs/issue45/membership.csv"),
        ]
    )
    selected = sample(rows, manifest, consumed)
    assert 1024 <= len(selected) <= 1088
    ids = {r["id"] for r in selected}
    assert len(ids) == len(selected) and not ids & set(consumed.id)
    assert ids <= set(manifest["sets"]["calibration_exploration"])
    for key in ("group", "source_group", "sha256"):
        assert not {r[key] for r in selected} & set(consumed[key])
    groups = {r["group"] for r in selected}
    assert ids == {r["id"] for r in rows if r["group"] in groups}


def test_advancement_cannot_hide_missing_route_support_pair_failure_or_cost():
    record = {
        "mAP50_95": 0.2,
        "classes": {c: {"AP50_95": 0.2, "source_groups": 5} for c in FIVE_CLASSES},
    }
    metrics = {
        a: {s: deepcopy(record) for s in ("all", "exposure", "detail", "both")} for a in ARMS
    }
    for r in metrics["routed"].values():
        r["mAP50_95"] = 0.21
        for cls in r["classes"].values():
            cls["AP50_95"] = 0.21
    runtime = {"routed": {"mean_ms": 5, "p95_ms": 10}}
    interval = {"estimand": {"class_ids": list(range(5))}, "recovery_ci_low": 0.001}
    assert assess(metrics, runtime, interval)["advance"]
    metrics["routed"]["exposure"]["classes"].pop("motorcycle")
    metrics["routed"]["both"]["mAP50_95"] = 0.2
    runtime["routed"]["mean_ms"] = 11
    interval["recovery_ci_low"] = 0
    result = assess(metrics, runtime, interval)
    assert not result["advance"] and not result["policy_frozen"] and not result["v3_trained"]
    for reason in ("support", "Route benefit", "interval", "Preprocessing"):
        assert any(reason in r for r in result["reasons"])


def test_default_verifies_compact_evidence_and_existing_attempt_blocks_imports(tmp_path):
    report = tmp_path / "docs/issue47"
    write_json(report / "summary.json", {"decision": {"advance": False}})
    write_json(report / "evidence.json", {"summary.json": file_hash(report / "summary.json")})
    assert not run(tmp_path)["decision"]["advance"]
    with pytest.raises(FileExistsError, match="attempt exists"):
        run(tmp_path, allow_inference=True)
    write_json(report / "summary.json", {})
    with pytest.raises(ValueError, match="evidence changed"):
        run(tmp_path)
