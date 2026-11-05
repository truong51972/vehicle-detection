"""Boundaries, fixed hypotheses and the early-stop gate for the new research round."""

import cv2
import numpy as np
import pytest

from vehicle_detection.common import digest
from vehicle_detection.preprocessing_experiments import apply_candidate
from vehicle_detection.research_round3 import assess, configs_from_evidence, exploration_access


def test_noise_range_has_strict_boundaries_and_severe_noise_skip(monkeypatch):
    records = [
        {"scenario": scenario, "noise_highpass": noise}
        for scenario, noise in (
            ("clean", 0.01),
            ("synthetic_shot_read_s2", 0.03),
            ("synthetic_shot_read_s3", 0.05),
        )
    ]
    config = configs_from_evidence(
        {"records": records, "checksum": digest(records)}, {"selected": {"config": {"steps": []}}}
    )["noise_range"]
    import vehicle_detection.preprocessing_experiments as experiment

    pixels = np.zeros((10, 10, 3), np.uint8)
    monkeypatch.setattr(experiment, "restore", lambda x, *args: x + 1)
    for noise, triggered in ((0.01, False), (0.010001, True), (0.04, True), (0.040001, False)):
        monkeypatch.setattr(
            experiment, "diagnostics", lambda x, noise=noise: {"noise_highpass": noise}
        )
        output, steps = apply_candidate(pixels, config, {"recipes": {}})
        assert bool(steps) == triggered
        assert bool(output.max()) == triggered
    with pytest.raises(ValueError, match="checksum"):
        configs_from_evidence({"records": records, "checksum": "wrong"}, {})


def test_access_guard_blocks_other_sources_writes_and_reserved_plan(tmp_path):
    source = tmp_path / "data"
    source.mkdir()
    allowed, other = source / "allowed.jpg", source / "other.jpg"
    pixels = np.zeros((10, 10, 3), np.uint8)
    cv2.imwrite(str(allowed), pixels)
    cv2.imwrite(str(other), pixels)
    pilot = {"rows": [{"source_path": allowed.name}]}
    with exploration_access(source, pilot) as reads:
        assert allowed.read_bytes()
        assert cv2.imread(str(allowed)) is not None
        for action in (
            other.read_bytes,
            lambda: cv2.imread(str(other)),
            lambda: allowed.write_bytes(b"changed"),
        ):
            with pytest.raises(ValueError, match="exploration"):
                action()
        from pathlib import Path

        with pytest.raises(ValueError, match="Reserved"):
            Path("runs/round2-quality-v2/confirmation_plan.json").read_text()
        assert reads == {allowed.name}
    assert other.read_bytes() == allowed.read_bytes()


def test_advancement_requires_both_controls_and_all_regression_gates():
    scenarios = [{"id": "clean"}, {"id": "noise"}]
    records = [
        {
            "candidate": name,
            "scenario": key,
            "family": key,
            "mAP50_95": score,
            "classes": [{"AP50_95": score}],
        }
        for name, score in (
            ("skip", 0.2),
            ("frozen_v2", 0.201),
            ("pr38_locked", 0.2015),
            ("bilateral", 0.21),
            ("noise_range", 0.22),
        )
        for key in ("clean", "noise")
    ]
    for row in records:
        if row["candidate"] == "bilateral" and row["scenario"] == "clean":
            row["classes"] = [{"AP50_95": 0.17}]
    result = {r["candidate"]: r for r in assess(records, scenarios)["candidates"]}
    assert not result["pr38_locked"]["advance"]
    assert not result["bilateral"]["advance"]
    assert "Class regression" in result["bilateral"]["reasons"][0]
    assert result["noise_range"]["advance"]


def test_source_group_family_interval_pairs_draws_before_family_average():
    pytest.importorskip("ultralytics")
    from vehicle_detection.research_round3_report import family_interval

    scenarios = [
        {"id": "clean", "family": "clean"},
        {"id": "mild", "family": "noise"},
        {"id": "severe", "family": "noise"},
    ]
    intervals = {
        key: {"recovery": gain, "draws": [{"valid": True, "recovery": gain} for _ in range(100)]}
        for key, gain in (("clean", 0.02), ("mild", 0.06), ("severe", -0.02))
    }
    # An absent rare class in any scenario invalidates the entire paired family draw.
    intervals["mild"]["draws"][0] = {"valid": False, "recovery": None}
    result = family_interval(intervals, scenarios)
    assert result["recovery"] == pytest.approx(0.02)
    assert result["ci95"] == pytest.approx([0.02, 0.02])
    assert result["valid_draws"] == 99 and result["invalid_draws"] == 1
    assert result["draws"][0]["missing_scenarios"] == ["mild"]
