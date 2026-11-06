"""CPU-only reporting continuation reuses fixed native analysis and immutable receipts."""

import pytest

from vehicle_detection.common import read_json, write_json
from vehicle_detection.full_resolution_report import cohort_interval
from vehicle_detection.full_resolution_training import STUDY


def test_cpu_cohort_keeps_1000_source_group_draws_and_reuses_only_verified_receipts(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    pytest.importorskip("ultralytics")
    from vehicle_detection import evaluation_stats, method_discovery

    output = tmp_path / f"runs/{STUDY}"
    rows = [{"id": "one", "group": "leakage", "source_group": "original"}]
    write_json(output / "rows.json", {"val": rows})
    write_json(output / "features.json", {})
    monkeypatch.setattr(method_discovery, "cohorts", lambda *args: {"dark": rows})
    monkeypatch.setattr(evaluation_stats, "load_stats", lambda path: {"one": {}})
    calls = []

    def paired(before, after, population, **kwargs):
        calls.append((population, kwargs))
        return {"resamples": 1000, "valid_draws": 999, "invalid_draws": 1}

    monkeypatch.setattr(evaluation_stats, "paired_interval", paired)
    identity = {"arrays": "frozen hash"}
    name, interval = cohort_interval(tmp_path, "dark", identity)
    assert name == "dark" and interval["invalid_draws"] == 1
    assert calls[0][0]["rows"][0]["group"] == "original"
    assert calls[0][1]["resamples"] == 1000 and calls[0][1]["seed"] == 2026
    monkeypatch.setattr(
        evaluation_stats,
        "paired_interval",
        lambda *args, **kwargs: pytest.fail("Recomputed completed cohort"),
    )
    assert cohort_interval(tmp_path, "dark", identity) == (name, interval)
    with pytest.raises(ValueError, match="receipt changed"):
        cohort_interval(tmp_path, "dark", {"arrays": "changed hash"})
    path = output / "reporting-recovery/dark.json"
    saved = read_json(path)
    saved["interval"]["valid_draws"] = 1000
    write_json(path, saved)
    with pytest.raises(ValueError, match="receipt changed"):
        cohort_interval(tmp_path, "dark", identity)
