"""Follow-up evidence uses measured actions and exactly matching, selective feature routing."""

import cv2
import numpy as np
import pytest
from test_evaluation_data import prepared as prepared
from test_quality_search import search_data as search_data
from test_study import bdd as bdd

from vehicle_detection.evaluation_data import evaluation_sets
from vehicle_detection.preprocessing_experiments import apply_candidate, calibration_sample
from vehicle_detection.quality_followup import (
    additional_settings,
    apply_quality_candidate,
    benchmark_runtime,
    clean_error_report,
    measure_operator_values,
    pareto_comparison,
)
from vehicle_detection.quality_search import search_rules
from vehicle_detection.stress import diagnostics


@pytest.mark.parametrize("shape", [(40, 64, 3), (720, 1280, 3)])
def test_selective_features_match_frozen_pixels_and_threshold_boundaries(shape):
    pixels = np.random.default_rng(38).integers(0, 256, shape, np.uint8)
    original = pixels.copy()
    scores = diagnostics(pixels)
    for metric in ("brightness", "contrast", "noise_highpass", "sharpness"):
        for threshold in (scores[metric] - 1e-12, scores[metric], scores[metric] + 1e-12):
            for bound in ("above", "below"):
                config = {
                    "rules": [
                        {
                            "metric": metric,
                            bound: threshold,
                            "steps": additional_settings()["gamma_0775"]["steps"],
                        }
                    ]
                }
                expected, a = apply_candidate(pixels, config, {})
                actual, b = apply_quality_candidate(pixels, config, {})
                np.testing.assert_array_equal(actual, expected)
                assert a == b
    np.testing.assert_array_equal(pixels, original)


def test_first_match_short_circuits_expensive_noise_and_unused_diagnostics(monkeypatch):
    config = {
        "rules": [
            {"metric": "brightness", "below": 0.5, "steps": []},
            {"metric": "noise_highpass", "above": 0.01, "steps": []},
        ]
    }
    for target in ("filter2D", "medianBlur", "Laplacian"):
        monkeypatch.setattr(cv2, target, lambda *a, **k: pytest.fail("Unused feature computed"))
    image = np.zeros((20, 30, 3), np.uint8)
    np.testing.assert_array_equal(apply_quality_candidate(image, config, {})[0], image)


def test_pareto_retains_cheaper_near_tie_and_excludes_regression_failure():
    search = {
        "controls": [
            {"candidate": "skip", "family": "clean", "mAP50_95": 0.2, "mean_ms": 0.1},
        ],
        "ranking": [
            {"candidate": "winner", "score": 0.24, "mean_ms": 9, "reasons": []},
            {"candidate": "near_tie", "score": 0.239, "mean_ms": 8, "reasons": []},
            {"candidate": "dominated", "score": 0.23, "mean_ms": 10, "reasons": []},
            {"candidate": "regression", "score": 0.3, "mean_ms": 2, "reasons": ["class"]},
        ],
    }
    result = {r["candidate"]: r for r in pareto_comparison(search)}
    assert result["winner"]["pareto"] and result["near_tie"]["pareto"]
    assert not result["dominated"]["pareto"]
    assert not result["regression"]["passes_exploration_regressions"]


def test_clean_inspection_counts_false_positives_at_fixed_confidence(search_data):
    pytest.importorskip("ultralytics")
    table, pilot, _, _, actions = search_data
    selected = search_rules(*search_data, max_mean_ms=3, max_p95_ms=3)["selected"]
    # Add a high-confidence unmatched detection and a low-confidence one to the selected action.
    stats = actions["clean"]["gamma"]["stats"]["image-0"]
    stats["pred_cls"] = np.r_[stats["pred_cls"], 0, 0]
    stats["conf"] = np.r_[stats["conf"], 0.8, 0.1]
    stats["tp"] = np.r_[stats["tp"], np.zeros((2, 10), bool)]
    report = clean_error_report(table, pilot, selected, actions)
    skip = report["comparisons"][0]
    assert skip["before"] == {"TP50": 25, "FP50": 25}
    assert skip["after"] == {"TP50": 50, "FP50": 1}
    assert skip["classes"][0]["after"] == {"TP50": 10, "FP50": 1}
    assert sum(report["routes"].values()) == 10


def test_runtime_uses_one_configuration_and_restores_threads(prepared, monkeypatch):
    protocol, data, *_, v2 = prepared
    manifest = evaluation_sets(protocol)
    pilot = calibration_sample(protocol, manifest, "calibration_exploration", target_images=1)
    original = cv2.getNumThreads()
    configs = {"skip": {"steps": []}, "frozen_v2": {"v2": True}}
    result = benchmark_runtime(protocol, manifest, pilot, data, configs, v2, repeats=1, threads=2)
    assert result["threads"] == 2 and cv2.getNumThreads() == original
    assert all(r["calls"] == len(pilot["rows"]) for r in result["records"])
    confirmation = calibration_sample(
        protocol, manifest, "calibration_confirmation", target_images=1
    )
    monkeypatch.setattr(
        "vehicle_detection.quality_followup.scenario_image",
        lambda *a, **k: pytest.fail("Reserved pixels opened"),
    )
    with pytest.raises(ValueError, match="exploration only"):
        benchmark_runtime(protocol, manifest, confirmation, data, configs, v2)
    with pytest.raises(ValueError, match="exploration only"):
        measure_operator_values(protocol, manifest, confirmation, data, data, data, v2, data)
