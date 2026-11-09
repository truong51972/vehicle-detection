"""Saved-result consistency checks; no acquisition or scientific confirmation."""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from vehicle_detection.common import file_hash, read_json
from vehicle_detection.five_class import FIVE_CLASSES
from vehicle_detection.posthoc_training import BASELINE, MATRIX, STUDY, compare_args, read_report


def test_completed_matrix_preserves_control_support_and_checkpoint_identity():
    root = Path(__file__).resolve().parents[1]
    result = read_report(root)
    assert result["status"] == "completed"
    assert "not independent confirmation" in result["scope"]
    assert not result["confirmation_accessed"] and not result["final_test_accessed"]
    assert set(result["metrics"]) == {f"{a}/{b}" for a, b in MATRIX}
    prior = read_json(root / f"docs/{BASELINE}/eval960_960_metrics.json")
    raw_checkpoint = read_json(root / f"docs/{BASELINE}/train960_checkpoint.json")
    adaptive_checkpoint = read_json(root / f"docs/{STUDY}/checkpoint.json")
    checkpoints = {
        "raw": raw_checkpoint["checkpoint_hashes"]["best"],
        "adaptive": adaptive_checkpoint["checkpoint_hashes"]["best"],
    }
    for key, groups in result["metrics"].items():
        assert set(groups) == {"all", "dark", "bright", "small_two_wheeler"}
        assert groups["all"]["images"] == 13975
        for cohort, record in groups.items():
            assert record["images"] == prior[cohort]["images"]
            assert set(record["classes"]) == set(FIVE_CLASSES)
            for metric, class_metric in (("mAP50", "AP50"), ("mAP50_95", "AP50_95")):
                assert np.isclose(
                    record[metric],
                    np.mean([v[class_metric] for v in record["classes"].values()]),
                    atol=1e-12,
                    rtol=0,
                )
            for name, values in record["classes"].items():
                control = prior[cohort]["classes"][name]
                for support in ("instances", "source_groups"):
                    assert values[support] == control[support]
                counts = record["counts"][name]
                assert counts["GT"] == control["instances"] == counts["TP50"] + counts["FN50"]
                assert counts["FP50"] >= 0
                if key == "raw/raw":
                    for metric in ("AP50", "AP50_95"):
                        assert abs(values[metric] - control[metric]) <= 1e-6
            if key == "raw/raw":
                assert record["counts"] == prior[cohort]["counts"]
        timing = result["timings"][key]
        assert timing["checkpoint_sha256"] == checkpoints[key.split("/")[0]]
        assert timing["batches"] == 437
        assert timing["tensor_shapes"] == [[576, 992]]
    assert result["data"]["images"] == {"train": 9781, "val": 13975}
    assert result["new_training_exposures"] == 9781 * 20
    assert result["new_explicit_evaluation_image_passes"] == 13975 * 4
    assert result["new_selection_and_terminal_image_passes"] == 13975 * 21


def test_complete_training_recipe_selection_and_prior_acceptance_are_preserved():
    root = Path(__file__).resolve().parents[1]
    report = root / f"docs/{STUDY}"
    result = read_report(root)
    history = pd.read_csv(report / "training_results.csv")
    assert history.epoch.tolist() == list(range(1, 21))
    compare_args(
        yaml.safe_load((report / "training_args.yaml").read_text()),
        yaml.safe_load((root / f"docs/{BASELINE}/train960_args.yaml").read_text()),
    )
    receipt = read_json(report / "checkpoint.json")
    assert receipt["train_shapes"] == [[960, 960]]
    assert result["selected_epoch"] == {"raw": 10, "adaptive": receipt["selected_epoch"]}
    passes = receipt["selection_passes"]
    assert len(passes) == 21
    assert [p["epoch"] for p in passes[:-1]] == list(range(1, 21))
    assert not any(p["terminal"] for p in passes[:-1]) and passes[-1]["terminal"]
    assert all(
        p["imgsz"] == 640
        and p["pad"] == 0
        and p["images"] == 13975
        and p["tensor_shapes"] == [[384, 640]]
        for p in passes
    )
    for measurement in result["training"].values():
        assert measurement["status"] == "completed" and measurement["imgsz"] == 960
        assert measurement["wall_seconds"] > 0
        assert 0 < measurement["peak_allocated_gib"] < 14
    original = read_json(root / "docs/issue47-v3-960-acceptance/validation.json")
    assert all(file_hash(root / p) == h for p, h in original["new_inputs"].items())
