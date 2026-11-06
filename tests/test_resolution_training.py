"""Prospective training-resolution gates and immutable evidence inspection."""

from copy import deepcopy

import numpy as np
import pytest

from vehicle_detection.common import file_hash, write_json
from vehicle_detection.five_class import FIVE_CLASSES
from vehicle_detection.resolution_training import (
    assess,
    check_populations,
    examples,
    run,
    select_training,
)


def test_training_prefix_keeps_whole_groups_and_ignores_labels_and_input_order():
    rows = [
        {"id": f"{g}{i}", "group": g, "role": "training_pool", "s20": 1}
        for g in "abcd"
        for i in range(2)
    ]
    rows.append({"id": "reserved", "group": "z", "role": "calibration", "s20": 0})
    selected = select_training(rows, 3, 4)
    assert len(selected) == 4
    assert selected == select_training(list(reversed(rows)), 3, 4)
    assert len({r["group"] for r in selected}) == 2
    with pytest.raises(ValueError, match="budget"):
        select_training(rows, 3, 3)


def test_populations_reject_source_leakage_and_unsupported_classes():
    def population(prefix):
        return [
            {
                **dict.fromkeys(("id", "group", "source_group", "sha256"), f"{prefix}{i}"),
                "boxes": [[c, 0, 0, 20, 20] for c in range(5)],
            }
            for i in range(5)
        ]

    train, evaluation = population("train"), population("eval")
    check_populations(train, evaluation)
    evaluation[0]["sha256"] = train[0]["sha256"]
    with pytest.raises(ValueError, match="overlap: sha256"):
        check_populations(train, evaluation)
    with pytest.raises(ValueError, match="support"):
        check_populations(train[:4], population("eval"))


def test_primary_contrast_holds_inference_size_fixed_and_keeps_every_failed_gate():
    record = {
        "mAP50_95": 0.2,
        "classes": {name: {"AP50_95": 0.2, "source_groups": 5} for name in FIVE_CLASSES},
    }
    result = {
        "metrics": {
            key: {s: deepcopy(record) for s in ("all", "dark", "bright")}
            for key in ("640/960", "960/960")
        },
        "training": {
            size: {"wall_seconds": 10, "peak_allocated_gib": 3} for size in ("640", "960")
        },
        "timings": {
            key: {"native_ms_per_image": {"inference": 2}} for key in ("640/960", "960/960")
        },
        "paired_interval": {"estimand": {"class_ids": list(range(5))}, "recovery_ci_low": 0.001},
    }
    for r in result["metrics"]["960/960"].values():
        r["mAP50_95"] = 0.21
        for c in r["classes"].values():
            c["AP50_95"] = 0.21
    assert assess(result)["useful_training_resolution_signal"]
    result["metrics"]["960/960"]["dark"]["classes"]["truck"]["AP50_95"] = 0.18
    result["metrics"]["960/960"]["all"]["classes"]["motorcycle"]["AP50_95"] = 0.19
    result["paired_interval"]["recovery_ci_low"] = 0
    result["training"]["960"].update(wall_seconds=31, peak_allocated_gib=15)
    result["timings"]["960/960"]["native_ms_per_image"]["inference"] = 4
    decision = assess(result)
    assert not decision["useful_training_resolution_signal"]
    assert len(decision["reasons"]) == 6
    assert not decision["s20_promoted"] and not decision["policy_selected"]


def test_default_replay_checks_hashes_and_opt_in_never_overwrites(tmp_path):
    report = tmp_path / "docs/issue47-resolution-training"
    write_json(report / "result.json", {"decision": {"policy_selected": False}})
    write_json(report / "evidence.json", {"result.json": file_hash(report / "result.json")})
    assert not run(tmp_path)["decision"]["policy_selected"]
    with pytest.raises(FileExistsError, match="attempt exists"):
        run(tmp_path, allow_training=True)
    write_json(report / "result.json", {})
    with pytest.raises(ValueError, match="evidence changed"):
        run(tmp_path)


def test_examples_are_deterministic_native_class_counts_not_box_attribution():
    values = {
        "conf": np.array([0.9, 0.1]),
        "pred_cls": np.array([3, 3]),
        "tp": np.zeros((2, 10), dtype=bool),
        "target_cls": np.array([3]),
    }
    before, after = {"a": values}, {"a": deepcopy(values)}
    after["a"]["tp"][0] = True
    assert examples(before, after) == [
        {
            "id": "a",
            "class": "motorcycle",
            "before": {"TP50": 0, "FP50": 1},
            "after": {"TP50": 1, "FP50": 0},
        }
    ]
