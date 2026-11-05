"""Quality search uses native dataset AP, bounded cost and exploration-only evidence."""

import copy

import numpy as np
import pandas as pd
import pytest
from test_evaluation_data import prepared as prepared
from test_study import bdd as bdd

from vehicle_detection.common import digest
from vehicle_detection.evaluation_data import evaluation_sets
from vehicle_detection.preprocessing_experiments import apply_candidate, calibration_sample
from vehicle_detection.quality_search import (
    collect_quality,
    diverse_beam,
    fresh_confirmation,
    quality_summary,
    route_assignments,
    search_rules,
)


def test_beam_keeps_leaders_and_adds_new_conditions_and_actions():
    leaders = [
        {"rules": [{"metric": "contrast", "above": 0.25, "action": f"action-{i}"}]}
        for i in range(4)
    ]
    alternatives = [
        {"rules": [r]}
        for r in (
            {"metric": "brightness", "below": 0.2, "action": "gamma"},
            {"metric": "noise_highpass", "above": 0.01, "action": "bilateral"},
            {"metric": "sharpness", "below": 20, "action": "sharpen"},
            {"metric": "contrast", "below": 0.1, "action": "gamma"},
        )
    ]
    beam = diverse_beam(leaders + alternatives)
    assert len(beam) == 8
    assert beam[:4] == [t["rules"][0] for t in leaders]
    assert {r["metric"] for r in beam} == {"brightness", "contrast", "noise_highpass", "sharpness"}
    assert diverse_beam([]) == []


def test_first_match_can_explicitly_skip_without_later_rules_overriding_it():
    values = pd.DataFrame({"brightness": [0.1, 0.5, 0.9]})
    rules = [
        {"metric": "brightness", "below": 0.2, "action": "skip"},
        {"metric": "brightness", "below": 0.7, "action": "gamma"},
    ]
    assert route_assignments(values, rules).tolist() == ["skip", "gamma", "skip"]


def test_quality_uses_exploration_only_and_reports_proxy_limits(prepared, monkeypatch):
    protocol, data, *_ = prepared
    manifest = evaluation_sets(protocol)
    pilot = calibration_sample(protocol, manifest, "calibration_exploration", target_images=1)
    table = collect_quality(protocol, manifest, pilot, data)
    assert len(table) == len(pilot["rows"]) * len(manifest["scenarios"])
    summary = quality_summary(table, pilot, manifest["scenarios"])
    assert len(summary["distributions"]) == 70
    assert {r["id"] for r in summary["representatives"]} <= {r["id"] for r in pilot["rows"]}
    assert len(summary["class_support"]) == 5
    assert "texture" in summary["limits"][0]
    monkeypatch.setattr(
        "vehicle_detection.quality_search.scenario_image",
        lambda *a, **k: pytest.fail("Non-exploration pixels opened"),
    )
    confirmation = calibration_sample(
        protocol, manifest, "calibration_confirmation", target_images=1
    )
    with pytest.raises(ValueError, match="exploration pixels only"):
        collect_quality(protocol, manifest, confirmation, data)
    changed = table.copy()
    changed.loc[0, "brightness"] = np.nan
    with pytest.raises(ValueError, match="finite, paired"):
        quality_summary(changed, pilot, manifest["scenarios"])


@pytest.fixture
def search_data():
    # Each class appears in ten independent groups. A filter helps dark images and
    # destroys all detections in bright images, so dataset AP rewards conditional Skip.
    rows, quality = [], []
    stats = {name: {} for name in ("skip", "frozen_v2", "gamma")}
    for i in range(10):
        sample_id = f"image-{i}"
        row = {
            "id": sample_id,
            "group": sample_id,
            "boxes": [[c, 1, 1, 10, 10] for c in (0, 1, 2, 4, 5)],
            "attributes": {"weather": "clear", "timeofday": "daytime", "scene": "city street"},
        }
        rows.append(row)
        quality.append(
            {
                "id": sample_id,
                "group": sample_id,
                "scenario": "clean",
                "brightness": 0.1 if i < 5 else 0.9,
                "contrast": 0.1,
                "noise_highpass": 0.01,
                "sharpness": 10.0,
                "noise": 0.0,
                "black_fraction": 0.0,
                "white_fraction": 0.0,
                "routing_ms": 0.1,
            }
        )
        for name in stats:
            success = i < 5 if name == "gamma" else i >= 5
            stats[name][sample_id] = {
                "target_cls": np.arange(5),
                "pred_cls": np.arange(5),
                "conf": np.full(5, 0.8),
                "tp": np.full((5, 10), success, dtype=bool),
            }
    pilot = {"sample_role": "calibration_exploration", "rows": rows, "pilot_id": "fixture"}
    scenarios = [{"id": "clean", "family": "clean"}]
    configs = {
        "skip": {"steps": []},
        "frozen_v2": {"v2": True},
        "gamma": {"steps": [{"gamma": {"gamma": 0.7}}]},
    }
    actions = {
        "clean": {
            name: {"stats": values, "costs": dict.fromkeys(values, 2.0 if name == "gamma" else 0.1)}
            for name, values in stats.items()
        }
    }
    return pd.DataFrame(quality), pilot, scenarios, configs, actions


def test_native_search_learns_conditional_skip_and_cost_can_veto_it(search_data):
    pytest.importorskip("ultralytics")
    from vehicle_detection.evaluation_stats import class_ap

    table, pilot, scenarios, configs, actions = search_data
    result = search_rules(*search_data, max_mean_ms=3, max_p95_ms=3)
    assert result["status"] == "pending_fresh_confirmation"
    selected = result["selected"]
    assert len(selected["rules"]) == 1
    rule = selected["rules"][0]
    assert rule == {"metric": "brightness", "below": 0.5, "action": "gamma"}
    assert selected["metrics"][0]["routes"] == {"gamma": 5, "skip": 5}
    # Recompute the selected native dataset AP from the routed arrays, without image AP labels.
    routed = {
        r["id"]: actions["clean"]["gamma" if i < 5 else "skip"]["stats"][r["id"]]
        for i, r in enumerate(pilot["rows"])
    }
    native = class_ap(routed, list(routed), detailed=True)
    assert selected["score"] == pytest.approx(np.mean([v["AP50_95"] for v in native.values()]))
    assert selected["metrics"][0]["mean_ms"] == pytest.approx(1.15)
    assert selected["metrics"][0]["p95_ms"] == pytest.approx(2.1)
    assert result["search_id"] == digest({k: v for k, v in result.items() if k != "search_id"})
    assert search_rules(*search_data, max_mean_ms=1, max_p95_ms=3)["selected"] is None
    assert search_rules(*search_data, max_mean_ms=3, max_p95_ms=1)["selected"] is None
    dark, light = (np.full((16, 16, 3), value, np.uint8) for value in (25, 230))
    assert apply_candidate(dark, selected["config"], {})[1] == configs["gamma"]["steps"]
    assert apply_candidate(light, selected["config"], {})[1] == []
    # No gain over both controls cannot release a config, even when cost is permissive.
    for key in actions["clean"]["gamma"]["stats"]:
        actions["clean"]["gamma"]["stats"][key] = actions["clean"]["skip"]["stats"][key]
    assert (
        search_rules(table, pilot, scenarios, configs, actions, max_mean_ms=3, max_p95_ms=3)[
            "selected"
        ]
        is None
    )


def test_search_rejects_unpaired_inputs_and_unsupported_classes(search_data):
    pytest.importorskip("ultralytics")
    table, pilot, scenarios, configs, actions = search_data
    with pytest.raises(ValueError, match="finite positive"):
        search_rules(*search_data, max_mean_ms=np.nan, max_p95_ms=10)
    with pytest.raises(ValueError, match="finite, paired"):
        search_rules(
            table.iloc[:-1], pilot, scenarios, configs, actions, max_mean_ms=10, max_p95_ms=10
        )
    changed = copy.deepcopy(pilot)
    changed["sample_role"] = "calibration_confirmation"
    with pytest.raises(ValueError, match="exploration only"):
        search_rules(table, changed, scenarios, configs, actions, max_mean_ms=10, max_p95_ms=10)
    changed = copy.deepcopy(pilot)
    for row in changed["rows"][4:]:
        row["boxes"] = [b for b in row["boxes"] if b[0] != 5]
    with pytest.raises(ValueError, match="five source groups"):
        search_rules(table, changed, scenarios, configs, actions, max_mean_ms=10, max_p95_ms=10)
    changed_actions = copy.deepcopy(actions)
    del changed_actions["clean"]["gamma"]["stats"]["image-0"]
    with pytest.raises(ValueError, match="paired IDs"):
        search_rules(
            table, pilot, scenarios, configs, changed_actions, max_mean_ms=10, max_p95_ms=10
        )


def test_two_rules_can_recover_distinct_cases_while_retaining_skip(search_data):
    pytest.importorskip("ultralytics")
    table, pilot, scenarios, configs, actions = search_data
    configs["sharpen"] = {"steps": [{"sharpen": {"amount": 0.15, "sigma": 1.0}}]}
    actions["clean"]["sharpen"] = copy.deepcopy(actions["clean"]["gamma"])
    for i in range(10):
        table.loc[i, "brightness"] = 0.1 if i < 4 else 0.9
        table.loc[i, "sharpness"] = 1 if 4 <= i < 7 else 10
        for name in configs:
            success = i >= 7
            if name == "gamma":
                success = i < 4
            elif name == "sharpen":
                success = 4 <= i < 7
            actions["clean"][name]["stats"][f"image-{i}"]["tp"][:] = success
    result = search_rules(table, pilot, scenarios, configs, actions, max_mean_ms=5, max_p95_ms=5)
    selected = result["selected"]
    assert len(selected["rules"]) == 2
    assert selected["metrics"][0]["routes"] == {"gamma": 4, "sharpen": 3, "skip": 3}
    assert selected["score"] > 0.99


def test_one_local_quantile_pass_can_improve_the_quartile_grid(search_data):
    pytest.importorskip("ultralytics")
    table, pilot, scenarios, configs, original = search_data
    rows, qualities = [], []
    actions = {"clean": {name: {"stats": {}, "costs": {}} for name in configs}}
    for i in range(20):
        sample_id = f"refine-{i}"
        rows.append({**copy.deepcopy(pilot["rows"][i % 10]), "id": sample_id, "group": sample_id})
        qualities.append(
            {
                **table.iloc[i % 10].to_dict(),
                "id": sample_id,
                "group": sample_id,
                "brightness": i / 20,
            }
        )
        for name in configs:
            values = copy.deepcopy(original["clean"][name]["stats"][f"image-{i % 10}"])
            values["tp"][:] = i < 11 if name == "gamma" else i >= 11
            actions["clean"][name]["stats"][sample_id] = values
            actions["clean"][name]["costs"][sample_id] = 2 if name == "gamma" else 0.1
    pilot = {**pilot, "rows": rows}
    quality = pd.DataFrame(qualities)
    selected = search_rules(
        quality, pilot, scenarios, configs, actions, max_mean_ms=3, max_p95_ms=3
    )["selected"]
    assert len(selected["rules"]) == 1
    assert selected["rules"][0]["below"] == pytest.approx(quality.brightness.quantile(0.55))
    assert selected["score"] > 0.99


def test_fresh_confirmation_excludes_used_groups_without_reading_pixels(prepared, monkeypatch):
    protocol, *_ = prepared
    manifest = evaluation_sets(protocol)
    used = [
        calibration_sample(protocol, manifest, role, target_images=1)
        for role in ("calibration_exploration", "calibration_confirmation")
    ]
    monkeypatch.setattr(
        "vehicle_detection.quality_search.scenario_image",
        lambda *a, **k: pytest.fail("Confirmation pixels opened"),
    )
    result = fresh_confirmation(protocol, manifest, used, target_images=1)
    assert result == fresh_confirmation(protocol, manifest, used, target_images=1)
    excluded = {g for p in used for r in p["rows"] for g in (r["group"], r["source_group"])}
    assert not excluded & {
        g for r in result["pilot"]["rows"] for g in (r["group"], r["source_group"])
    }
    assert {r["id"] for r in result["pilot"]["rows"]} <= set(
        manifest["sets"]["calibration_confirmation"]
    )
    exhausted = calibration_sample(
        protocol, manifest, "calibration_confirmation", target_images=1000
    )
    empty = fresh_confirmation(protocol, manifest, [*used, exhausted], target_images=1)
    assert empty["available_images"] == 0 and not empty["target_met"]
    assert empty["status"] == "metadata_only_not_evaluated"
