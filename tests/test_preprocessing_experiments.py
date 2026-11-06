"""Round 2 treatments retain paired calibration identities and never use reserved pixels."""

import copy
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
from test_evaluation_data import prepared as prepared
from test_study import bdd as bdd

from vehicle_detection.common import digest, file_hash, read_json, write_json
from vehicle_detection.evaluation_data import evaluation_sets
from vehicle_detection.five_class import FIVE_CLASSES, SIX_TO_FIVE, remap_boxes
from vehicle_detection.preprocess import apply_recipe
from vehicle_detection.preprocessing_experiments import (
    apply_candidate,
    calibration_sample,
    candidates,
    confirmation_decision,
    export_candidate,
    rank_candidates,
    validate_candidate,
)


def labels_for(pilot, directory):
    write_json(
        directory / "conversion.json",
        {
            "classes": FIVE_CLASSES,
            "mapping": {str(k): v for k, v in SIX_TO_FIVE.items()},
        },
    )
    for row in pilot["rows"]:
        target = directory / "labels/tune" / f"{row['id']}.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "".join(
                f"{cls} {(x1 + x2) / 2 / row['width']} {(y1 + y2) / 2 / row['height']} "
                f"{(x2 - x1) / row['width']} {(y2 - y1) / row['height']}\n"
                for cls, x1, y1, x2, y2 in remap_boxes(row["boxes"])
            )
        )


def test_bounded_candidates_preserve_pixels_and_explicit_order(prepared):
    *_, v2 = prepared
    image = np.random.default_rng(17).integers(0, 256, (32, 40, 3), dtype=np.uint8)
    original = image.copy()
    configs = candidates()
    assert len(configs) == 20
    for config in configs.values():
        output, decisions = apply_candidate(image, config, v2)
        assert output.shape == image.shape and output.dtype == image.dtype
        np.testing.assert_array_equal(image, original)
        np.testing.assert_array_equal(output, apply_candidate(image, config, v2)[0])
        assert len(decisions) <= 2
    np.testing.assert_array_equal(apply_candidate(image, configs["skip"], v2)[0], image)
    np.testing.assert_array_equal(
        apply_candidate(image, configs["frozen_v2"], v2)[0],
        apply_recipe(image, "adaptive", v2["recipes"])[0],
    )
    forward = apply_candidate(image, configs["gamma_070_then_clahe_12"], v2)[0]
    reverse = apply_candidate(image, configs["clahe_12_then_gamma_070"], v2)[0]
    assert not np.array_equal(forward, reverse)
    dark = np.full_like(image, 10)
    assert apply_candidate(dark, configs["route_dark"], v2)[1] == configs["gamma_070"]["steps"]
    assert apply_candidate(np.full_like(image, 200), configs["route_dark"], v2)[1] == []
    for invalid in (
        {"steps": [{"gamma": {"gamma": 0}}]},
        {"steps": [{"unknown": {}}]},
        {"steps": configs["gamma_070"]["steps"] * 3},
        {"rules": [{"metric": "scenario", "below": 1, "steps": []}]},
        {"rules": [{"metric": "brightness", "below": float("nan"), "steps": []}]},
    ):
        with pytest.raises(ValueError):
            validate_candidate(invalid)


def test_samples_use_saved_disjoint_halves_and_exports_encode_once(prepared, tmp_path, monkeypatch):
    protocol, data, *_, v2 = prepared
    manifest = evaluation_sets(protocol)
    exploration = calibration_sample(protocol, manifest, "calibration_exploration", target_images=1)
    confirmation = calibration_sample(
        protocol, manifest, "calibration_confirmation", target_images=1
    )
    assert not {r["group"] for r in exploration["rows"]} & {
        r["group"] for r in confirmation["rows"]
    }
    with pytest.raises(ValueError, match="Only exploration"):
        calibration_sample(protocol, manifest, "final_test")
    changed = copy.deepcopy(manifest)
    changed["sets"]["calibration_exploration"] = manifest["sets"]["development_validation"]
    changed["manifest_id"] = digest({k: v for k, v in changed.items() if k != "manifest_id"})
    with pytest.raises(ValueError, match="manifest changed"):
        calibration_sample(protocol, changed, "calibration_exploration")
    converted = tmp_path / "labels"
    labels_for(exploration, converted)
    scenario = manifest["scenarios"][5]
    writes = []
    original_write, original_read = cv2.imwrite, cv2.imread

    def write(path, pixels, params):
        writes.append((path, params))
        assert params == [cv2.IMWRITE_JPEG_QUALITY, 95]
        return original_write(path, pixels, params)

    def read(path, *args, **kwargs):
        assert "val" not in Path(path).parts
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(cv2, "imwrite", write)
    monkeypatch.setattr(cv2, "imread", read)
    monkeypatch.setattr(cv2, "imencode", lambda *a, **kw: pytest.fail("Extra JPEG encoding"))
    outputs = []
    for name in ("skip", "gamma_070", "route_dark"):
        output = tmp_path / name
        args = (
            protocol,
            manifest,
            exploration,
            data,
            converted,
            output,
            scenario,
            candidates()[name],
            v2,
        )
        meta = export_candidate(*args)
        assert len(meta) == len(exploration["rows"])
        assert not list((output / "images/test").iterdir())
        count = len(writes)
        assert export_candidate(*args).id.tolist() == meta.id.tolist()
        assert len(writes) == count
        outputs.append(meta)
    assert len(writes) == 3 * len(exploration["rows"])
    assert all(t.label_sha256.tolist() == outputs[0].label_sha256.tolist() for t in outputs)
    source = data / exploration["rows"][0]["source_path"]
    before = file_hash(source)
    (tmp_path / "skip/images/tune" / f"{exploration['rows'][0]['id']}.jpg").write_bytes(b"stale")
    with pytest.raises(ValueError, match="changed"):
        export_candidate(
            protocol,
            manifest,
            exploration,
            data,
            converted,
            tmp_path / "skip",
            scenario,
            candidates()["skip"],
            v2,
        )
    assert file_hash(source) == before
    with pytest.raises(ValueError, match="Immutable"):
        export_candidate(
            protocol,
            manifest,
            exploration,
            data,
            converted,
            tmp_path / "gamma_070",
            scenario,
            candidates()["clahe_12"],
            v2,
        )


def metric_table(configs, scenarios):
    return pd.DataFrame(
        [
            {
                "candidate": name,
                "scenario": s["id"],
                "family": s["family"],
                "full_class_support": True,
                "minimum_class_groups": 5,
                "classes": [{"class": c, "AP50_95": score} for c in FIVE_CLASSES],
                "mAP50_95": score,
                "mean_ms": 1.0,
            }
            for name, score in configs.items()
            for s in scenarios
        ]
    )


def test_selection_needs_full_paired_matrix_and_confirmation_never_searches_fallback():
    scenarios = [
        {"id": "clean", "family": "clean"},
        *[{"id": f"noise{i}", "family": "noise"} for i in range(3)],
    ]
    table = metric_table({"skip": 0.4, "frozen_v2": 0.4, "gamma_070": 0.42}, scenarios)
    shortlist = {"candidate": "gamma_070"}
    assert rank_candidates(table, scenarios)[0]["candidate"] == "gamma_070"
    controls = table[table.candidate != "gamma_070"]
    assert rank_candidates(controls, scenarios)[0]["candidate"] == "skip"
    chosen = confirmation_decision(table, scenarios, shortlist, candidates(), {"plan": "test"})
    assert chosen["status"] == "selected_calibrated_candidate"
    assert chosen["policy_id"] == digest({k: v for k, v in chosen.items() if k != "policy_id"})
    table.loc[(table.candidate == "gamma_070") & (table.scenario == "clean"), "mAP50_95"] = 0.3
    failed = confirmation_decision(table, scenarios, shortlist, candidates(), {})
    assert failed["status"] == "confirmation_failed"
    assert failed["candidate"] == shortlist["candidate"] and failed["reasons"]
    table.loc[table.index[0], "full_class_support"] = False
    with pytest.raises(ValueError, match="every scenario"):
        rank_candidates(table, scenarios)
    with pytest.raises(ValueError, match="only the prelocked"):
        confirmation_decision(controls, scenarios, shortlist, candidates(), {})


def test_replay_rejects_changed_configs_and_unconfirmed_policy():
    import vehicle_detection.preprocessing_experiments as experiments
    from vehicle_detection.preprocessing_experiments import replay_policy

    scenarios = [{"id": "clean", "family": "clean"}]
    table = metric_table({"skip": 0.4, "frozen_v2": 0.4, "gamma_070": 0.42}, scenarios)
    plan = {
        "candidates": candidates(),
        "numeric_versions": {"numpy": np.__version__, "opencv": cv2.__version__},
        "implementation": {
            name: file_hash(Path(experiments.__file__).with_name(name))
            for name in ("preprocessing_experiments.py", "preprocess.py", "stress.py")
        },
    }
    plan["plan_id"] = digest(plan)
    policy = confirmation_decision(
        table, scenarios, {"candidate": "gamma_070"}, candidates(), {"plan": plan, "v2_recipes": {}}
    )
    image = np.full((16, 20, 3), 50, np.uint8)
    np.testing.assert_array_equal(
        replay_policy(image, policy)[0], apply_candidate(image, candidates()["gamma_070"], {})[0]
    )
    changed = copy.deepcopy(policy)
    changed["config"]["steps"][0]["gamma"]["gamma"] = 0.5
    with pytest.raises(ValueError, match="intact confirmed"):
        replay_policy(image, changed)
    changed["policy_id"] = digest({k: v for k, v in changed.items() if k != "policy_id"})
    with pytest.raises(ValueError, match="preregistered"):
        replay_policy(image, changed)
    changed = {**policy, "status": "confirmation_failed"}
    with pytest.raises(ValueError, match="intact confirmed"):
        replay_policy(image, changed)
    table.loc[table.candidate == "gamma_070", "minimum_class_groups"] = 4
    with pytest.raises(ValueError, match="five groups"):
        rank_candidates(table, scenarios)
    table.loc[table.candidate == "gamma_070", "minimum_class_groups"] = 5
    table.loc[table.candidate == "gamma_070", "mAP50_95"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        rank_candidates(table, scenarios)


def test_class_regression_can_veto_a_good_overall_confirmation():
    scenarios = [{"id": "clean", "family": "clean"}]
    table = metric_table({"skip": 0.4, "frozen_v2": 0.4, "gamma_070": 0.42}, scenarios)
    table.loc[table.candidate == "gamma_070", "classes"].iloc[0][4]["AP50_95"] = 0.3
    decision = confirmation_decision(table, scenarios, {"candidate": "gamma_070"}, candidates(), {})
    assert decision["status"] == "confirmation_failed"
    assert any("bicycle" in reason for reason in decision["reasons"])


def test_fixed_development_transfer_is_disjoint_and_cannot_admit_final_ids(
    prepared, tmp_path, monkeypatch
):
    from vehicle_detection.preprocessing_experiments import (
        development_sample,
        validate_sample,
        verify_candidate_export,
    )

    protocol, data, *_, v2 = prepared
    manifest = evaluation_sets(protocol)
    selected = development_sample(protocol, manifest, target_images=2)
    assert selected == development_sample(protocol, manifest, target_images=2)
    assert selected["role"] == selected["sample_role"] == "development_validation"
    calibration_ids = {r["id"] for r in protocol["splits"]["calibration"]}
    assert not {r["id"] for r in selected["rows"]} & calibration_ids
    converted = tmp_path / "transfer-labels"
    labels_for(selected, converted)
    destination = tmp_path / "transfer-export"
    meta = export_candidate(
        protocol,
        manifest,
        selected,
        data,
        converted,
        destination,
        manifest["scenarios"][0],
        candidates()["gamma_070"],
        v2,
    )
    assert set(meta.role) == {"development_validation"}
    assert set(meta.id) == {r["id"] for r in selected["rows"]}
    assert not list((destination / "images/test").iterdir())
    assert verify_candidate_export(selected, destination).id.tolist() == meta.id.tolist()
    assert read_json(destination / "export.json")["role"] == "development_validation"
    changed = copy.deepcopy(selected)
    changed["rows"] = [protocol["splits"]["final_test"][0]]
    changed["pilot_id"] = digest({k: v for k, v in changed.items() if k != "pilot_id"})
    monkeypatch.setattr(
        "vehicle_detection.preprocessing_experiments.scenario_image",
        lambda *a, **k: pytest.fail("Reserved pixels prepared"),
    )
    with pytest.raises(ValueError, match="frozen role"):
        validate_sample(protocol, manifest, changed)
    with pytest.raises(ValueError, match="frozen role"):
        export_candidate(
            protocol,
            manifest,
            changed,
            data,
            converted,
            tmp_path / "unsafe",
            manifest["scenarios"][0],
            candidates()["skip"],
            v2,
        )
    (destination / "labels/tune" / f"{selected['rows'][0]['id']}.txt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        verify_candidate_export(selected, destination)
