import copy

import numpy as np
import pandas as pd
import pytest
from test_stress import pilot as pilot
from test_study import bdd as bdd

from vehicle_detection.common import digest, file_hash, write_json
from vehicle_detection.robustness_spec import final_stress_spec


def toy_stats(classes, *, rare_improved=False):
    labels = np.array(classes)
    tp = np.ones((len(classes), 10), dtype=bool)
    if 3 in classes and not rare_improved:
        tp[labels == 3] = False
    return {
        "tp": tp,
        "conf": np.full(len(classes), 0.9),
        "pred_cls": labels.copy(),
        "target_cls": labels.copy(),
    }


def test_rare_class_omission_never_changes_the_estimand():
    pytest.importorskip("ultralytics")
    from vehicle_detection.stress_eval import native_ap, paired_interval

    pilot = {"rows": [{"id": f"i{i}", "group": f"g{i}"} for i in range(8)]}
    before = {r["id"]: toy_stats([0, 3] if i == 0 else [0]) for i, r in enumerate(pilot["rows"])}
    after = copy.deepcopy(before)
    after["i0"] = toy_stats([0, 3], rare_improved=True)
    result = paired_interval(before, after, pilot, resamples=100)
    assert result["estimand"]["class_ids"] == [0]
    assert result["recovery"] == result["recovery_ci_low"] == result["recovery_ci_high"] == 0
    assert result["valid_draws"] == 100 and result["invalid_draws"] == 0
    rare = result["per_class"][3]
    assert rare["source_groups"] == rare["images"] == 1
    assert not rare["included"] and rare["ci_low"] is None and rare["ci_high"] is None
    assert (
        rare["recovery"] > 0.9
    )  # Descriptive improvement is retained without rare-class inference.
    assert 0 < rare["present_draws"] < 100
    assert any(d["class_support"][3]["images"] == 0 for d in result["draws"])
    assert all(d["recovery"] == 0 for d in result["draws"])
    assert native_ap(before, ["i1"], classes=[0, 3]) is None
    assert native_ap(before, ["i1"], classes=[0]) > 0.9


def test_missing_required_class_draws_are_audited_not_imputed():
    pytest.importorskip("ultralytics")
    from vehicle_detection.stress_eval import paired_interval

    pilot = {"rows": [{"id": f"i{i}", "group": f"g{i}"} for i in range(4)]}
    stats = {r["id"]: toy_stats([0 if i < 2 else 1]) for i, r in enumerate(pilot["rows"])}
    result = paired_interval(stats, stats, pilot, resamples=100, min_class_groups=2)
    assert result["estimand"]["class_ids"] == [0, 1]
    assert 0 < result["invalid_draws"] < 100
    assert result["valid_draws"] + result["invalid_draws"] == 100
    invalid = [d for d in result["draws"] if not d["valid"]]
    assert len(invalid) == result["invalid_draws"]
    assert all(d["missing_class_ids"] and d["recovery"] is None for d in invalid)
    assert result["recovery_ci_low"] == result["recovery_ci_high"] == 0
    assert all(p["valid_draws"] + p["excluded_draws"] == 100 for p in result["per_class"])
    with pytest.raises(ValueError, match="source-group support"):
        paired_interval(stats, stats, pilot, min_class_groups=5)
    with pytest.raises(ValueError, match="positive integer"):
        paired_interval(stats, stats, pilot, min_class_groups=0)


def test_recorded_statistics_verified_without_model_construction(tmp_path, monkeypatch):
    pytest.importorskip("ultralytics")
    from vehicle_detection import stress_eval

    monkeypatch.setattr(stress_eval, "YOLO", lambda *a, **kw: pytest.fail("No inference allowed"))
    stats = toy_stats([0])
    stat_path = tmp_path / "paired_stats.npz"
    np.savez_compressed(stat_path, ids=np.array(["i"]), **{f"{k}_0": v for k, v in stats.items()})
    identity = {
        "role": "calibration",
        "pilot_id": "p",
        "protocol_id": "protocol",
        "checkpoint_sha256": "weights",
        "settings": {},
        "code_sha256": "original",
        "files_identity": "files",
    }
    result = {"artifacts": {"paired_stats.npz": file_hash(stat_path)}}
    path = tmp_path / "native_result.json"
    write_json(path, {"identity": identity, "result": result, "checksum": digest(result)})
    expected = {
        "native_result_sha256": file_hash(path),
        "statistics_sha256": file_hash(stat_path),
        "files_identity": "files",
    }
    args = dict(
        pilot={"pilot_id": "p", "rows": [{"id": "i"}]},
        protocol_id="protocol",
        checkpoint="weights",
        settings={},
        code="original",
    )
    loaded, values = stress_eval.load_recorded_calibration(tmp_path, expected, **args)
    assert loaded == result and set(values) == {"i"}
    with pytest.raises(ValueError, match="identity/role"):
        stress_eval.load_recorded_calibration(tmp_path, expected, **{**args, "code": "new"})
    stat_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        stress_eval.load_recorded_calibration(tmp_path, expected, **args)


def test_final_spec_bounds_cost_without_final_pixels(pilot, monkeypatch):
    from vehicle_detection import common
    from vehicle_detection import protocol as protocol_module

    protocol, _, _ = pilot

    def no_pixels(*args, **kwargs):
        pytest.fail("Final specification must not access image pixels")

    monkeypatch.setattr(common, "read_image", no_pixels)
    monkeypatch.setattr(protocol_module, "read_image", no_pixels)
    conditions = [("clean", 0), *[(f, s) for f in ("shot_read", "motion", "veil") for s in (2, 3)]]
    matrix = pd.DataFrame(
        [
            {
                "family": f,
                "severity": s,
                "operator": op,
                "image_GiB": 0.1,
                "images": 100,
                "model_inference_ms": 1.0,
                "corruption_p95_ms": 2.0,
                "restore_p95_ms": 3.0,
            }
            for f, s in conditions
            for op in ("skip", "adaptive_v2")
        ]
    )
    source = {
        "corruptions": {
            f: [{"level": i} for i in range(3)] for f in ("shot_read", "motion", "veil")
        },
        "settings": {},
    }
    support_policy = {
        "version": "fixed-supported-macro-v2",
        "minimum_source_groups": 5,
        "missing_class_draw": "exclude",
        "limit": "engineering minimum",
    }
    spec = final_stress_spec(protocol, source, matrix, "v2", "gate", support_policy=support_policy)
    count = len(protocol["splits"]["final_test"])
    assert spec["budget"]["evaluations"] == len(spec["evaluation_matrix"]) == 42
    assert spec["budget"]["total_image_passes"] == 42 * count
    assert spec["budget"]["prepared_image_files"] == 14 * count
    assert spec["budget"]["max_simultaneous_datasets"] == 2
    assert spec["treatments"] == ["Raw", "Gamma-or-Skip"]
    assert spec["status"] == "specification_only_not_executed"
    assert spec == final_stress_spec(
        protocol, source, matrix, "v2", "gate", support_policy=support_policy
    )
