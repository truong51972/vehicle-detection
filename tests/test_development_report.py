"""Synthetic matching statistics verify orchestration, supports and saved analysis reuse."""

import copy

import numpy as np
import pytest
from test_adaptive import frozen as frozen
from test_study import bdd as bdd

pytest.importorskip("ultralytics")

from vehicle_detection import development_report as reporting  # noqa: E402
from vehicle_detection.common import digest, file_hash, read_json, write_json  # noqa: E402
from vehicle_detection.protocol import summarize  # noqa: E402


def test_report_preserves_pairs_supports_and_reuses_without_recomputing(
    frozen, tmp_path, monkeypatch
):
    protocol, _ = frozen
    protocol = copy.deepcopy(protocol)
    template = protocol["splits"]["development_validation"][0]
    rows = []
    for i in range(8):
        row = {
            **copy.deepcopy(template),
            "id": f"fixture-{i}",
            "group": f"group-{i}",
            "source_group": f"source-{i}",
            "sha256": digest(i),
            "dhash64": digest([i]),
        }
        row["attributes"]["weather"] = "rare" if i == 0 else "common"
        rows.append(row)
    protocol["splits"]["development_validation"] = rows
    protocol["summaries"]["development_validation"] = summarize(rows)
    protocol["manifest_id"] = digest({k: v for k, v in protocol.items() if k != "manifest_id"})
    checkpoints = {}
    for name in ("pretrained", "raw", "adaptive"):
        checkpoints[name] = tmp_path / f"{name}.pt"
        checkpoints[name].write_bytes(b"statistical fixture only")
    for model in checkpoints:
        for treatment in ("raw", "adaptive"):
            root = tmp_path / "evaluations" / f"{model}-{treatment}"
            root.mkdir(parents=True)
            labels = np.array([int(b[0]) for b in rows[0]["boxes"]])
            arrays = {
                "tp": np.ones((len(labels), 10), bool),
                "conf": np.full(len(labels), 0.9),
                "pred_cls": labels,
                "target_cls": labels,
            }
            np.savez_compressed(
                root / "paired_stats.npz",
                ids=np.array([r["id"] for r in rows]),
                **{f"{key}_{i}": value for key, value in arrays.items() for i in range(8)},
            )
            identity = {
                "role": "development_validation",
                "protocol_id": protocol["manifest_id"],
                "policy_id": "fixture",
                "checkpoint_sha256": file_hash(checkpoints[model]),
                "settings": {"fixture": True},
            }
            result = {
                "overall": {"mAP50_95": 0.995, "mAP50": 0.995},
                "classes": [],
                "speed_ms": {},
                "artifacts": {"paired_stats.npz": file_hash(root / "paired_stats.npz")},
            }
            write_json(
                root / "native_result.json",
                {"identity": identity, "result": result, "checksum": digest(result)},
            )
    plan = tmp_path / "plan.md"
    plan.write_text("Fixture plan")
    args = (
        protocol,
        {"policy_id": "fixture"},
        checkpoints,
        tmp_path / "evaluations",
        tmp_path / "report",
        plan,
    )
    result = reporting.development_report(*args, resamples=5)
    overall = [r for r in result["comparisons"] if r["axis"] == "overall"]
    assert len(overall) == 2 and all(r["recovery"] == 0 for r in overall)
    assert all(r["valid_draws"] == 5 for r in overall)
    assert any(r["condition"] == "rare" and r["images"] == 1 for r in result["conditions"])

    def no_bootstrap(*args, **kwargs):
        pytest.fail("Valid saved report should not recompute bootstrap")

    original_interval = reporting.paired_interval
    monkeypatch.setattr(reporting, "paired_interval", no_bootstrap)
    assert reporting.development_report(*args, resamples=5) == result

    def no_valid_draws(*args, **kwargs):
        raise ValueError("Fewer than two valid fixed-class draws; interval is not estimable")

    monkeypatch.setattr(reporting, "paired_interval", no_valid_draws)
    unsupported_args = (*args[:4], tmp_path / "unsupported", args[-1])
    unsupported = reporting.development_report(*unsupported_args, resamples=2)
    assert len(unsupported["comparisons"]) == 2
    assert all(row["status"] == "unsupported" for row in unsupported["comparisons"])
    assert all("recovery_ci_low" not in row for row in unsupported["comparisons"])

    def invalid_ground_truth(*args, **kwargs):
        raise ValueError("Paired ground truth changed")

    monkeypatch.setattr(reporting, "paired_interval", invalid_ground_truth)
    with pytest.raises(ValueError, match="Paired ground truth changed"):
        invalid_args = (*args[:4], tmp_path / "invalid", args[-1])
        reporting.development_report(*invalid_args, resamples=2)
    monkeypatch.setattr(reporting, "paired_interval", original_interval)

    # Historical summary source hashes remain valid and readable without recomputation.
    historical = read_json(tmp_path / "report/summary.json")
    historical["identity"]["version"] = "development-analysis-v1"
    historical["identity"]["code_sha256"] = "original-study-report"
    historical["identity"]["statistics_code_sha256"] = "original-native-ap"
    historical["result"]["provenance"] = historical["identity"]
    historical["checksum"] = digest(historical["result"])
    write_json(tmp_path / "report/summary.json", historical)
    assert reporting.development_report(*args, resamples=5) == historical["result"]
    with pytest.raises(ValueError, match="Historical analysis inputs/settings"):
        reporting.development_report(*args, resamples=0)

    point_args = (*args[:4], tmp_path / "points-only", args[-1])
    points = reporting.development_report(*point_args, resamples=0)
    overall_points = [r for r in points["comparisons"] if r["axis"] == "overall"]
    assert all(r["status"] == "not_resampled" and r["recovery"] == 0 for r in overall_points)
    assert all("recovery_ci_low" not in r for r in overall_points)
    cache = tmp_path / "evaluations/raw-raw/native_result.json"
    changed = read_json(cache)
    changed["identity"]["role"] = "final_test"
    write_json(cache, changed)
    with pytest.raises(ValueError, match="role/protocol"):
        reporting.development_report(*args, resamples=5)
