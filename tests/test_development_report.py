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
    assert len(overall) == 7 and all(r["recovery"] == 0 for r in overall)
    assert all(r["valid_draws"] == 5 for r in overall)
    assert all(
        r["status"] == "unsupported" for r in result["comparisons"] if r["condition"] == "rare"
    )

    def no_bootstrap(*args, **kwargs):
        pytest.fail("Valid saved report should not recompute bootstrap")

    monkeypatch.setattr(reporting, "paired_interval", no_bootstrap)
    assert reporting.development_report(*args, resamples=5) == result
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
