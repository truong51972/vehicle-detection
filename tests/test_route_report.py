"""Synthetic export routes and native matches; no inference or quality evidence."""

import copy
import json

import cv2
import numpy as np
import pandas as pd
import pytest
from test_adaptive import frozen as frozen
from test_study import bdd as bdd

pytest.importorskip("ultralytics")

from vehicle_detection import development_report as reporting  # noqa: E402
from vehicle_detection.common import digest, file_hash, read_json, write_json  # noqa: E402
from vehicle_detection.datasets import verify_role  # noqa: E402
from vehicle_detection.five_class import FIVE_CLASSES, SIX_TO_FIVE  # noqa: E402
from vehicle_detection.offline import export_variant, verify_alignment  # noqa: E402
from vehicle_detection.protocol import summarize  # noqa: E402


@pytest.fixture
def routed(frozen, tmp_path):
    protocol = copy.deepcopy(frozen[0])
    template = protocol["splits"]["development_validation"][0]
    rows, records = [], []
    conversion = tmp_path / "labels"
    write_json(
        conversion / "conversion.json",
        {
            "classes": FIVE_CLASSES,
            "mapping": {str(k): v for k, v in SIX_TO_FIVE.items()},
        },
    )
    for i in range(8):
        image = tmp_path / f"fixture-{i}.jpg"
        assert cv2.imwrite(str(image), np.full((64, 80, 3), i * 20, np.uint8))
        row = {
            **copy.deepcopy(template),
            "id": image.stem,
            "sha256": file_hash(image),
            "group": f"group-{i}",
            "source_group": f"source-{i}",
            "dhash64": digest(i),
            "boxes": [[0, 0, 0, 16, 16]] + ([[5, 0, 0, 16, 16]] if i < 4 else []),
        }
        row["attributes"]["timeofday"] = "night" if i % 2 else "daytime"
        rows.append(row)
        records.append({"id": row["id"], "path": str(image), "sha256": row["sha256"]})
        label = conversion / "labels/tune" / f"{row['id']}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("0 0.1 0.125 0.2 0.25\n" + ("4 0.1 0.125 0.2 0.25\n" if i < 4 else ""))
    protocol["splits"]["development_validation"] = rows
    protocol["summaries"]["development_validation"] = summarize(rows)
    protocol["manifest_id"] = digest({k: v for k, v in protocol.items() if k != "manifest_id"})
    recipes = {"raw": {}, "gamma": {"gamma": 0.85}}
    policy = {"policy_id": "fixture", "recipes": recipes}
    inputs = {t: tmp_path / t for t in ("raw", "adaptive")}
    for t, path in inputs.items():
        export_variant({"tune": records}, conversion, path, "raw_reencoded", recipes)
        if t == "adaptive":
            export = read_json(path / "export.json")
            export.update(variant="adaptive", recipe=recipes)
            write_json(path / "export.json", export)
            meta = pd.read_csv(path / "metadata.csv")
            meta["variant"] = "adaptive"
            meta["decisions"] = [
                json.dumps([{"name": "gamma", "params": recipes["gamma"]}]) if i < 4 else "[]"
                for i in range(8)
            ]
            meta.to_csv(path / "metadata.csv", index=False)
        verify_alignment([path])
        bound = verify_role(
            path,
            rows,
            "tune",
            "raw_reencoded" if t == "raw" else "adaptive",
            recipes if t == "adaptive" else None,
            deep=False,
        )
        for model in ("pretrained", "raw", "adaptive"):
            output = tmp_path / "evaluations" / f"{model}-{t}"
            output.mkdir(parents=True)
            arrays = {}
            for i, _row in enumerate(rows):
                labels = np.array([0, 4] if i < 4 else [0])
                values = {
                    "tp": np.full((len(labels), 10), t == "adaptive" or i >= 4, bool),
                    "conf": np.full(len(labels), 0.9),
                    "pred_cls": labels,
                    "target_cls": labels,
                }
                arrays.update({f"{k}_{i}": v for k, v in values.items()})
            np.savez_compressed(
                output / "paired_stats.npz", ids=np.array([r["id"] for r in rows]), **arrays
            )
            identity = {
                "role": "development_validation",
                "protocol_id": protocol["manifest_id"],
                "policy_id": "fixture",
                "dataset": bound,
                "class_names": FIVE_CLASSES,
                "settings": {"fixture": True},
                "checkpoint_sha256": digest(model),
            }
            result = {"artifacts": {"paired_stats.npz": file_hash(output / "paired_stats.npz")}}
            write_json(
                output / "native_result.json",
                {
                    "identity": identity,
                    "result": result,
                    "checksum": digest(result),
                },
            )
    return protocol, policy, inputs, tmp_path / "evaluations", tmp_path / "report"


def test_routes_pair_ids_and_reaggregate_native_subsets_without_inference(routed, monkeypatch):
    from ultralytics import YOLO

    def forbidden(*args, **kwargs):
        pytest.fail("Inference/training or cached reaggregation triggered")

    monkeypatch.setattr(YOLO, "predict", forbidden)
    monkeypatch.setattr(YOLO, "val", forbidden)
    monkeypatch.setattr(YOLO, "train", forbidden)
    result = reporting.route_report(*routed)
    assert [(r["images"], r["percent"], r["groups"]) for r in result["populations"]] == [
        (4, 50, 4)
    ] * 2
    assert sum(r["images"] for r in result["timeofday"]) == 8
    assert all(r["images"] == 2 for r in result["timeofday"])
    assert all(r["native_stat_mismatches"] == 0 for r in result["skip_control"])
    for r in result["comparisons"]:
        assert r["class_ids"] == ([0, 4] if r["route"] == "gamma" else [0])
        assert r["delta_mAP50_95"] == pytest.approx(0.995 if r["route"] == "gamma" else 0)
    absent = [r for r in result["classes"] if r["class"] == "bus"]
    assert absent and all(
        r["raw_AP50_95"] is None and r["adaptive_AP50_95"] is None for r in absent
    )
    monkeypatch.setattr(reporting, "class_ap", forbidden)
    monkeypatch.setattr(reporting, "load_stats", forbidden)
    assert reporting.route_report(*routed) == result


def test_routes_reject_unpaired_ids_and_changed_skip_export(routed):
    _, _, inputs, _, _ = routed
    path = inputs["adaptive"] / "metadata.csv"
    meta = pd.read_csv(path)
    broken = meta.copy()
    broken.loc[0, "id"] = "not-development"
    broken.to_csv(path, index=False)
    with pytest.raises(ValueError, match="frozen role IDs"):
        reporting.route_report(*routed)
    image = inputs["adaptive"] / "images/tune/fixture-7.jpg"
    assert cv2.imwrite(str(image), np.zeros((64, 80, 3), np.uint8))
    meta.loc[7, "output_sha256"] = file_hash(image)
    meta.to_csv(path, index=False)
    verify_alignment([inputs["adaptive"]])
    with pytest.raises(ValueError, match="Skip export bytes differ"):
        reporting.route_report(*routed)


def test_routes_flag_unexpected_native_skip_differences(routed):
    _, _, _, directory, _ = routed
    path = directory / "raw-adaptive/paired_stats.npz"
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    arrays["tp_7"][:] = False
    np.savez_compressed(path, **arrays)
    result_path = path.parent / "native_result.json"
    document = read_json(result_path)
    document["result"]["artifacts"][path.name] = file_hash(path)
    document["checksum"] = digest(document["result"])
    write_json(result_path, document)
    result = reporting.route_report(*routed)
    control = next(r for r in result["skip_control"] if r["model"] == "raw")
    assert control["native_stat_mismatches"] == 1 and control["example_ids"] == ["fixture-7"]
    skip = next(
        r
        for r in result["comparisons"]
        if r["model"] == "raw" and r["route"] == "skip" and r["timeofday"] == "all"
    )
    assert skip["delta_mAP50_95"] < 0
