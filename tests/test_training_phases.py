"""P4/P5 reuse must prove input parity and must never silently restart training."""

from copy import deepcopy
from importlib.metadata import version
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml
from test_adaptive import frozen as frozen
from test_finetuning import pair_inputs as pair_inputs
from test_study import bdd as bdd

from vehicle_detection.common import digest, file_hash, read_json, write_json
from vehicle_detection.training_preparation import prepare_training_data, validate_preparation
from vehicle_detection.training_report import training_evidence, verify_historical_reuse


@pytest.fixture
def phase_inputs(pair_inputs, bdd, tmp_path):
    protocol, subsets, policy = pair_inputs[:3]
    return (
        protocol,
        subsets,
        policy,
        bdd,
        tmp_path / "prepared",
        tmp_path / "conversion",
        tmp_path / "conversion",
    )


def test_prepare_and_reuse_preserve_sources_exports_and_receipts(phase_inputs, monkeypatch):
    import vehicle_detection.training_preparation as preparation

    receipt, routes = prepare_training_data(*phase_inputs)
    assert receipt["original_pixel_replay"] and receipt["paired_labels_geometry_ids"]
    assert sum(receipt["route_counts"].values()) == len(routes)
    assert receipt["exports"]["raw"]["ids"] == receipt["exports"]["adaptive"]["ids"]
    assert validate_preparation(receipt, phase_inputs[4]) == receipt
    before = {
        str(path): (file_hash(path), path.stat().st_mtime_ns)
        for root in (phase_inputs[3], phase_inputs[4])
        for path in root.rglob("*")
        if path.is_file()
    }
    monkeypatch.setattr(preparation, "scan_bdd", lambda *a, **k: pytest.fail("Rescanned sources"))
    assert prepare_training_data(*phase_inputs) == (receipt, routes)
    assert before == {
        str(path): (file_hash(path), path.stat().st_mtime_ns)
        for root in (phase_inputs[3], phase_inputs[4])
        for path in root.rglob("*")
        if path.is_file()
    }
    changed = deepcopy(receipt)
    changed["route_counts"] = {}
    with pytest.raises(ValueError, match="receipt changed"):
        validate_preparation(changed, phase_inputs[4])


def test_pixel_replay_rejects_false_route_metadata(phase_inputs):
    prepare_training_data(*phase_inputs)
    root = phase_inputs[4] / "datasets/S10/adaptive"
    meta = pd.read_csv(root / "metadata.csv")
    meta.loc[0, "decisions"] = '[{"name": "fake"}]'
    meta.to_csv(root / "metadata.csv", index=False)
    verified = read_json(root / "verified.json")
    verified["metadata_sha256"] = file_hash(root / "metadata.csv")
    write_json(root / "verified.json", verified)
    with pytest.raises(ValueError, match="route replay differs"):
        prepare_training_data(*phase_inputs)


def test_final_role_rejected_before_pixels(phase_inputs, monkeypatch):
    import vehicle_detection.datasets as datasets

    prepare_training_data(*phase_inputs)
    root = phase_inputs[4] / "datasets/S10/raw"
    meta = pd.read_csv(root / "metadata.csv")
    meta.loc[0, "id"] = phase_inputs[0]["splits"]["final_test"][0]["id"]
    meta.to_csv(root / "metadata.csv", index=False)
    monkeypatch.setattr(datasets, "verify_alignment", lambda *a, **k: pytest.fail("Pixels read"))
    with pytest.raises(ValueError, match="frozen role IDs"):
        prepare_training_data(*phase_inputs)


def test_historical_reuse_authenticates_every_native_artifact(tmp_path):
    write_json(tmp_path / "pair.json", {"fixture": True})
    historical = {"pair_manifest_sha256": file_hash(tmp_path / "pair.json"), "branches": {}}
    for branch in ("raw", "adaptive"):
        root = tmp_path / branch
        (root / "weights").mkdir(parents=True)
        for filename in ("args.yaml", "results.csv", "weights/best.pt", "weights/last.pt"):
            (root / filename).write_text("fixture " + filename)
        historical["branches"][branch] = {
            "args_sha256": file_hash(root / "args.yaml"),
            "history_sha256": file_hash(root / "results.csv"),
            "checkpoints": {
                name: {"sha256": file_hash(root / f"weights/{name}.pt")}
                for name in ("best", "last")
            },
        }
    verify_historical_reuse(tmp_path, historical)
    (tmp_path / "adaptive/weights/last.pt").write_bytes(b"foreign checkpoint")
    with pytest.raises(ValueError, match="cannot claim reuse"):
        verify_historical_reuse(tmp_path, historical)


def test_p4_receipt_rejects_artifact_escape(tmp_path):
    receipt = {
        "version": "round2-training-data-v1",
        "classes": ["car", "truck", "bus", "motorcycle", "bicycle"],
        "artifacts": {"../elsewhere": "wrong"},
        "exports": {},
    }
    receipt["preparation_id"] = digest(receipt)
    with pytest.raises(ValueError, match="artifact changed"):
        validate_preparation(receipt, tmp_path)


@pytest.mark.parametrize("augmentations", [None, [], ["unexpected"]])
def test_checkpoint_empty_augmentation_serialization_only(tmp_path, monkeypatch, augmentations):
    import vehicle_detection.training_report as report

    settings = {"augmentations": []}
    experiment = {
        "settings": settings,
        "training": {"raw": {}, "adaptive": {}},
        "development": {},
        "protocol_id": "protocol",
        "subsets_id": "subsets",
        "policy_id": "v2",
        "freeze": 11,
        "checkpoint_sha256": "pretrained",
    }
    write_json(tmp_path / "pair.json", experiment)
    preparation = {
        **experiment,
        "exports": {"raw": {}, "adaptive": {}, "development": {}},
        "preparation_id": "p4",
    }
    completed = {b: {"best_epoch": 1, "final_epoch": 1} for b in ("raw", "adaptive")}
    for branch in completed:
        native = tmp_path / branch
        (native / "weights").mkdir(parents=True)
        args = {**settings, "freeze": 11, "data": str(tmp_path / f"{branch}.yaml")}
        (native / "args.yaml").write_text(yaml.safe_dump(args))
        pd.DataFrame(
            {"epoch": [1], "metrics/mAP50-95(B)": [0.1], "metrics/mAP50(B)": [0.2]}
        ).to_csv(native / "results.csv", index=False)
        for name in ("best", "last"):
            (native / f"weights/{name}.pt").write_bytes(b"fixture")

    def load(path):
        branch = path.split("/")[-3]
        return SimpleNamespace(
            names=dict(enumerate(["car", "truck", "bus", "motorcycle", "bicycle"])),
            model=SimpleNamespace(
                yaml={"yaml_file": "yolo26n.yaml", "end2end": True, "backbone": [None] * 11},
                model=[None] * 12,
            ),
            ckpt={
                "version": version("ultralytics"),
                "train_args": {
                    "augmentations": augmentations,
                    "freeze": 11,
                    "data": str(tmp_path / f"{branch}.yaml"),
                },
                "train_metrics": {"metrics/mAP50-95(B)": 0.1, "metrics/mAP50(B)": 0.2},
            },
        )

    monkeypatch.setattr(report, "YOLO", load)
    if augmentations:
        with pytest.raises(ValueError, match="checkpoint head/version/settings"):
            training_evidence(experiment, preparation, completed, tmp_path)
    else:
        assert training_evidence(experiment, preparation, completed, tmp_path)["next_issue"] == 35
