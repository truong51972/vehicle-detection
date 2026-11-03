"""The new protocol must not open final-test images, even during integrity checks."""

import copy
from pathlib import Path

import pytest
from test_study import bdd as bdd

from vehicle_detection import protocol
from vehicle_detection.common import digest, read_json
from vehicle_detection.data import scan_bdd
from vehicle_detection.protocol import (
    build_protocol,
    class_condition_audit,
    development_records,
    freeze_protocol,
    group_key,
    historical_comparison,
    validate_protocol,
)


def test_reproducible_frozen_roles_without_holdout_pixels(bdd, monkeypatch, tmp_path):
    records, discovery, sources = scan_bdd(bdd)
    read_image, file_hash = protocol.read_image, protocol.file_hash

    def train_only(path):
        assert "val" not in Path(path).parts
        return read_image(path)

    def safe_hash(path):
        assert not (Path(path).suffix == ".jpg" and "val" in Path(path).parts)
        return file_hash(path)

    monkeypatch.setattr(protocol, "read_image", train_only)
    monkeypatch.setattr(protocol, "file_hash", safe_hash)
    kwargs = {"excluded_test_ids": ["val_0"], "discovery": discovery}
    first = build_protocol(records, sources, bdd, **kwargs)
    assert first == build_protocol(list(reversed(records)), sources, bdd, **kwargs)
    assert len(first["splits"]["final_test"]) == 2
    assert sum(len(first["splits"][role]) for role in protocol.FRACTIONS) == 6
    assert first["summaries"]["official_train"]["instances:car"] == 6
    assert all(
        d["inadequate_support"]
        for d in first["diagnostics"]
        if d["feature"].startswith("presence:")
    )
    path = tmp_path / "protocol.json"
    freeze_protocol(path, first)
    freeze_protocol(path, first)
    assert read_json(path) == first
    changed = copy.deepcopy(first)
    changed["seed"] += 1
    with pytest.raises(ValueError, match="checksum"):
        validate_protocol(changed)
    changed["manifest_id"] = digest({k: v for k, v in changed.items() if k != "manifest_id"})
    with pytest.raises(ValueError, match="Immutable"):
        freeze_protocol(path, changed)
    selected = development_records(first, records, bdd)
    assert {r["id"] for r in selected["tune"]} == {
        r["id"] for r in first["splits"]["development_validation"]
    }
    selected_id = selected["tune"][0]["id"]
    next(r for r in records if r["id"] == selected_id)["labels"] = []
    with pytest.raises(ValueError, match="source/labels changed"):
        development_records(first, records, bdd)


def test_duplicate_candidates_stay_together_and_leakage_is_rejected(bdd):
    records, _, sources = scan_bdd(bdd)
    train = [r for r in records if r["official_split"] == "train"]
    Path(train[1]["path"]).write_bytes(Path(train[0]["path"]).read_bytes())
    manifest = build_protocol(records, sources, bdd, [])
    roles = {r["id"]: role for role, rows in manifest["splits"].items() for r in rows}
    assert roles[train[0]["id"]] == roles[train[1]["id"]]
    assert manifest["group_audit"]["duplicate_candidates"]["sha256"] == 1
    duplicate = copy.deepcopy(manifest["splits"]["development_validation"][0])
    manifest["splits"]["training_pool"].append(duplicate)
    manifest["summaries"]["training_pool"] = protocol.summarize(manifest["splits"]["training_pool"])
    manifest["manifest_id"] = digest({k: v for k, v in manifest.items() if k != "manifest_id"})
    with pytest.raises(ValueError, match="leakage"):
        validate_protocol(manifest)


def test_group_fallback_exposure_groups_and_invalid_boxes(bdd):
    records, _, sources = scan_bdd(bdd)
    val = [r for r in records if r["official_split"] == "val"]
    for r in val[:2]:
        r["group"] = "shared-video"
    train = next(r for r in records if r["official_split"] == "train")
    train["labels"][0]["box2d"]["x2"] = -1
    manifest = build_protocol(records, sources, bdd, [val[0]["id"]])
    assert len(manifest["splits"]["final_test"]) == 1
    assert {r["reason"] for r in manifest["exclusions"]} == {
        "prior_exposure",
        "group_related_to_exposure",
        "invalid_boxes",
    }
    assert manifest["label_audit"]["invalid_geometry"] == 1
    assert (
        group_key({"id": "abcdef12-34567890", "group": "abcdef12-34567890"})
        == "bdd_prefix:abcdef12"
    )
    assert group_key({"id": "fixture", "group": "fixture"}) == "sample:fixture"


def test_discovery_can_account_for_missing_and_malformed_inputs(bdd):
    import json

    path = bdd / "bdd100k_labels_images_train.json"
    frames = json.loads(path.read_text())
    del frames[0]["labels"][0]["box2d"]
    path.write_text(json.dumps(frames))
    (bdd / "images/100k/val/val_0.jpg").unlink()
    with pytest.raises(ValueError, match="without box2d"):
        scan_bdd(bdd)
    records, discovery, sources = scan_bdd(bdd, strict=False)
    manifest = build_protocol(records, sources, bdd, ["val_0"], discovery=discovery)
    assert discovery["missing_image_ids"]["val"] == ["val_0"]
    assert manifest["label_audit"]["malformed_box"] == 1
    assert len(manifest["splits"]["final_test"]) == 2


def test_rare_class_zero_support_is_reported_without_fabricated_samples(bdd):
    records, _, sources = scan_bdd(bdd)
    for record in records:
        if record["official_split"] == "train":
            record["labels"] = [label for label in record["labels"] if label["category"] != "train"]
    manifest = build_protocol(records, sources, bdd, [])
    rare = [d for d in manifest["diagnostics"] if d["feature"] == "presence:train"]
    assert len(rare) == 3 and all(d["support"] == 0 and d["inadequate_support"] for d in rare)
    assert all(d["fallback"] for d in rare)


def test_membership_and_identity_survive_a_dataset_move(bdd, tmp_path):
    import shutil

    records, discovery, sources = scan_bdd(bdd)
    first = build_protocol(records, sources, bdd, [], discovery=discovery)
    moved = tmp_path / "new-location/bdd"
    shutil.move(bdd, moved)
    with pytest.raises(ValueError):
        build_protocol(records, sources, moved, [], discovery=discovery)
    new_records, new_discovery, new_sources = scan_bdd(moved)
    assert first == build_protocol(new_records, new_sources, moved, [], discovery=new_discovery)


def test_optional_legacy_file_cannot_change_canonical_identity(bdd, tmp_path):
    from vehicle_detection.common import write_json

    records, discovery, sources = scan_bdd(bdd)
    legacy = tmp_path / "runs/phase2-full/selection.json"
    write_json(legacy, {"selection_id": "legacy", "splits": {"train": ["train_0"]}})
    with_history = build_protocol(records, sources, bdd, [], discovery=discovery)
    report = historical_comparison(with_history, read_json(legacy))
    legacy.unlink()
    without_history = build_protocol(records, sources, bdd, [], discovery=discovery)
    assert with_history["manifest_id"] == without_history["manifest_id"]
    assert with_history["splits"] == without_history["splits"]
    assert with_history == without_history
    assert "historical_diff" not in with_history
    assert report["selection_id"] == "legacy"
    assert sum(report["transitions"].values()) == 6
    assert report["manifest_id"] == with_history["manifest_id"]


def test_joint_class_support_counts_zero_cells_and_drift_without_pixels(bdd, monkeypatch):
    records, _, sources = scan_bdd(bdd)
    for row in records:
        row["attributes"] = {"weather": "clear", "timeofday": "daytime"}
        if row["official_split"] == "train":
            row["labels"] = [row["labels"][0]]
    records[0]["attributes"] = {"weather": "rainy", "timeofday": "night"}
    records[0]["labels"].append(
        {**records[0]["labels"][0], "box2d": {"x1": 2, "y1": 2, "x2": 31, "y2": 40}}
    )
    manifest = build_protocol(records, sources, bdd, [])
    before = copy.deepcopy(manifest)

    def no_pixels(*args, **kwargs):
        pytest.fail("Joint audit must use frozen annotations only")

    monkeypatch.setattr(protocol, "read_image", no_pixels)
    monkeypatch.setattr(protocol, "file_hash", no_pixels)
    audit = class_condition_audit(manifest)
    assert manifest == before
    assert {row["role"] for row in audit} == {*protocol.ROLES, "official_train"}
    assert len(audit) == 5 * 6 * 2 * 2  # Include zero cells for every class/condition/role.
    for role in [*protocol.ROLES, "official_train"]:
        for axis in ["weather", "timeofday"]:
            car = [
                r for r in audit if r["role"] == role and r["axis"] == axis and r["class"] == "car"
            ]
            summary = manifest["summaries"][role]
            assert sum(r["class_images"] for r in car) == summary["presence:car"]
            assert sum(r["instances"] for r in car) == summary["instances:car"]
            assert sum(r["condition_images"] for r in car) == summary["images"]
    rain = next(
        r
        for r in audit
        if r["role"] == "official_train"
        and r["axis"] == "weather"
        and r["condition"] == "rainy"
        and r["class"] == "car"
    )
    assert (rain["condition_images"], rain["class_images"], rain["instances"]) == (1, 1, 2)
    assert rain["rare_support"] and not rain["zero_support"] and rain["presence_drift_pp"] == 0
    empty = next(
        r
        for r in audit
        if r["role"] == "training_pool" and r["axis"] == "weather" and r["class"] == "train"
    )
    assert empty["zero_support"] and empty["rare_support"]
    absent_condition = next(
        r for r in audit if r["role"] in protocol.FRACTIONS and r["condition_images"] == 0
    )
    assert absent_condition["conditional_presence_drift_pp"] is None
    drift = next(
        r
        for r in audit
        if r["role"] in protocol.FRACTIONS
        and r["axis"] == "weather"
        and r["class"] == "car"
        and r["condition"] == "rainy"
        and r["class_images"]
    )
    assert drift["presence_drift_pp"] == pytest.approx(100 * (1 / drift["images"] - 1 / 6))
    assert drift["instances_drift_per_image"] == pytest.approx(2 / drift["images"] - 2 / 6)
