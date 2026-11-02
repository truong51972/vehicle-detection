import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from vehicle_detection.common import read_json
from vehicle_detection.data import (
    CLASSES,
    assert_disjoint,
    box_labels,
    discover,
    frames,
    load_manifest,
    partition,
    prepare,
    survey,
    yolo_labels,
)


def test_discover_nested_bdd_and_partition(bdd, config):
    records, audit, sources = discover(bdd)
    assert len(records) == 12
    assert audit["ignored_category:lane"] == 12
    assert len(sources) == 2
    result = partition(records, 42, config["data"])
    assert result == partition(list(reversed(records)), 42, config["data"])
    assert result != partition(records, 43, config["data"])
    assert len(result["train"]) == 6
    assert len(result["tune"]) == 2
    assert {r["official_split"] for r in result["holdout"]} == {"val"}
    assert_disjoint(result)


def test_yolo_geometry_and_categories():
    from collections import Counter

    audit = Counter()
    labels = [
        {"category": name, "box2d": {"x1": 0, "y1": 0, "x2": 80, "y2": 40}}
        for name in ("car", "truck", "bus", "train", "motor", "bike", "motorcycle", "bicycle")
    ]
    boxes, invalid = box_labels(labels, audit)
    assert not invalid
    assert [b[0] for b in boxes] == [0, 1, 2, 3, 4, 5, 4, 5]
    text = yolo_labels([[0, 10, 12, 50, 40]], 80, 64)
    values = [float(v) for v in text.split()]
    np.testing.assert_allclose(values, [0, 0.375, 0.40625, 0.5, 0.4375])
    assert yolo_labels([], 80, 64) == ""


@pytest.mark.parametrize(
    "box",
    [
        [0, -1, 0, 30, 30],
        [0, 0, 0, 81, 40],
        [0, 30, 0, 20, 20],
        [0, 0, 0, float("nan"), 40],
        [6, 0, 0, 20, 20],
    ],
)
def test_invalid_boxes_are_never_clipped(box):
    with pytest.raises(ValueError, match="invalid box"):
        yolo_labels([box], 80, 64)


@pytest.mark.parametrize("wrapper", ["array", "scalabel", "single", "old"])
def test_stream_label_schemas(tmp_path, wrapper):
    record = {"name": "x.png", "labels": [{"category": "car", "box2d": {}}]}
    if wrapper == "array":
        obj = [record]
    elif wrapper == "scalabel":
        obj = {"config": {}, "frames": [record]}
    elif wrapper == "old":
        obj = {"name": "x.png", "frames": [{"objects": record["labels"]}]}
    else:
        obj = record
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(obj))
    output = list(frames(path))
    assert len(output) == 1
    assert output[0]["name"] == "x.png"


def test_missing_detection_annotations_fails(tmp_path):
    (tmp_path / "train").mkdir()
    (tmp_path / "train/tags.json").write_text('[{"name":"x.jpg","attributes":{}}]')
    with pytest.raises(ValueError, match="box2d annotations required"):
        discover(tmp_path)


def test_invalid_vehicle_image_excluded_instead_of_false_negative(bdd):
    path = next(bdd.rglob("*train.json"))
    labels = json.loads(path.read_text())
    labels[0]["labels"][0]["box2d"]["x2"] = -10
    path.write_text(json.dumps(labels))
    records, audit, _ = discover(bdd)
    assert len(records) == 11
    assert audit["invalid_boxes"] == 1
    assert audit["excluded_images_with_invalid_vehicle_labels"] == 1


def test_canonical_100k_wins_over_segmentation_subset(bdd):
    original = next(bdd.rglob("train_0.png"))
    duplicate = bdd / "seg/images/train/train_0.png"
    duplicate.parent.mkdir(parents=True)
    duplicate.write_bytes(b"different subset copy")
    records, audit, _ = discover(bdd)
    record = next(r for r in records if r["id"] == "train_0")
    assert record["path"] == str(original)
    assert audit["subset_copies_ignored_for_100k"] == 1


def test_prepare_and_source_integrity(bdd, tmp_path, config):
    target = tmp_path / "prepared"
    manifest = prepare(bdd, target, config)
    assert load_manifest(target) == manifest
    dataset = yaml.safe_load((target / "dataset.yaml").read_text())
    assert dataset["val"] == "images/tune"
    assert "holdout" not in dataset.values()
    assert list(dataset["names"].values()) == CLASSES
    image = manifest["partitions"]["train"][0]
    assert (target / "images/train" / Path(image["path"]).name).is_symlink()
    with pytest.raises(ValueError, match="already exists"):
        prepare(bdd, target, config)
    Path(image["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Source image changed"):
        load_manifest(target)


def test_manifest_tampering_fails(bdd, tmp_path, config):
    target = tmp_path / "prepared"
    prepare(bdd, target, config)
    obj = read_json(target / "manifest.json")
    obj["seed"] = 99
    (target / "manifest.json").write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="Manifest identity"):
        load_manifest(target)


def test_duplicate_image_bytes_leakage_fails(bdd, tmp_path, config):
    train_image = next(bdd.rglob("train_0.png"))
    val_image = next(bdd.rglob("val_0.png"))
    shutil.copyfile(train_image, val_image)
    with pytest.raises(ValueError, match="leakage on sha256"):
        prepare(bdd, tmp_path / "prepared", config)
    assert not (tmp_path / "prepared").exists()


def test_shared_video_never_crosses_official_split(bdd, config):
    records, _, _ = discover(bdd)
    for record in records:
        record["group"] = "same-video"
    with pytest.raises(ValueError, match="Video/group leakage"):
        partition(records, 42, config["data"])


def test_group_partition_independent_of_annotation_order(bdd, config):
    records, _, _ = discover(bdd)
    for index, record in enumerate(r for r in records if r["official_split"] == "train"):
        record["group"] = f"video-{index // 2}"
    forward = partition(records, 42, config["data"])
    assert forward == partition(list(reversed(records)), 42, config["data"])
    assert_disjoint(forward)


def test_survey_artifacts(bdd, tmp_path, config):
    output = tmp_path / "survey"
    result = survey(bdd, output, config)
    assert result["counts"]["train"]["boxes"] == 8
    assert len(result["quality_health"]) == 4
    assert all(r["status"] == "ok" for r in result["quality_health"])
    assert result["warnings"]
    assert (output / "survey.png").is_file()
    assert (output / "boxes.csv").is_file()
