"""Five-class labels are derived without modifying historical six-class evidence."""

import cv2
import numpy as np
import pytest
from test_study import recipes as recipes

from vehicle_detection.common import file_hash, write_json
from vehicle_detection.datasets import verify_role
from vehicle_detection.five_class import (
    FIVE_CLASSES,
    derive_labels,
    remap_boxes,
    support_audit,
)
from vehicle_detection.offline import export_variant, verify_alignment


def test_label_remapping_keeps_order_coordinates_and_negative_frames(tmp_path):
    source, destination = tmp_path / "original", tmp_path / "derived"
    write_json(
        source / "conversion.json",
        {
            "classes": ["car", "truck", "bus", "train", "motorcycle", "bicycle"],
            "box_convention": "continuous",
        },
    )
    rows = [
        {"id": "mixed", "boxes": [[0, 1, 2, 9, 12], [3, 1, 1, 4, 4], [4, 2, 3, 8, 10]]},
        {"id": "rail-only", "boxes": [[3, 2, 3, 4, 5]]},
        {"id": "empty", "boxes": []},
    ]
    labels = {
        "mixed": "0 0.125 0.25 0.1 0.2\n3 0.3 0.2 0.1 0.1\n4 0.51 0.62 0.2 0.2\n",
        "rail-only": "3 0.5 0.5 0.1 0.1\n",
        "empty": "",
    }
    for sample_id, value in labels.items():
        path = source / "labels/train" / f"{sample_id}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    before = {name: file_hash(source / "labels/train" / f"{name}.txt") for name in labels}
    identity = derive_labels(source, destination, {"train": rows})
    assert identity == derive_labels(source, destination, {"train": rows})
    assert (destination / "labels/train/mixed.txt").read_text() == (
        "0 0.125 0.25 0.1 0.2\n3 0.51 0.62 0.2 0.2\n"
    )
    assert (destination / "labels/train/rail-only.txt").read_text() == ""
    assert (destination / "labels/train/empty.txt").read_text() == ""
    assert remap_boxes(rows[0]["boxes"]) == [[0, 1, 2, 9, 12], [3, 2, 3, 8, 10]]
    audited = support_audit(
        [
            {**row, "attributes": {"weather": "clear", "timeofday": "daytime", "scene": "city"}}
            for row in rows
        ]
    )
    assert audited["newly_negative_images"] == 1
    assert audited["old_instances:3"] == 2
    assert audited["new_instances:car"] == 1
    assert audited["new_instances:motorcycle"] == 1
    assert [file_hash(source / "labels/train" / f"{name}.txt") for name in labels] == [
        before[name] for name in labels
    ]
    with pytest.raises(ValueError, match="Only training/development"):
        derive_labels(source, tmp_path / "holdout", {"test": rows})
    with pytest.raises(ValueError, match="independent directory"):
        derive_labels(source, source / "five", {"train": rows})
    (destination / "labels/train/mixed.txt").write_text("tampered\n")
    with pytest.raises(ValueError, match="Derived label changed"):
        derive_labels(source, destination, {"train": rows})


def test_five_class_export_reuses_native_verification_and_keeps_six_class_source(tmp_path, recipes):
    original, derived = tmp_path / "original", tmp_path / "five"
    write_json(
        original / "conversion.json",
        {
            "classes": ["car", "truck", "bus", "train", "motorcycle", "bicycle"],
            "box_convention": "continuous",
        },
    )
    images, records, rows = tmp_path / "images", [], []
    images.mkdir()
    for sample_id, six_classes in [("mixed", [0, 3]), ("rail-only", [3])]:
        image = images / f"{sample_id}.jpg"
        assert cv2.imwrite(str(image), np.full((24, 32, 3), 90, dtype=np.uint8))
        records.append({"id": sample_id, "path": str(image), "sha256": file_hash(image)})
        rows.append(
            {
                "id": sample_id,
                "sha256": file_hash(image),
                "width": 32,
                "height": 24,
                "boxes": [[cls, 0, 0, 16, 12] for cls in six_classes],
            }
        )
        label = original / "labels/train" / f"{sample_id}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("".join(f"{cls} 0.25 0.25 0.5 0.5\n" for cls in six_classes))
    derive_labels(original, derived, {"train": rows})
    paths = [tmp_path / "raw", tmp_path / "adaptive"]
    for path, variant in zip(paths, ("raw_reencoded", "adaptive"), strict=True):
        export_variant({"train": records}, derived, path, variant, recipes)
        verify_role(
            path,
            rows,
            "train",
            variant,
            recipes if variant == "adaptive" else None,
        )
    assert verify_alignment(paths) == 2
    assert [p.read_text() for p in (paths[0] / "labels/train").glob("*.txt")] != []
    assert (paths[0] / "labels/train/rail-only.txt").read_text() == ""
    assert (paths[0] / "labels/train/mixed.txt").read_text() == (
        paths[1] / "labels/train/mixed.txt"
    ).read_text()
    assert (original / "labels/train/rail-only.txt").read_text().startswith("3 ")
    # Existing verification reads class names directly from the export identity.
    assert verify_alignment(paths) == 2
