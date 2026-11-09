"""Stream local BDD labels and map them to the frozen five-class taxonomy."""

import re
from collections import Counter
from pathlib import Path

import ijson
import numpy as np

from .common import file_hash

CLASSES = ["car", "truck", "bus", "train", "motorcycle", "bicycle"]
ALIASES = {"motor": "motorcycle", "bike": "bicycle"}
FIVE_CLASSES = ["car", "truck", "bus", "motorcycle", "bicycle"]
# Railway `train` is dropped; images that become negative are kept.
SIX_TO_FIVE = {0: 0, 1: 1, 2: 2, 3: None, 4: 3, 5: 4}


def remap_boxes(boxes):
    return [[SIX_TO_FIVE[c], *xyxy] for c, *xyxy in boxes if SIX_TO_FIVE[c] is not None]


def scan_bdd(root, *, strict=True):
    root = Path(root).resolve()
    image_roots = list(root.glob("**/images/100k"))
    if len(image_roots) != 1:
        raise ValueError("Expected exactly one canonical images/100k directory under dataset root")
    paths = {}
    for split in ("train", "val"):
        for path in (image_roots[0] / split).rglob("*"):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                key = (split, path.name)
                if key in paths:
                    raise ValueError(f"Ambiguous image: {key}")
                paths[key] = str(path)
    records, audit, sources, seen, missing_examples = [], Counter(), {}, set(), []
    missing_ids = {"train": [], "val": []}
    for split in ("train", "val"):
        candidates = [
            p
            for p in root.rglob("*.json")
            if p.stem in {f"bdd100k_labels_images_{split}", f"det_{split}"}
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"Install one official {split} detection JSON; images alone are insufficient"
            )
        path = candidates[0]
        with path.open("rb") as stream:
            prefix = "item" if stream.read(4096).lstrip().startswith(b"[") else "frames.item"
            stream.seek(0)
            for frame in ijson.items(stream, prefix, use_float=True):
                audit[f"frames:{split}"] += 1
                if "labels" not in frame or not isinstance(frame["labels"], list):
                    raise ValueError(
                        "Expected BDD 2018 / Scalabel 2020 detection frames with labels"
                    )
                name = Path(frame["name"]).name
                sample_id = Path(name).stem
                if sample_id in seen:
                    raise ValueError(f"Duplicate/cross-split ID: {sample_id}")
                seen.add(sample_id)
                if (split, name) not in paths:
                    audit[f"missing_images:{split}"] += 1
                    missing_ids[split].append(sample_id)
                    if len(missing_examples) < 10:
                        missing_examples.append(sample_id)
                    continue
                labels = []
                for label in frame["labels"]:
                    category = ALIASES.get(label.get("category"), label.get("category"))
                    if category not in CLASSES:
                        audit["ignored_nonvehicle_or_polygon_labels"] += 1
                        continue
                    if strict and not isinstance(label.get("box2d"), dict):
                        raise ValueError(f"Vehicle without box2d: {sample_id}")
                    labels.append(label)
                records.append(
                    {
                        "id": sample_id,
                        "name": name,
                        "path": paths[split, name],
                        "official_split": split,
                        "group": frame.get("videoName") or sample_id,
                        "group_source": "videoName" if frame.get("videoName") else "sample_id",
                        "attributes": frame.get("attributes") or {},
                        "labels": labels,
                    }
                )
        sources[split] = {"path": str(path), "sha256": file_hash(path), "schema": prefix}
    if any(
        not any(r["labels"] for r in records if r["official_split"] == split)
        for split in ("train", "val")
    ):
        raise ValueError("Both official splits require usable detection box2d annotations")
    return (
        sorted(records, key=lambda r: r["id"]),
        {
            **dict(audit),
            "missing_image_examples": missing_examples,
            "missing_image_ids": missing_ids,
        },
        sources,
    )


def group_key(record):
    if record["group"] != record["id"] or record.get("group_source") == "videoName":
        return "video:" + record["group"]
    if re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{8}", record["id"]):
        return "bdd_prefix:" + record["id"].split("-")[0]
    return "sample:" + record["id"]


def annotation_profile(record, width, height):
    boxes, excluded, errors = [], Counter(), Counter()
    for label in record["labels"]:
        attrs = label.get("attributes") or {}
        if attrs.get("crowd") or attrs.get("ignored"):
            excluded["crowd_or_ignored"] += 1
            continue
        try:
            cls = CLASSES.index(ALIASES.get(label["category"], label["category"]))
            coords = [float(label["box2d"][k]) for k in ("x1", "y1", "x2", "y2")]
        except (KeyError, TypeError, ValueError, OverflowError):
            errors["malformed_box"] += 1
            continue
        x1, y1, x2, y2 = coords
        if not np.isfinite(coords).all():
            errors["nonfinite_box"] += 1
        elif not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            errors["invalid_geometry"] += 1
        else:
            boxes.append([cls, *coords])
    unique = list(dict.fromkeys(map(tuple, boxes)))
    excluded["duplicate_boxes"] = len(boxes) - len(unique)
    return [list(b) for b in unique], dict(excluded), dict(errors)
