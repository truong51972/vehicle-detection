"""BDD 2018/Scalabel discovery, audit, disjoint manifests, and YOLO export."""

import csv
import math
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import ijson
import numpy as np
import yaml

from .common import digest, file_hash, read_image, read_json, write_json
from .preprocess import quality

CLASSES = ["car", "truck", "bus", "train", "motorcycle", "bicycle"]
CATEGORY_MAP = {name: i for i, name in enumerate(CLASSES)} | {"motor": 4, "bike": 5}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def official_split(path):
    tokens = re.split(r"[/_.\-]+", str(path).lower())
    splits = set(tokens) & {"train", "val"}
    if len(splits) != 1:
        return None
    return splits.pop()


def frames(path):
    """Stream split arrays, Scalabel {frames: [...]}, and single-frame JSON files."""
    with path.open("rb") as stream:
        first = stream.read(4096).lstrip()[:1]
        stream.seek(0)
        if first == b"[":
            prefix = "item"
        elif first == b"{":
            keys = (
                value
                for prefix, event, value in ijson.parse(stream)
                if prefix == "" and event == "map_key"
            )
            key = next((k for k in keys if k in {"name", "frames"}), None)
            prefix = "" if key == "name" else "frames.item"
            stream.seek(0)
        else:
            raise ValueError(f"Invalid annotation JSON: {path}")
        yield from ijson.items(stream, prefix, use_float=True)


def box_labels(labels, audit):
    boxes, invalid = [], False
    for label in labels:
        if not isinstance(label, dict):
            audit["malformed_labels"] += 1
            invalid = True
            continue
        category = label.get("category", "")
        if not isinstance(category, str):
            audit["malformed_labels"] += 1
            invalid = True
            continue
        if category not in CATEGORY_MAP:
            audit[f"ignored_category:{category}"] += 1
            continue
        box = label.get("box2d")
        if not isinstance(box, dict):
            audit["vehicle_without_box2d"] += 1
            invalid = True
            continue
        try:
            xyxy = [float(box[k]) for k in ("x1", "y1", "x2", "y2")]
            if not all(math.isfinite(v) for v in xyxy) or xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]:
                raise ValueError("Invalid box")
        except (KeyError, TypeError, ValueError, OverflowError):
            audit["invalid_boxes"] += 1
            invalid = True
            continue
        boxes.append([CATEGORY_MAP[category], *xyxy])
    return boxes, invalid


def discover(root):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {root}")
    images, annotations = defaultdict(list), []
    for directory, directories, files in os.walk(root):
        directories.sort()
        for filename in sorted(files):
            path = Path(directory) / filename
            split = official_split(path.relative_to(root))
            if not split:
                continue
            if path.suffix.lower() in IMAGE_SUFFIXES:
                images[split, filename].append(path)
            elif path.suffix.lower() == ".json":
                annotations.append((split, path))
    audit = Counter()
    records, source_files, seen, box_splits = [], [], set(), set()
    for split, path in annotations:
        count = 0
        try:
            for frame in frames(path):
                if not isinstance(frame, dict) or not isinstance(frame.get("name"), str):
                    audit["non_frame_records"] += 1
                    continue
                labels = frame.get("labels")
                if labels is None and isinstance(frame.get("frames"), list):
                    old = frame["frames"]
                    if len(old) != 1 or not isinstance(old[0], dict):
                        raise ValueError(f"Expected one detection frame per image in {path}")
                    labels = old[0].get("objects")
                if not isinstance(labels, list):
                    audit["missing_detection_labels"] += 1
                    continue
                if any(
                    isinstance(label, dict) and isinstance(label.get("box2d"), dict)
                    for label in labels
                ):
                    box_splits.add(split)
                count += 1
                name = Path(frame["name"]).name
                sample_id = Path(name).stem
                if sample_id in seen:
                    raise ValueError(
                        f"Duplicate image ID {sample_id}; isolate one detection label set"
                    )
                seen.add(sample_id)
                candidates = images.get((split, name), [])
                if not candidates:
                    audit[f"missing_images:{split}"] += 1
                    continue
                if len(candidates) > 1:
                    canonical = [p for p in candidates if "images/100k/" in p.as_posix()]
                    if len(canonical) == 1:
                        audit["subset_copies_ignored_for_100k"] += len(candidates) - 1
                        candidates = canonical
                    elif len({file_hash(p) for p in candidates}) == 1:
                        audit["identical_image_copies_deduplicated"] += len(candidates) - 1
                        candidates = candidates[:1]
                if len(candidates) != 1:
                    raise ValueError(f"Ambiguous image {name}: {candidates}")
                boxes, invalid = box_labels(labels, audit)
                if invalid:
                    audit["excluded_images_with_invalid_vehicle_labels"] += 1
                    continue
                attributes = frame.get("attributes") or {}
                if not isinstance(attributes, dict):
                    raise ValueError(f"Expected frame attributes mapping for {name} in {path}")
                records.append(
                    {
                        "id": sample_id,
                        "path": str(candidates[0]),
                        "official_split": split,
                        "group": str(frame.get("videoName") or sample_id),
                        "boxes": boxes,
                        "attributes": {
                            key: str(attributes.get(key, "unknown"))
                            for key in ("weather", "timeofday", "scene")
                        },
                    }
                )
        except (ijson.JSONError, UnicodeError) as exc:
            raise ValueError(f"Cannot parse detection labels {path}: {exc}") from exc
        if count:
            source_files.append({"path": str(path), "sha256": file_hash(path), "frames": count})
            audit[f"annotated_frames:{split}"] += count
    if box_splits != {"train", "val"}:
        raise ValueError(
            "Detection box2d annotations required for official train AND val. "
            "Install bdd100k_labels_images_train.json and "
            "bdd100k_labels_images_val.json (or Scalabel det_train/val). "
            "Images, segmentation masks, and image-level tags alone are insufficient."
        )
    if any(not any(r["official_split"] == split for r in records) for split in ("train", "val")):
        raise ValueError(
            "No usable train/val images match detection annotations; check nested layout"
        )
    audit["discovered_images"] = sum(len(v) for v in images.values())
    audit["unique_image_names_per_split"] = len(images)
    audit["usable_annotated_images"] = len(records)
    audit["images_without_usable_annotations"] = len(images) - len(records)
    return sorted(records, key=lambda r: r["id"]), dict(audit), source_files


def partition(records, seed, config):
    groups = defaultdict(list)
    official_val = []
    val_groups = set()
    for record in records:
        if record["official_split"] == "train":
            groups[record["group"]].append(record)
        else:
            official_val.append(record)
            val_groups.add(record["group"])
    if set(groups) & val_groups:
        raise ValueError("Video/group leakage between official train and val")
    keys = sorted(groups)
    if len(keys) < 2:
        raise ValueError("Need at least two independent official train images/video groups")
    rng = random.Random(seed)
    rng.shuffle(keys)
    for selected in groups.values():
        selected.sort(key=lambda r: r["id"])
    boundary = max(1, min(len(keys) - 1, round(len(keys) * config["tune_fraction"])))
    result = {
        "tune": [record for key in keys[:boundary] for record in groups[key]],
        "train": [record for key in keys[boundary:] for record in groups[key]],
        "holdout": sorted(official_val, key=lambda r: r["id"]),
    }
    for split, selected in result.items():
        rng.shuffle(selected)
        result[split] = sorted(selected[: config[f"{split}_limit"]], key=lambda r: r["id"])
    assert_disjoint(result)
    return result


def assert_disjoint(partitions):
    owners = {}
    for split, records in partitions.items():
        if not records:
            raise ValueError(f"Empty {split} partition")
        ids = set()
        for record in records:
            if record["id"] in ids:
                raise ValueError(f"Duplicate ID in {split}: {record['id']}")
            ids.add(record["id"])
            for field in ("id", "group", "path", "sha256"):
                if field not in record:
                    continue
                key = field, record[field]
                if key in owners and owners[key] != split:
                    raise ValueError(f"Partition leakage on {field}: {record[field]}")
                owners[key] = split


def yolo_labels(boxes, width, height):
    lines = []
    for cls, x1, y1, x2, y2 in boxes:
        if (
            not all(math.isfinite(v) for v in (cls, x1, y1, x2, y2))
            or int(cls) != cls
            or not 0 <= cls < len(CLASSES)
            or not 0 <= x1 < x2 <= width
            or not 0 <= y1 < y2 <= height
        ):
            raise ValueError(
                f"Out-of-bounds/invalid box for {width}x{height}: {[cls, x1, y1, x2, y2]}"
            )
        lines.append(
            f"{int(cls)} {(x1 + x2) / (2 * width):.10f} {(y1 + y2) / (2 * height):.10f} "
            f"{(x2 - x1) / width:.10f} {(y2 - y1) / height:.10f}"
        )
    return "\n".join(lines) + ("\n" if lines else "")


def prepare(root, output, config, discovered=None):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError(f"Prepared directory already exists: {output}; use a new output directory")
    records, audit, sources = discovered if discovered is not None else discover(root)
    partitions = partition(records, config["seed"], config["data"])
    # Validate all selected files before creating any export, never clip or rescale boxes.
    for selected in partitions.values():
        for record in selected:
            image = read_image(record["path"])
            record["height"], record["width"] = image.shape[:2]
            yolo_labels(record["boxes"], record["width"], record["height"])
            record["sha256"] = file_hash(record["path"])
    assert_disjoint(partitions)
    manifest = {
        "schema": 1,
        "classes": CLASSES,
        "dataset_root": str(Path(root).resolve()),
        "seed": config["seed"],
        "data_config": config["data"],
        "sources": sources,
        "audit": audit,
        "partitions": partitions,
        "leakage_checks": ["id", "group", "path", "sha256"],
    }
    manifest["identity"] = digest(manifest)
    for split, selected in partitions.items():
        for record in selected:
            image_dir, label_dir = output / "images" / split, output / "labels" / split
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            source = Path(record["path"])
            (image_dir / (record["id"] + source.suffix)).symlink_to(source)
            (label_dir / f"{record['id']}.txt").write_text(
                yolo_labels(record["boxes"], record["width"], record["height"])
            )
    dataset = {
        "path": str(output),
        "train": "images/train",
        "val": "images/tune",
        "names": dict(enumerate(CLASSES)),
    }
    (output / "dataset.yaml").write_text(yaml.safe_dump(dataset))
    write_json(output / "manifest.json", manifest)
    return manifest


def load_manifest(prepared, verify_images=True):
    manifest = read_json(Path(prepared) / "manifest.json")
    identity = manifest.pop("identity")
    if digest(manifest) != identity or manifest["classes"] != CLASSES:
        raise ValueError("Manifest identity/classes changed; prepare a new dataset")
    manifest["identity"] = identity
    assert_disjoint(manifest["partitions"])
    if verify_images:
        for records in manifest["partitions"].values():
            for record in records:
                if file_hash(record["path"]) != record["sha256"]:
                    raise ValueError(f"Source image changed: {record['path']}")
    return manifest


def survey(root, output, config, discovered=None):
    records, audit, sources = discovered if discovered is not None else discover(root)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    counts, sizes, metadata = {}, [], {}
    for split in ("train", "val"):
        selected = [r for r in records if r["official_split"] == split]
        boxes = Counter(CLASSES[int(b[0])] for r in selected for b in r["boxes"])
        images_per_class = Counter(
            name for r in selected for name in {CLASSES[int(b[0])] for b in r["boxes"]}
        )
        counts[split] = {
            "images": len(selected),
            "boxes": sum(boxes.values()),
            "classes": {name: boxes[name] for name in CLASSES},
            "images_per_class": {name: images_per_class[name] for name in CLASSES},
        }
        metadata[split] = {
            key: dict(Counter(r["attributes"][key] for r in selected))
            for key in ("weather", "timeofday", "scene")
        }
        for r in selected:
            for cls, x1, y1, x2, y2 in r["boxes"]:
                sizes.append(
                    {
                        "split": split,
                        "class": CLASSES[cls],
                        "width": x2 - x1,
                        "height": y2 - y1,
                        "area": (x2 - x1) * (y2 - y1),
                        "aspect_ratio": (x2 - x1) / (y2 - y1),
                    }
                )
    sample = random.Random(config["seed"]).sample(
        records, min(len(records), config["data"]["quality_samples"])
    )
    health = []
    for r in sample:
        row = {"id": r["id"], "split": r["official_split"]}
        try:
            image = read_image(r["path"])
            row.update(quality(image))
            row["height"], row["width"] = image.shape[:2]
            yolo_labels(r["boxes"], row["width"], row["height"])
            row["status"] = "ok"
        except ValueError as exc:
            row.update(status="invalid", error=str(exc))
        health.append(row)
    summary = {
        "counts": counts,
        "metadata": metadata,
        "audit": audit,
        "sources": sources,
        "seed": config["seed"],
        "quality_sample_ids": [r["id"] for r in sample],
        "quality_health": health,
        "warnings": (
            ["Partial annotation coverage: fewer than 70k train/10k val frames"]
            if counts["train"]["images"] < 70000 or counts["val"]["images"] < 10000
            else []
        ),
        "geometry_note": "Box distributions in source pixels; image health and bounds "
        "checked on sampled images only. Prepare validates every selected image.",
    }
    write_json(output / "survey.json", summary)
    if sizes:
        with (output / "boxes.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(sizes[0]))
            writer.writeheader()
            writer.writerows(sizes)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for split in counts:
        axes[0].plot(CLASSES, [counts[split]["classes"][k] for k in CLASSES], "o-", label=split)
    axes[0].set(title="Vehicle boxes", ylabel="Count")
    axes[0].tick_params(axis="x", rotation=40)
    axes[0].legend()
    for axis, key in zip(axes[1:], ("area", "aspect_ratio"), strict=True):
        values = [r[key] for r in sizes]
        axis.hist(np.log10(values) if values else [], bins=40)
        axis.set(title=f"Box log10({key})", ylabel="Count")
    fig.tight_layout()
    fig.savefig(output / "survey.png")
    plt.close(fig)
    return summary
