"""Stream local BDD labels, choose disjoint subsets, and call the official converters."""

import importlib.metadata
import random
from collections import Counter, defaultdict
from pathlib import Path

import ijson
import numpy as np
from PIL import Image

from .common import file_hash, read_image, read_json, write_json

CLASSES = ["car", "truck", "bus", "train", "motorcycle", "bicycle"]
ALIASES = {"motor": "motorcycle", "bike": "bicycle"}


def scan_bdd(root):
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
                    if len(missing_examples) < 10:
                        missing_examples.append(sample_id)
                    continue
                labels = []
                for label in frame["labels"]:
                    category = ALIASES.get(label.get("category"), label.get("category"))
                    if category not in CLASSES:
                        audit["ignored_nonvehicle_or_polygon_labels"] += 1
                        continue
                    if not isinstance(label.get("box2d"), dict):
                        raise ValueError(f"Vehicle without box2d: {sample_id}")
                    labels.append(label)
                records.append(
                    {
                        "id": sample_id,
                        "name": name,
                        "path": paths[split, name],
                        "official_split": split,
                        "group": frame.get("videoName") or sample_id,
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
        {**dict(audit), "missing_image_examples": missing_examples},
        sources,
    )


def choose_splits(records, seed, limits, excluded_test_ids=(), tune_fraction=0.2):
    groups = defaultdict(list)
    test = []
    for record in records:
        if record["official_split"] == "train":
            groups[record["group"]].append(record)
        elif record["id"] not in excluded_test_ids:
            test.append(record)
    if set(groups) & {r["group"] for r in test}:
        raise ValueError("Official train/val video-group leakage")
    keys = sorted(groups)
    if len(keys) < 2 or not 0 < tune_fraction < 1:
        raise ValueError("Need two training groups and 0 < tune_fraction < 1")
    rng = random.Random(seed)
    rng.shuffle(keys)
    boundary = max(1, min(len(keys) - 1, round(len(keys) * tune_fraction)))
    partitions = {
        "train": [r for k in keys[boundary:] for r in sorted(groups[k], key=lambda r: r["id"])],
        "tune": [r for k in keys[:boundary] for r in sorted(groups[k], key=lambda r: r["id"])],
        "test": sorted(test, key=lambda r: r["id"]),
    }
    owners = {}
    for split, selected in partitions.items():
        if limits[split] is not None and (
            isinstance(limits[split], bool)
            or not isinstance(limits[split], int)
            or limits[split] < 1
        ):
            raise ValueError("Subset limits must be positive integers or None")
        rng.shuffle(selected)
        partitions[split] = selected = sorted(selected[: limits[split]], key=lambda r: r["id"])
        if not selected:
            raise ValueError(f"Empty {split} subset")
        for r in selected:
            image = read_image(r["path"])
            r["height"], r["width"] = image.shape[:2]
            r["sha256"] = file_hash(r["path"])
            for key in ("id", "group", "sha256"):
                token = (key, r[key])
                if token in owners and owners[token] != split:
                    raise ValueError(f"Split leakage on {key}: {r['id']}")
                owners[token] = split
    return partitions


def convert_labels(partitions, output, convention="continuous"):
    """Official BDD→Scalabel→COCO and Ultralytics COCO→YOLO, with geometry assertions."""
    from bdd100k.common.utils import load_bdd100k_config
    from bdd100k.label.to_coco import scalabel2coco_detection
    from bdd100k.label.to_scalabel import bdd100k_to_scalabel
    from scalabel.label.io import load
    from ultralytics.data.converter import convert_coco

    if convention not in {"continuous", "inclusive"}:
        raise ValueError("Choose explicit continuous (2018 study) or inclusive endpoint convention")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    native = output / "coco"
    native.mkdir(exist_ok=True)
    cfg = load_bdd100k_config("det")
    cfg.scalabel.imageSize = None  # Read actual dimensions rather than assuming 1280x720.
    mapping = {name: i + 1 for i, name in enumerate(CLASSES)}
    for split, records in partitions.items():
        raw = [
            {
                "name": r["name"],
                "size": {"width": r["width"], "height": r["height"]},
                "attributes": r["attributes"],
                "labels": r["labels"],
            }
            for r in records
        ]
        source_path = output / f"{split}_bdd.json"
        write_json(source_path, raw)
        frames = bdd100k_to_scalabel(load(str(source_path), nprocs=0).frames, cfg)
        coco = scalabel2coco_detection(frames, cfg.scalabel)
        categories = {c["id"]: c["name"] for c in coco["categories"]}
        assert set(CLASSES) <= set(categories.values())
        for ann in coco["annotations"]:
            ann["category_id"] = mapping[categories[ann["category_id"]]]
            if convention == "continuous":
                # Scalabel 0.3.1 adds 1 to extents; bridge to the study's x2-x1 convention.
                ann["bbox"][2] -= 1
                ann["bbox"][3] -= 1
                ann["area"] = ann["bbox"][2] * ann["bbox"][3]
        coco["categories"] = [{"id": i + 1, "name": name} for i, name in enumerate(CLASSES)]
        write_json(native / f"{split}.json", coco)
    # convert_coco increments existing output paths, so always use a fresh temporary directory.
    import tempfile

    with tempfile.TemporaryDirectory(dir=output) as temporary:
        destination = Path(temporary) / "yolo"
        convert_coco(str(native), str(destination), cls91to80=False)
        for split, records in partitions.items():
            for r in records:
                converted = destination / "labels" / split / f"{r['id']}.txt"
                text = converted.read_text() if converted.exists() else ""
                target = output / "labels" / split / converted.name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text)
                actual = (
                    np.loadtxt(target, ndmin=2).reshape(-1, 5) if text.strip() else np.empty((0, 5))
                )
                expected = []
                for label in r["labels"]:
                    attrs = label.get("attributes") or {}
                    if attrs.get("crowd") or attrs.get("ignored"):
                        continue
                    b = label["box2d"]
                    x1, y1, x2, y2 = [float(b[k]) for k in ("x1", "y1", "x2", "y2")]
                    x2 += convention == "inclusive"
                    y2 += convention == "inclusive"
                    if not (
                        np.isfinite([x1, y1, x2, y2]).all()
                        and 0 <= x1 < x2 <= r["width"]
                        and 0 <= y1 < y2 <= r["height"]
                    ):
                        raise ValueError(f"Invalid source geometry: {r['id']}")
                    cls = CLASSES.index(ALIASES.get(label["category"], label["category"]))
                    expected.append([cls, x1, y1, x2, y2])
                if len(actual):
                    assert np.all(actual[:, 0] == actual[:, 0].astype(int))
                    decoded = actual.copy()
                    decoded[:, 1] = (actual[:, 1] - actual[:, 3] / 2) * r["width"]
                    decoded[:, 2] = (actual[:, 2] - actual[:, 4] / 2) * r["height"]
                    decoded[:, 3] = (actual[:, 1] + actual[:, 3] / 2) * r["width"]
                    decoded[:, 4] = (actual[:, 2] + actual[:, 4] / 2) * r["height"]
                else:
                    decoded = np.empty((0, 5))
                # Native converter preserves first-seen order when deduplicating boxes.
                expected = np.array(list(dict.fromkeys(map(tuple, expected)))).reshape(-1, 5)
                np.testing.assert_allclose(
                    decoded, expected, atol=0.01, rtol=0, err_msg=f"Conversion mismatch: {r['id']}"
                )
    versions = {p: importlib.metadata.version(p) for p in ("bdd100k", "scalabel", "ultralytics")}
    write_json(
        output / "conversion.json",
        {
            "versions": versions,
            "classes": CLASSES,
            "box_convention": convention,
            "geometry_tolerance_pixels": 0.01,
            "coco_category_ids": list(range(1, 7)),
            "yolo_category_ids": list(range(6)),
        },
    )
    return read_json(output / "conversion.json")


def audit_dataset(records, convention="continuous"):
    """Audit every supplied annotation against image headers without decoding all image pixels."""
    if convention not in {"continuous", "inclusive"}:
        raise ValueError("Unknown coordinate convention")
    counts, errors, examples, owners, ids = Counter(), Counter(), {}, {}, set()

    def error(reason, sample_id):
        errors[reason] += 1
        bucket = examples.setdefault(reason, [])
        if sample_id not in bucket and len(bucket) < 10:
            bucket.append(sample_id)

    for r in records:
        counts["images"] += 1
        counts["boxes"] += len(r["labels"])
        if r["id"] in ids:
            error("duplicate_id", r["id"])
        ids.add(r["id"])
        previous = owners.setdefault(r["group"], r["official_split"])
        if previous != r["official_split"]:
            error("cross_split_group", r["id"])
        try:
            with Image.open(r["path"]) as image:
                width, height = image.size
                if min(width, height) < 2:
                    raise ValueError("Invalid image size")
            counts["readable_headers"] += 1
        except (OSError, ValueError):
            error("missing_or_invalid_image_header", r["id"])
            continue
        for label in r["labels"]:
            try:
                x1, y1, x2, y2 = [float(label["box2d"][k]) for k in ("x1", "y1", "x2", "y2")]
            except (KeyError, TypeError, ValueError, OverflowError):
                error("malformed_box", r["id"])
                continue
            x2 += convention == "inclusive"
            y2 += convention == "inclusive"
            if not np.isfinite([x1, y1, x2, y2]).all():
                error("nonfinite_box", r["id"])
            elif x1 >= x2 or y1 >= y2:
                error("reversed_or_empty_box", r["id"])
            elif not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                error("out_of_bounds_box", r["id"])
            else:
                counts["valid_boxes"] += 1
    return {
        "counts": dict(counts),
        "errors": dict(errors),
        "examples": examples,
        "error_count": sum(errors.values()),
        "box_convention": convention,
        "scope": "all supplied boxes and image headers; no full-pixel decode or content hashes",
    }
