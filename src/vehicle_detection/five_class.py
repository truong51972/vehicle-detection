"""Five-class YOLO labels derived from immutable historical six-class labels."""

from collections import Counter
from pathlib import Path

from .common import digest, file_hash, freeze_json, read_json
from .data import CLASSES as SIX_CLASSES

FIVE_CLASSES = ["car", "truck", "bus", "motorcycle", "bicycle"]
SIX_TO_FIVE = {0: 0, 1: 1, 2: 2, 3: None, 4: 3, 5: 4}


def remap_boxes(boxes):
    return [[SIX_TO_FIVE[c], *xyxy] for c, *xyxy in boxes if SIX_TO_FIVE[c] is not None]


def support_audit(rows):
    """Annotation-only counts, including reserved-final annotations; never decode pixels."""
    counts = Counter(images=len(rows))
    for row in rows:
        old = [int(box[0]) for box in row["boxes"]]
        mapped = remap_boxes(row["boxes"])
        for cls in old:
            counts[f"old_instances:{cls}"] += 1
        for cls in set(old):
            counts[f"old_presence:{cls}"] += 1
        for cls, x1, y1, x2, y2 in mapped:
            counts[f"new_instances:{FIVE_CLASSES[cls]}"] += 1
            area = (x2 - x1) * (y2 - y1)
            size = "small" if area < 32**2 else "medium" if area < 96**2 else "large"
            counts[f"new_size:{size}"] += 1
        for cls in {box[0] for box in mapped}:
            counts[f"new_presence:{FIVE_CLASSES[cls]}"] += 1
        counts["old_negative_images"] += int(not old)
        counts["new_negative_images"] += int(not mapped)
        counts["newly_negative_images"] += int(bool(old) and not mapped)
        for axis in ("weather", "timeofday", "scene"):
            counts[f"{axis}:{row['attributes'][axis]}"] += 1
    return dict(counts)


def derive_labels(source, destination, partitions):
    """Change class IDs only; the existing exporter validates YOLO geometry and image roles."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Five-class conversion must use an independent directory")
    if not partitions or set(partitions) - {"train", "tune"}:
        raise ValueError("Only training/development labels may be materialized")
    conversion = read_json(source / "conversion.json")
    if conversion.get("classes") != SIX_CLASSES or conversion.get("box_convention") != "continuous":
        raise ValueError("Expected verified six-class continuous-label conversion")

    derived = {}
    for split, rows in partitions.items():
        if len({row["id"] for row in rows}) != len(rows):
            raise ValueError("Duplicate source IDs")
        for row in rows:
            if row.get("official_split") == "val":
                raise ValueError("Final-test labels cannot be materialized")
            label = source / "labels" / split / f"{row['id']}.txt"
            if label.is_symlink() or not label.is_file():
                raise ValueError(f"Missing/unsafe historical label: {label}")
            lines = label.read_text().splitlines(keepends=True)
            classes = [line.split()[0] for line in lines]
            if classes != [str(box[0]) for box in row["boxes"]]:
                raise ValueError(f"Historical label class order changed: {label}")
            output = []
            for line, cls in zip(lines, classes, strict=True):
                target = SIX_TO_FIVE[int(cls)]
                if target is not None:
                    output.append(str(target) + line[len(cls) :])
            derived[split, row["id"]] = "".join(output)

    identity = {
        "version": "five-class-labels-v1",
        "classes": FIVE_CLASSES,
        "mapping": {str(k): v for k, v in SIX_TO_FIVE.items()},
        "source_conversion_sha256": file_hash(source / "conversion.json"),
        "role_ids_sha256": digest(
            {s: sorted(row["id"] for row in rows) for s, rows in partitions.items()}
        ),
    }
    destination.mkdir(parents=True, exist_ok=True)
    if not (destination / "conversion.json").exists() and any(destination.iterdir()):
        raise ValueError("Unknown five-class conversion directory")
    freeze_json(destination / "conversion.json", identity)
    for (split, sample_id), content in derived.items():
        target = destination / "labels" / split / f"{sample_id}.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.is_symlink() or target.read_text() != content:
                raise ValueError(f"Derived label changed: {target}")
        else:
            target.write_text(content)
    return identity
