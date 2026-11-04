"""Explicit five-class derived-label mapping; historical six-class data stays immutable."""

from collections import Counter
from pathlib import Path

from .common import digest, file_hash, freeze_json, read_json

FIVE_CLASSES = ["car", "truck", "bus", "motorcycle", "bicycle"]
SIX_TO_FIVE = {0: 0, 1: 1, 2: 2, 3: None, 4: 3, 5: 4}


def remap_boxes(boxes):
    """Drop railway-train boxes; preserve every surviving coordinate and its order."""
    return [[SIX_TO_FIVE[cls], *coords] for cls, *coords in boxes if SIX_TO_FIVE[cls] is not None]


def derive_labels(source, destination, partitions):
    """Derive labels from verified six-class YOLO files, never from final-test pixels.

    Only train and tune are allowed. The source conversion, every source label,
    and the original six-class outputs remain untouched.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Five-class conversion must use an independent directory")
    if not set(partitions) <= {"train", "tune"} or not partitions:
        raise ValueError("Only training/development labels may be materialized")
    conversion = read_json(source / "conversion.json")
    if "classes" in conversion and conversion["classes"] != [
        "car", "truck", "bus", "train", "motorcycle", "bicycle"
    ]:
        raise ValueError("Expected historical six-class source conversion")
    if "box_convention" in conversion and conversion["box_convention"] != "continuous":
        raise ValueError("Expected continuous BDD box convention")

    source_files, converted, summary = [], {}, {}
    for split, rows in partitions.items():
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError("Duplicate source IDs")
        counter = Counter(images=len(rows))
        for row in rows:
            path = source / "labels" / split / f"{row['id']}.txt"
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Missing/unsafe historical label: {path}")
            lines = path.read_text().splitlines(keepends=True)
            expected = [int(b[0]) for b in row["boxes"]]
            observed = []
            kept = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    raise ValueError(f"Invalid six-class YOLO label: {path}")
                cls = int(parts[0])
                if str(cls) != parts[0] or cls not in SIX_TO_FIVE:
                    raise ValueError(f"Unknown source label class: {path}")
                # Validate numeric coordinates without re-encoding surviving values.
                coords = [float(v) for v in parts[1:]]
                if not all(0 <= v <= 1 for v in coords):
                    raise ValueError(f"Invalid normalized coordinates: {path}")
                observed.append(cls)
                mapped = SIX_TO_FIVE[cls]
                if mapped is not None:
                    separator = line[len(parts[0]):]
                    kept.append(str(mapped) + separator)
                    counter[f"instances:{FIVE_CLASSES[mapped]}"] += 1
                else:
                    counter["removed_railway_instances"] += 1
            if observed != expected:
                raise ValueError(f"Six-class source labels differ from frozen boxes: {path}")
            if not kept:
                counter["negative_images"] += 1
            if observed and not kept:
                counter["new_negative_images"] += 1
            result = "".join(kept)
            converted[split, row["id"]] = result
            source_files.append((split, row["id"], file_hash(path)))
        summary[split] = dict(counter)

    identity = {
        "version": "five-class-labels-v1",
        "classes": FIVE_CLASSES,
        "source_classes": ["car", "truck", "bus", "train", "motorcycle", "bicycle"],
        "mapping": {str(k): v for k, v in SIX_TO_FIVE.items()},
        "source_conversion_sha256": file_hash(source / "conversion.json"),
        "source_labels_sha256": digest(sorted(source_files)),
        "role_ids_sha256": digest(
            {s: sorted(r["id"] for r in rows) for s, rows in partitions.items()}
        ),
        "box_convention": "continuous",
        "summary": summary,
        "final_test_pixels_accessed": False,
    }
    destination.mkdir(parents=True, exist_ok=True)
    if not (destination / "conversion.json").exists() and any(destination.iterdir()):
        raise ValueError("Unknown five-class conversion directory")
    freeze_json(destination / "conversion.json", identity)
    for (split, sample_id), value in converted.items():
        target = destination / "labels" / split / f"{sample_id}.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.is_symlink() or target.read_text() != value:
                raise ValueError(f"Derived label changed: {target}")
        else:
            target.write_text(value)
    return identity
