"""Safe offline image exports; no detector, metric engine or experiment runner."""

import json
import os
import time
from pathlib import Path

import cv2
import pandas as pd
import yaml

from .common import digest, file_hash, read_image, read_json, write_json
from .data import CLASSES
from .preprocess import apply_recipe, validate_recipes


def export_variant(partitions, converted, destination, variant, recipes):
    validate_recipes(recipes)
    destination, converted = Path(destination).resolve(), Path(converted).resolve()
    records = [(split, r) for split, rows in partitions.items() for r in rows]
    for _, r in records:
        source = Path(r["path"]).resolve()
        if destination == source or destination in source.parents or source in destination.parents:
            raise ValueError("Export directory overlaps source data")
        if file_hash(source) != r["sha256"]:
            raise ValueError(f"Source image changed: {source}")
    labels = {r["id"]: file_hash(converted / "labels" / s / f"{r['id']}.txt") for s, r in records}
    identity = {
        "variant": variant,
        "recipe": recipes if variant == "adaptive" else recipes[variant],
        "source": digest([(s, r["id"], r["sha256"], labels[r["id"]]) for s, r in records]),
        "operator_code": file_hash(Path(__file__).with_name("preprocess.py")),
        "conversion": read_json(converted / "conversion.json"),
    }
    destination.mkdir(parents=True, exist_ok=True)
    identity_path = destination / "export.json"
    if identity_path.exists():
        if read_json(identity_path) != identity:
            raise ValueError(f"Export inputs/recipe changed: {destination}; use a fresh directory")
    elif any(destination.iterdir()):
        raise ValueError(f"Unknown existing export directory: {destination}")
    else:
        write_json(identity_path, identity)
    journal = destination / "metadata.jsonl"
    completed = {}
    if journal.exists():
        data = journal.read_bytes()
        # A hard kill may leave only the final append incomplete; keep all complete rows.
        complete = data[: data.rfind(b"\n") + 1]
        if complete != data:
            journal.write_bytes(complete)
        for line in complete.splitlines():
            row = json.loads(line)
            completed[row["id"]] = row
    result = []
    with journal.open("a") as log:
        for split, r in records:
            source = Path(r["path"]).resolve()
            suffix = source.suffix if variant == "raw" else ".png"
            target = destination / "images" / split / f"{r['id']}{suffix}"
            label = destination / "labels" / split / f"{r['id']}.txt"
            old = completed.get(r["id"])
            label_source = converted / "labels" / split / label.name
            # Never follow an unexpected image symlink when repairing a generated file.
            if target.is_symlink() and (variant != "raw" or target.resolve() != source):
                raise ValueError(f"Unsafe export symlink: {target}")
            if label.is_symlink():
                raise ValueError(f"Unexpected label symlink: {label}")
            reusable = (
                old is not None
                and target.is_file()
                and label.is_file()
                and file_hash(target) == old["output_sha256"]
                and file_hash(label) == labels[r["id"]]
            )
            if reusable:
                result.append({**old, "reused": True})
                continue
            started = time.perf_counter()
            image = read_image(source)
            tick = time.perf_counter()
            output, decisions, scores = apply_recipe(image, variant, recipes)
            filter_ms = (time.perf_counter() - tick) * 1000
            assert output.shape == image.shape and output.dtype == image.dtype
            target.parent.mkdir(parents=True, exist_ok=True)
            label.parent.mkdir(parents=True, exist_ok=True)
            if variant == "raw":
                if target.exists() and not target.is_symlink():
                    raise ValueError(f"Refusing to replace existing raw file: {target}")
                if not target.is_symlink():
                    target.symlink_to(source)
            else:
                temporary = target.with_suffix(".tmp.png")
                temporary.unlink(missing_ok=True)
                if not cv2.imwrite(str(temporary), output):
                    raise OSError(f"Image write failed: {temporary}")
                os.replace(temporary, target)
            temporary = label.with_suffix(".tmp")
            temporary.unlink(missing_ok=True)
            temporary.write_bytes(label_source.read_bytes())
            os.replace(temporary, label)
            row = {
                "id": r["id"],
                "split": split,
                "variant": variant,
                "source": str(source),
                "image": str(target),
                "source_sha256": r["sha256"],
                "output_sha256": file_hash(target),
                "label_sha256": labels[r["id"]],
                "decisions": decisions,
                "quality": scores,
                "preprocess_ms": filter_ms,
                "preparation_ms": (time.perf_counter() - started) * 1000,
            }
            log.write(json.dumps(row, allow_nan=False) + "\n")
            log.flush()
            result.append({**row, "reused": False})
    yaml_path = destination / "data.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(destination),
                "train": "images/train",
                "val": "images/tune",
                "test": "images/test",
                "names": dict(enumerate(CLASSES)),
            }
        )
    )
    table = pd.DataFrame(result)
    table.assign(
        decisions=table.decisions.map(json.dumps), quality=table.quality.map(json.dumps)
    ).to_csv(destination / "metadata.csv", index=False)
    return table


def verify_alignment(datasets):
    """Fail if exported variants differ in sample IDs, labels, dimensions or split assignment."""
    reference = None
    for directory in map(Path, datasets):
        meta = pd.read_csv(directory / "metadata.csv")
        if meta["id"].duplicated().any():
            raise ValueError("Duplicate exported IDs")
        current = {}
        yaml_data = yaml.safe_load((directory / "data.yaml").read_text())
        if (
            yaml_data["names"] != dict(enumerate(CLASSES))
            or yaml_data["val"] != "images/tune"
            or yaml_data["test"] != "images/test"
            or yaml_data["train"] != "images/train"
            or Path(yaml_data["path"]).resolve() != directory.resolve()
        ):
            raise ValueError("Wrong class mapping or train/tune/test YAML")
        for split in ("train", "tune", "test"):
            expected = set(meta.loc[meta.split == split, "id"])
            actual = {
                p.stem
                for p in (directory / "images" / split).iterdir()
                if p.suffix.lower() in {".jpg", ".png", ".jpeg"}
            }
            if actual != expected:
                raise ValueError(f"Export image ID mismatch: {directory}/{split}")
        for row in meta.to_dict("records"):
            image_path = Path(row["image"])
            label = directory / "labels" / row["split"] / f"{row['id']}.txt"
            if (
                file_hash(image_path) != row["output_sha256"]
                or file_hash(label) != row["label_sha256"]
            ):
                raise ValueError(f"Export image/label changed: {row['id']}")
            current[row["id"]] = (row["split"], file_hash(label), read_image(image_path).shape)
        if reference is not None and current != reference:
            raise ValueError("Variant labels, IDs, dimensions or split assignments differ")
        reference = current
    return len(reference or {})
