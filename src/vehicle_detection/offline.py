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


def encoding_policy(output_format, jpeg_quality):
    if output_format not in {"jpg", "png"}:
        raise ValueError("output_format must be jpg or png")
    if (
        isinstance(jpeg_quality, bool)
        or not isinstance(jpeg_quality, int)
        or not 1 <= jpeg_quality <= 100
    ):
        raise ValueError("jpeg_quality must be an integer in [1, 100]")
    return {
        "format": output_format,
        "jpeg_quality": jpeg_quality if output_format == "jpg" else None,
        "opencv": cv2.__version__,
    }


def export_variant(
    partitions, converted, destination, variant, recipes, *, output_format="jpg", jpeg_quality=95
):
    validate_recipes(recipes)
    if variant == "raw":
        raise ValueError("Choose raw_original or raw_reencoded explicitly")
    encoding = encoding_policy(output_format, jpeg_quality)
    original = variant == "raw_original"
    recipe = "raw" if variant in {"raw_original", "raw_reencoded"} else variant
    if original:
        encoding = {**encoding, "format": "original", "jpeg_quality": None}
    destination, converted = Path(destination).resolve(), Path(converted).resolve()
    records = [(split, r) for split, rows in partitions.items() for r in rows]
    source_paths, parents = {}, {}
    for _, r in records:
        path = Path(r["path"])
        if path.parent not in parents:
            parents[path.parent] = path.parent.resolve()
        source = parents[path.parent] / path.name
        if source.is_symlink():
            source = source.resolve()
        source_paths[r["id"]] = source
        if destination == source or destination in source.parents or source in destination.parents:
            raise ValueError("Export directory overlaps source data")
        if file_hash(source) != r["sha256"]:
            raise ValueError(f"Source image changed: {source}")
        if original and source.suffix.lower() in {".jpg", ".jpeg"}:
            with source.open("rb") as stream:
                stream.seek(-2, 2)
                if stream.read() != b"\xff\xd9":
                    raise ValueError(
                        f"Invalid JPEG end marker; native YOLO would re-encode: {source}"
                    )
    labels = {r["id"]: file_hash(converted / "labels" / s / f"{r['id']}.txt") for s, r in records}
    identity = {
        "variant": variant,
        "recipe": recipes if recipe == "adaptive" else recipes[recipe],
        "encoding": encoding,
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
    for split in ("train", "tune", "test"):
        for kind in ("images", "labels"):
            (destination / kind / split).mkdir(parents=True, exist_ok=True)
    (destination / "verified.json").unlink(missing_ok=True)
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
            source = source_paths[r["id"]]
            suffix = source.suffix if original else f".{output_format}"
            target = destination / "images" / split / f"{r['id']}{suffix}"
            label = destination / "labels" / split / f"{r['id']}.txt"
            old = completed.get(r["id"])
            label_source = converted / "labels" / split / label.name
            # Never follow an unexpected image symlink when repairing a generated file.
            # Owned raw links store absolute validated sources; read the link without retraversal.
            if target.is_symlink() and (not original or target.readlink() != source):
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
            output, decisions, scores = apply_recipe(image, recipe, recipes)
            filter_ms = (time.perf_counter() - tick) * 1000
            assert output.shape == image.shape and output.dtype == image.dtype
            target.parent.mkdir(parents=True, exist_ok=True)
            label.parent.mkdir(parents=True, exist_ok=True)
            if original:
                if target.exists() and not target.is_symlink():
                    raise ValueError(f"Refusing to replace existing raw file: {target}")
                if not target.is_symlink():
                    target.symlink_to(source)
            else:
                temporary = target.with_suffix(f".tmp.{output_format}")
                temporary.unlink(missing_ok=True)
                parameters = (
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality] if output_format == "jpg" else []
                )
                if not cv2.imwrite(str(temporary), output, parameters):
                    raise OSError(f"Image write failed: {temporary}")
                if read_image(temporary).shape != image.shape:
                    raise ValueError(f"Invalid encoded image geometry: {temporary}")
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
                "width": image.shape[1],
                "height": image.shape[0],
                "encoding": encoding,
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


def verify_alignment(datasets, *, deep=True):
    """Deep export verification, or a cheap immutable-manifest/inventory check for training."""
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
            if {p.stem for p in (directory / "labels" / split).glob("*.txt")} != expected:
                raise ValueError(f"Export label ID mismatch: {directory}/{split}")
        export = read_json(directory / "export.json")
        stamp = {
            "export_sha256": file_hash(directory / "export.json"),
            "metadata_sha256": file_hash(directory / "metadata.csv"),
        }
        if not deep:
            if not (directory / "verified.json").is_file():
                raise ValueError(
                    "Missing verified export manifest; run notebook 01 deep verification"
                )
            verified = read_json(directory / "verified.json")
            if any(verified.get(k) != value for k, value in stamp.items()):
                raise ValueError(
                    "Verified export identity changed; run notebook 01 deep verification"
                )
        for row in meta.to_dict("records"):
            image_path = Path(row["image"])
            label = directory / "labels" / row["split"] / f"{row['id']}.txt"
            if not image_path.is_file() or not label.is_file():
                raise ValueError(f"Missing exported image/label: {row['id']}")
            # Regular training/development exports are owned files, not redirections.
            # Historical raw_original deliberately uses validated source-image links.
            if label.is_symlink() or (
                export.get("variant") != "raw_original" and image_path.is_symlink()
            ):
                raise ValueError(f"Unexpected exported symlink: {row['id']}")
            if deep and (
                file_hash(image_path) != row["output_sha256"]
                or file_hash(label) != row["label_sha256"]
            ):
                raise ValueError(f"Export image/label changed: {row['id']}")
            shape = (int(row["height"]), int(row["width"]), 3)
            if deep and read_image(image_path).shape != shape:
                raise ValueError(f"Export image geometry changed: {row['id']}")
            current[row["id"]] = (row["split"], row["label_sha256"], shape)
        files_identity = digest(
            [
                (
                    r["id"],
                    r["split"],
                    r["output_sha256"],
                    r["label_sha256"],
                    r["width"],
                    r["height"],
                )
                for r in meta.to_dict("records")
            ]
        )
        if deep:
            write_json(directory / "verified.json", {**stamp, "files_identity": files_identity})
        elif verified.get("files_identity") != files_identity:
            raise ValueError("Verified export manifest changed")
        if reference is not None and current != reference:
            raise ValueError("Variant labels, IDs, dimensions or split assignments differ")
        reference = current
    return len(reference or {})


def estimate_disk(sample, image_count, variants, recipes, *, output_format="jpg", jpeg_quality=95):
    """Estimate extra image bytes from an actual seeded sample encoded with the export policy."""
    import numpy as np

    validate_recipes(recipes)
    encoding_policy(output_format, jpeg_quality)
    if not sample or image_count < 1:
        raise ValueError("A nonempty representative sample and positive image count are required")
    sizes = {variant: [] for variant in variants}
    parameters = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality] if output_format == "jpg" else []
    for r in sample:
        image = read_image(r["path"])
        for variant in variants:
            if variant == "raw_original":
                sizes[variant].append(0)
                continue
            recipe = "raw" if variant == "raw_reencoded" else variant
            processed, _, _ = apply_recipe(image, recipe, recipes)
            ok, buffer = cv2.imencode(f".{output_format}", processed, parameters)
            if not ok:
                raise OSError("Sample image encoding failed")
            sizes[variant].append(buffer.nbytes)
    return pd.DataFrame(
        [
            {
                "variant": v,
                "sample_images": len(sample),
                "planned_images": image_count,
                "mean_bytes_per_image": float(np.mean(values)),
                "estimated_GiB": float(np.mean(values)) * image_count / 2**30,
                "p90_budget_GiB": float(np.percentile(values, 90)) * image_count / 2**30,
            }
            for v, values in sizes.items()
        ]
    )
