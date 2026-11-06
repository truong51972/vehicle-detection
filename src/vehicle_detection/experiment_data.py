"""Shared frozen five-class data and deterministic, offline experiment exports."""

import re
import shutil
from pathlib import Path

import cv2
import pandas as pd
import yaml

from .common import digest, file_hash, read_image
from .data import scan_bdd
from .datasets import verify_label
from .five_class import FIVE_CLASSES, remap_boxes
from .preprocess import apply_recipe, validate_recipes
from .protocol import annotation_profile, group_key


def shared_splits(root):
    """Read fixed memberships/GT; never discover or resplit an experiment's population."""
    root = Path(root).resolve()
    shared = yaml.safe_load((root / "configs/data.yaml").read_text())
    path = root / shared["splits"]
    if (
        file_hash(path) != shared["splits_sha256"]
        or shared["classes"] != FIVE_CLASSES
        or shared["seed"] != 2026
        or shared["training_subset"] not in {"S10", "S20"}
    ):
        raise ValueError("Shared split table or taxonomy changed")
    rows = pd.read_csv(path, keep_default_na=False).to_dict("records")
    owners, selections = {}, {}
    roles = {"training_pool", "calibration", "development_validation", "final_test"}
    if {r["role"] for r in rows} != roles:
        raise ValueError("Expected all four frozen roles")
    for row in rows:
        source = Path(row["source_path"])
        if (
            not re.fullmatch(r"[\w-]+", row["id"])
            or source.is_absolute()
            or ".." in source.parts
            or row["s10"] not in (0, 1)
            or row["s20"] not in (0, 1)
            or row["s10"] > row["s20"]
            or (row["s20"] and row["role"] != "training_pool")
            or (row["role"] == "final_test" and row["sha256"])
            or (row["role"] != "final_test" and not row["sha256"])
            or row["width"] < 2
            or row["height"] < 2
        ):
            raise ValueError("Invalid split row or final-test membership")
        for key in ("id", "group", "source_group", "sha256"):
            token = (key, row[key])
            if not row[key]:
                continue
            if token in owners and (key == "id" or owners[token] != row["role"]):
                raise ValueError(f"Duplicate ID or cross-role leakage: {row['id']}")
            owners[token] = row["role"]
            if key != "id":
                flags = (row["s10"], row["s20"])
                if token in selections and selections[token] != flags:
                    raise ValueError(f"Subset divides a source/duplicate group: {row['id']}")
                selections[token] = flags
    if not 0 < sum(r["s10"] for r in rows) < sum(r["s20"] for r in rows):
        raise ValueError("Expected nonempty, strictly nested frozen S10/S20 populations")
    return shared, rows


def load_experiment(root, experiment):
    if not re.fullmatch(r"v\d+", experiment):
        raise ValueError("Expected a stable experiment ID such as v0")
    config = yaml.safe_load((Path(root) / f"configs/{experiment}.yaml").read_text())
    if config["id"] != experiment or config["variant"] not in {"raw", "adaptive", "local_shadow"}:
        raise ValueError("Wrong experiment identity or policy")
    validate_recipes(config["recipes"])
    if config["variant"] not in config["recipes"]:
        raise ValueError("Missing preprocessing recipe")
    if config["variant"] == "raw" and config["recipes"] != {"raw": {}}:
        raise ValueError("Raw must use Skip only")
    if experiment == "v0" and config["variant"] != "raw":
        raise ValueError("v0 is the Raw baseline")
    if config["seed"] != 2026 or config["jpeg_quality"] != 95:
        raise ValueError("Keep the shared seed and JPEG95 encoding")
    return config


def source_rows(root):
    """Read annotation metadata only; retain the frozen six-to-five geometry conversion."""
    root = Path(root).resolve()
    shared, rows = shared_splits(root)
    data = (root / shared["data_root"]).resolve()
    records, _, _ = scan_bdd(data, strict=False)
    lookup = {r["id"]: r for r in records}
    for row in rows:
        record = lookup[row["id"]]
        if (
            Path(record["path"]).relative_to(data).as_posix() != row["source_path"]
            or record["official_split"] != ("val" if row["role"] == "final_test" else "train")
            or group_key(record) != row["source_group"]
            or digest(record["labels"]) != row["labels_sha256"]
        ):
            raise ValueError(f"Source metadata/GT changed: {row['id']}")
        boxes, _, errors = annotation_profile(record, row["width"], row["height"])
        if errors:
            raise ValueError(f"Invalid source boxes: {row['id']}")
        row["boxes"] = remap_boxes(boxes)
    return shared, rows


def audit_sources(root):
    """Audit sources/annotations against frozen GT; final images remain unopened/unhashed."""
    root = Path(root).resolve()
    shared, rows = source_rows(root)
    data = (root / shared["data_root"]).resolve()
    for row in rows:
        if row["role"] != "final_test":
            source = data / row["source_path"]
            if source.resolve() != source:
                raise ValueError("Unexpected source symlink")
            if file_hash(source) != row["sha256"] or read_image(source).shape != (
                row["height"],
                row["width"],
                3,
            ):
                raise ValueError(f"Source image/geometry changed: {row['id']}")
    return pd.Series([r["role"] for r in rows]).value_counts()


def populations(rows, subset):
    flag = {"S10": "s10", "S20": "s20"}[subset]
    return {
        "train": [r for r in rows if r[flag]],
        "val": [r for r in rows if r["role"] == "development_validation"],
    }


def _data_config(root, experiment, shared):
    config = load_experiment(root, experiment)
    return {
        "path": str(Path(root).resolve() / f"runs/{experiment}/data"),
        "train": "images/train",
        "val": "images/val",
        "names": dict(enumerate(FIVE_CLASSES)),
        "experiment": experiment,
        "training_subset": shared["training_subset"],
        "seed": shared["seed"],
        "config_sha256": digest(config),
        "splits_sha256": shared["splits_sha256"],
        "opencv": cv2.__version__,
    }


def verify_data(root, experiment):
    """Check one native YAML, inventories, frozen labels and exported bytes, without sources."""
    root = Path(root).resolve()
    shared, rows = shared_splits(root)
    expected = _data_config(root, experiment, shared)
    directory = Path(expected["path"])
    if directory.resolve() != directory or (directory / "data.yaml").is_symlink():
        raise ValueError("Unexpected prepared directory symlink")
    saved = yaml.safe_load((directory / "data.yaml").read_text())
    # The operator hash records export provenance; reuse checks the frozen output bytes.
    if {k: v for k, v in saved.items() if k not in {"files_sha256", "operator_sha256"}} != expected:
        raise ValueError("Prepared experiment inputs/policy changed; use a new experiment ID")
    files = []
    for split, selected in populations(rows, shared["training_subset"]).items():
        for kind, suffix in (("images", ".jpg"), ("labels", ".txt")):
            folder = directory / kind / split
            if folder.resolve() != folder:
                raise ValueError("Unexpected prepared directory symlink")
            paths = list(folder.iterdir())
            if {p.name for p in paths} != {r["id"] + suffix for r in selected}:
                raise ValueError("Prepared IDs differ from shared split/GT")
            if any(p.is_symlink() or not p.is_file() for p in paths):
                raise ValueError("Unexpected prepared symlink/directory")
        for row in selected:
            label = directory / "labels" / split / f"{row['id']}.txt"
            image = directory / "images" / split / f"{row['id']}.jpg"
            files.append((split, row["id"], file_hash(image), file_hash(label)))
    if digest(files) != saved["files_sha256"]:
        raise ValueError("Prepared image/label bytes changed")
    return saved


def prepare_data(root, experiment):
    """Export originals once; an interrupted temporary export is never reused silently."""
    root = Path(root).resolve()
    config = load_experiment(root, experiment)
    shared, rows = shared_splits(root)
    native = _data_config(root, experiment, shared)
    destination = Path(native["path"])
    if destination.resolve() != destination:
        raise ValueError("Unexpected export directory symlink")
    if destination.exists():
        return verify_data(root, experiment)
    native["operator_sha256"] = file_hash(Path(__file__).with_name("preprocess.py"))
    shared, rows = source_rows(root)
    data = (root / shared["data_root"]).resolve()
    if destination.is_relative_to(data) or data.is_relative_to(destination):
        raise ValueError("Export overlaps original sources")
    temporary = destination.with_name("data.tmp")
    temporary.mkdir(parents=True, exist_ok=False)
    files = []
    try:
        for split, selected in populations(rows, shared["training_subset"]).items():
            for kind in ("images", "labels"):
                (temporary / kind / split).mkdir(parents=True)
            for row in selected:
                source = data / row["source_path"]
                if source.resolve() != source or not source.is_relative_to(data):
                    raise ValueError("Unexpected source path/symlink")
                if file_hash(source) != row["sha256"]:
                    raise ValueError(f"Source image changed: {row['id']}")
                image = read_image(source)
                if image.shape != (row["height"], row["width"], 3):
                    raise ValueError("Source geometry changed")
                output, _, _ = apply_recipe(image, config["variant"], config["recipes"])
                if output.shape != image.shape or output.dtype != image.dtype:
                    raise ValueError("Preprocessing changed geometry/dtype")
                target = temporary / "images" / split / f"{row['id']}.jpg"
                if not cv2.imwrite(str(target), output, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise OSError(f"Cannot encode {target}")
                if read_image(target).shape != image.shape:
                    raise ValueError("Encoded geometry changed")
                label = temporary / "labels" / split / f"{row['id']}.txt"
                # Match native COCO conversion bytes to preserve historical five-class GT.
                label.write_text(
                    "".join(
                        f"{c} {(x1 + (x2 - x1) / 2) / row['width']:g} "
                        f"{(y1 + (y2 - y1) / 2) / row['height']:g} "
                        f"{(x2 - x1) / row['width']:g} {(y2 - y1) / row['height']:g}\n"
                        for c, x1, y1, x2, y2 in row["boxes"]
                    )
                )
                verify_label(label, row)
                files.append((split, row["id"], file_hash(target), file_hash(label)))
        native["files_sha256"] = digest(files)
        (temporary / "data.yaml").write_text(yaml.safe_dump(native))
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary)
        raise
    return native
