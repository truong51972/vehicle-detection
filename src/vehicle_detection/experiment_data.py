"""Shared frozen five-class data and deterministic, offline per-run experiment exports."""

import datetime
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from .common import digest, file_hash, read_image, relative
from .data import FIVE_CLASSES, annotation_profile, group_key, remap_boxes, scan_bdd
from .preprocess import apply_recipe, validate_recipes

EXPERIMENTS_DIR = Path("notebooks/experiments")
VARIANTS = {"raw", "adaptive", "local_shadow", "adaptive_local_contrast", "adaptive_canny_detail"}
# Ultralytics split → frozen role. Calibration selects checkpoints; development is reported.
ROLES = {"val": "calibration", "test": "development_validation"}
FEATURES = ("brightness", "contrast", "sharpness", "noise", "black_fraction", "white_fraction")
REQUIRED_TRAINING = {"epochs", "imgsz", "batch", "optimizer", "lr0", "freeze", "amp", "device"}
REQUIRED_EVALUATION = {"imgsz", "batch", "conf", "iou", "max_det", "device"}


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


def config_path(root, experiment):
    """Each version owns `notebooks/experiments/<version>/<version>.yaml` beside its notebooks."""
    if not re.fullmatch(r"v\d+", experiment):
        raise ValueError("Expected a stable experiment ID such as v0")
    return Path(root) / EXPERIMENTS_DIR / experiment / f"{experiment}.yaml"


def experiments(root):
    """Version IDs with a config, in numeric order."""
    found = [p.parent.name for p in (Path(root) / EXPERIMENTS_DIR).glob("v*/v*.yaml")]
    return sorted((v for v in found if config_path(root, v).is_file()), key=lambda v: int(v[1:]))


def load_experiment(root, experiment):
    """One version YAML owns its policy plus every native preparation/training/eval argument."""
    config = yaml.safe_load(config_path(root, experiment).read_text())
    if config["id"] != experiment or config["variant"] not in VARIANTS:
        raise ValueError("Wrong experiment identity or policy")
    if not isinstance(config.get("name"), str) or not config["name"]:
        raise ValueError("Give the version a display name")
    workers = (config.get("preparation") or {}).get("workers")
    if type(workers) is not int or workers < 1:
        raise ValueError("preparation.workers must be a positive integer")
    validate_recipes(config["recipes"])
    if config["variant"] not in config["recipes"]:
        raise ValueError("Missing preprocessing recipe")
    if config["variant"] == "raw" and config["recipes"] != {"raw": {}}:
        raise ValueError("Raw must use Skip only")
    if experiment == "v0" and config["variant"] != "raw":
        raise ValueError("v0 is the Raw baseline")
    if config["seed"] != 2026 or config["jpeg_quality"] != 95:
        raise ValueError("Keep the shared seed and JPEG95 encoding")
    training, evaluation = config.get("training") or {}, config.get("evaluation") or {}
    if not REQUIRED_TRAINING <= training.keys() or not REQUIRED_EVALUATION <= evaluation.keys():
        raise ValueError("Each version must declare its native training and evaluation args")
    forbidden = {"data", "project", "name", "seed", "resume", "exist_ok", "split", "model"}
    if forbidden & (training.keys() | evaluation.keys()):
        raise ValueError("Paths, names, seed and resume are owned by the lifecycle, not the YAML")
    if Path(config["checkpoint"]).is_absolute() or not re.fullmatch(
        r"[0-9a-f]{64}", str(config.get("checkpoint_sha256"))
    ):
        raise ValueError("Use a project-relative checkpoint path pinned by its SHA256")
    return config


def policy_identity(config):
    return {k: config[k] for k in ("id", "variant", "recipes", "jpeg_quality", "seed")}


def run_dir(root, experiment, date=None, *, create=False):
    """`runs/<version>-<YYYY-MM-DD>`: explicit date, today when creating, else newest run."""
    runs = Path(root).resolve() / "runs"
    if date is None and create:
        date = datetime.date.today().isoformat()
    if date is not None:
        try:
            valid = datetime.date.fromisoformat(str(date)).isoformat() == str(date)
        except ValueError:
            valid = False
        if not valid:
            raise ValueError(f"Run date must be YYYY-MM-DD, got {date!r}")
        return runs / f"{experiment}-{date}"
    pattern = re.compile(rf"{experiment}-\d{{4}}-\d\d-\d\d")
    found = sorted(p for p in runs.glob(f"{experiment}-*") if pattern.fullmatch(p.name))
    if not found:
        raise FileNotFoundError(
            f"No runs/{experiment}-YYYY-MM-DD yet: in {EXPERIMENTS_DIR.as_posix()}/{experiment}/"
            "02_train_prepare.ipynb set ALLOW_PREPARATION = True and run it first"
        )
    return found[-1]


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
        **{split: [r for r in rows if r["role"] == role] for split, role in ROLES.items()},
    }


def verify_label(path, source):
    """Check native YOLO coordinates against frozen five-class boxes, including negatives."""
    actual = [list(map(float, line.split())) for line in Path(path).read_text().splitlines()]
    expected = [
        [
            c,
            (x1 + x2) / 2 / source["width"],
            (y1 + y2) / 2 / source["height"],
            (x2 - x1) / source["width"],
            (y2 - y1) / source["height"],
        ]
        for c, x1, y1, x2, y2 in source["boxes"]
    ]
    if len(actual) != len(expected) or any(len(box) != 5 for box in actual):
        raise ValueError("Export labels differ from the frozen boxes")
    if actual and not np.allclose(actual, expected, atol=1e-6, rtol=0):
        raise ValueError("Export labels differ from the frozen boxes")


def _data_config(config, shared, run):
    # No `path` key: Ultralytics resolves splits next to data.yaml, so runs stay relocatable.
    return {
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": dict(enumerate(FIVE_CLASSES)),
        "experiment": config["id"],
        "run": run.name,
        "training_subset": shared["training_subset"],
        "seed": shared["seed"],
        "policy_sha256": digest(policy_identity(config)),
        "splits_sha256": shared["splits_sha256"],
        "opencv": cv2.__version__,
    }


def _hash_pair(directory, split, sample_id):
    return (
        split,
        sample_id,
        file_hash(directory / "images" / split / f"{sample_id}.jpg"),
        file_hash(directory / "labels" / split / f"{sample_id}.txt"),
    )


def verify_data(root, experiment, date=None, *, workers=None):
    """Check data.yaml, inventories and exported bytes of one run, without reading sources."""
    root = Path(root).resolve()
    config = load_experiment(root, experiment)
    shared, rows = shared_splits(root)
    run = run_dir(root, experiment, date)
    directory = run / "data"
    saved = yaml.safe_load((directory / "data.yaml").read_text())
    ignored = {"files_sha256", "metadata_sha256", "operator_sha256"}
    if {k: v for k, v in saved.items() if k not in ignored} != _data_config(config, shared, run):
        raise ValueError(
            "Prepared policy/splits differ from the current config; prepare a new run date"
        )
    workers = workers or config["preparation"]["workers"]
    jobs = []
    for split, selected in populations(rows, shared["training_subset"]).items():
        for kind, suffix in (("images", ".jpg"), ("labels", ".txt")):
            folder = directory / kind / split
            paths = list(folder.iterdir())
            if {p.name for p in paths} != {r["id"] + suffix for r in selected}:
                raise ValueError("Prepared IDs differ from shared split/GT")
            if any(p.is_symlink() or not p.is_file() for p in paths):
                raise ValueError("Unexpected prepared symlink/directory")
        jobs += [(split, r["id"]) for r in selected]
    with ThreadPoolExecutor(workers) as pool:
        files = list(pool.map(lambda job: _hash_pair(directory, *job), jobs))
    if digest(files) != saved["files_sha256"]:
        raise ValueError("Prepared image/label bytes changed")
    if file_hash(directory / "metadata.csv") != saved["metadata_sha256"]:
        raise ValueError("Prepared metadata changed")
    return {**saved, "path": relative(directory, root)}


def _export(row, split, config, data, temporary):
    source = data / row["source_path"]
    if not source.resolve().is_relative_to(data):
        raise ValueError("Unexpected source path/symlink")
    if file_hash(source) != row["sha256"]:
        raise ValueError(f"Source image changed: {row['id']}")
    image = read_image(source)
    if image.shape != (row["height"], row["width"], 3):
        raise ValueError("Source geometry changed")
    output, applied, scores = apply_recipe(image, config["variant"], config["recipes"])
    if output.shape != image.shape or output.dtype != image.dtype:
        raise ValueError("Preprocessing changed geometry/dtype")
    target = temporary / "images" / split / f"{row['id']}.jpg"
    if not cv2.imwrite(str(target), output, [cv2.IMWRITE_JPEG_QUALITY, config["jpeg_quality"]]):
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
    metadata = {
        "id": row["id"],
        "split": split,
        "source_group": row["source_group"],
        "applied": "+".join(a["name"] for a in applied),
        **{k: scores[k] for k in FEATURES},
    }
    return _hash_pair(temporary, split, row["id"]), metadata


def prepare_data(root, experiment, date=None, *, workers=None):
    """Export originals once into runs/<version>-<date>/data; reuse only verified bytes."""
    root = Path(root).resolve()
    config = load_experiment(root, experiment)
    workers = workers or config["preparation"]["workers"]
    run = run_dir(root, experiment, date, create=True)
    destination = run / "data"
    if destination.exists():
        return verify_data(
            root, experiment, run.name.removeprefix(f"{experiment}-"), workers=workers
        )
    shared, rows = source_rows(root)
    native = _data_config(config, shared, run)
    native["operator_sha256"] = file_hash(Path(__file__).with_name("preprocess.py"))
    data = (root / shared["data_root"]).resolve()
    if destination.resolve().is_relative_to(data) or data.is_relative_to(destination.resolve()):
        raise ValueError("Export overlaps original sources")
    temporary = destination.with_name("data.tmp")
    temporary.mkdir(parents=True, exist_ok=False)
    files, metadata = [], []
    try:
        with ThreadPoolExecutor(workers) as pool:
            for split, selected in populations(rows, shared["training_subset"]).items():
                for kind in ("images", "labels"):
                    (temporary / kind / split).mkdir(parents=True)
                exported = pool.map(
                    lambda row, split=split: _export(row, split, config, data, temporary),
                    selected,
                )
                for hashes, meta in exported:
                    files.append(hashes)
                    metadata.append(meta)
        pd.DataFrame(metadata).to_csv(temporary / "metadata.csv", index=False)
        native["files_sha256"] = digest(files)
        native["metadata_sha256"] = file_hash(temporary / "metadata.csv")
        (temporary / "data.yaml").write_text(yaml.safe_dump(native, sort_keys=False))
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    (run / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    return {**native, "path": relative(destination, root)}
