"""Native Ultralytics training of one version on its own run export."""

from pathlib import Path

import pandas as pd
import yaml
from ultralytics import YOLO

from .common import file_hash, read_json, relative, write_json
from .data import FIVE_CLASSES
from .experiment_data import load_experiment, run_dir, verify_data

# COCO-80 IDs of the five vehicle classes in the pinned checkpoint.
COCO_VEHICLES = {2: "car", 7: "truck", 5: "bus", 3: "motorcycle", 1: "bicycle"}


def verify_coco_names(names):
    if len(names) != 80 or any(names.get(i) != name for i, name in COCO_VEHICLES.items()):
        raise ValueError("Expected COCO-80 names/IDs; pretrained class semantics differ")


def training_settings(config):
    return {**config["training"], "seed": config["seed"]}


def validate_training_args(args, run, settings):
    """Check native args.yaml against the version YAML without constructing a trainer."""
    if Path(args.get("data", "")).parts[-3:] != (run.name, "data", "data.yaml"):
        raise ValueError("Native training belongs to another run")
    changed = sorted(k for k, value in settings.items() if args.get(k) != value)
    if changed:
        raise ValueError(
            f"Native training settings differ from the version YAML ({', '.join(changed)}); "
            "restore the YAML or start a new run date"
        )


def completed_training(root, run, settings):
    """Return the history of a finished run, or None while no completed marker exists."""
    train = run / "train"
    marker = train / "complete.json"
    if not marker.is_file():
        return None
    validate_training_args(yaml.safe_load((train / "args.yaml").read_text()), run, settings)
    saved = read_json(marker)
    if file_hash(train / "weights/best.pt") != saved["best_sha256"]:
        raise ValueError("Selected checkpoint changed after training")
    history = pd.read_csv(train / "results.csv")
    history.columns = history.columns.str.strip()
    if len(history) != saved["epochs_run"]:
        raise ValueError("Training history changed after completion")
    return history


def train_experiment(root, experiment, date=None, *, allow_training=False, resume=False):
    """Train once from the version's pinned COCO checkpoint; calibration (val) selects best.pt."""
    root = Path(root).resolve()
    config = load_experiment(root, experiment)
    run = run_dir(root, experiment, date)
    prepared = verify_data(root, experiment, run.name.removeprefix(f"{experiment}-"))
    settings = training_settings(config)
    train = run / "train"
    history = completed_training(root, run, settings)
    if history is not None:
        if resume:
            raise ValueError("Training is already complete")
        return history
    args_path, last = train / "args.yaml", train / "weights/last.pt"
    if args_path.exists():
        validate_training_args(yaml.safe_load(args_path.read_text()), run, settings)
    elif train.exists() and any(train.iterdir()):
        raise ValueError("Unknown native training directory")
    if not allow_training:
        raise FileNotFoundError("No completed run; explicitly set ALLOW_TRAINING=True")
    data = root / prepared["path"] / "data.yaml"
    if resume:
        if not args_path.is_file() or not last.is_file():
            raise FileNotFoundError("Native resume requires this run's args.yaml/last.pt")
        model = YOLO(str(last))
        if Path(model.ckpt["train_args"]["data"]).resolve() != data or model.names != dict(
            enumerate(FIVE_CLASSES)
        ):
            raise ValueError("Resume checkpoint belongs to another run/taxonomy")
        model.train(resume=True)
    else:
        if args_path.exists():
            raise ValueError("Interrupted training; explicitly set RESUME=True")
        checkpoint = root / config["checkpoint"]
        if file_hash(checkpoint) != config["checkpoint_sha256"]:
            raise ValueError("Checkpoint bytes differ from the version's pinned checkpoint_sha256")
        model = YOLO(str(checkpoint))
        verify_coco_names(model.names)
        # Ultralytics needs absolute runtime paths; repository artifacts stay relative.
        model.train(data=str(data), project=str(run), name="train", exist_ok=True, **settings)
    best = train / "weights/best.pt"
    if not best.is_file() or not (train / "results.csv").is_file():
        raise FileNotFoundError("Native training did not publish best.pt/results.csv")
    history = pd.read_csv(train / "results.csv")
    history.columns = history.columns.str.strip()
    write_json(
        train / "complete.json",
        {
            "run": relative(run, root),
            "epochs_run": len(history),
            "best_sha256": file_hash(best),
            "selection": "calibration (Ultralytics val split, native fitness)",
        },
    )
    return history
