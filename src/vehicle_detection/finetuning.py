"""Native S10 training on saved offline exports; no paired receipt chain."""

from pathlib import Path

import pandas as pd
import yaml
from ultralytics import YOLO

from .baseline import PRETRAINED_SHA256, verify_coco_names
from .common import file_hash
from .experiment_data import load_experiment, verify_data
from .five_class import FIVE_CLASSES
from .training_settings import TRAIN_SETTINGS, validate_training_args


def backbone_boundary(model):
    """Verify the pinned YOLO26n layout before passing its boundary to native freeze."""
    if model.yaml.get("yaml_file") != "yolo26n.yaml" or not model.yaml.get("end2end"):
        raise ValueError("Expected the installed end-to-end YOLO26n architecture")
    boundary = len(model.yaml["backbone"])
    if boundary != 11 or not 0 < boundary < len(model.model):
        raise ValueError("Pinned YOLO26n backbone changed; review native freeze boundary")
    return boundary


def train_experiment(root, experiment, *, allow_training=False, resume=False):
    """Select on shared Raw development, using native args/history/checkpoints for reuse."""
    root = Path(root).resolve()
    config = load_experiment(root, experiment)
    prepared = verify_data(root, experiment)
    raw = verify_data(root, "v0")
    directory = root / f"runs/{experiment}"
    data = directory / "train.yaml"
    native = {
        **prepared,
        "val": str(Path(raw["path"]) / "images/val"),
        "selection_files_sha256": raw["files_sha256"],
    }
    if data.is_symlink():
        raise ValueError("Unexpected training YAML symlink")
    if data.exists() and yaml.safe_load(data.read_text()) != native:
        raise ValueError("Training data changed; use a new experiment ID")
    if not data.exists():
        data.write_text(yaml.safe_dump(native))
    settings = {**TRAIN_SETTINGS, **config["training"], "seed": config["seed"]}
    run = directory / "train"
    if run.resolve() != run:
        raise ValueError("Unexpected training directory symlink")
    args_path = run / "args.yaml"
    best, last, history = run / "weights/best.pt", run / "weights/last.pt", run / "results.csv"
    if any(p.resolve() != p for p in (args_path, best, last, history)):
        raise ValueError("Unexpected native artifact symlink")
    if args_path.exists():
        args = yaml.safe_load(args_path.read_text())
        validate_training_args(args, data, settings)
        if best.is_file() and history.is_file():
            table = pd.read_csv(history)
            if table.epoch.tolist() == list(range(1, settings["epochs"] + 1)):
                if resume:
                    raise ValueError("Training is already complete")
                return table
    elif run.exists() and any(run.iterdir()):
        raise ValueError("Unknown native training directory")
    if not allow_training:
        raise FileNotFoundError("No completed run; explicitly set ALLOW_TRAINING=True")
    if resume:
        if not args_path.is_file() or not last.is_file():
            raise FileNotFoundError("Native resume requires this experiment's args.yaml/last.pt")
        model = YOLO(str(last))
        if model.ckpt["train_args"]["data"] != str(data) or model.names != dict(
            enumerate(FIVE_CLASSES)
        ):
            raise ValueError("Resume checkpoint belongs to another experiment/taxonomy")
        model.train(resume=True)
    else:
        if args_path.exists():
            raise ValueError("Interrupted training; explicitly RESUME=True")
        checkpoint = root / config["checkpoint"]
        if (
            config["checkpoint_sha256"] != PRETRAINED_SHA256
            or file_hash(checkpoint) != PRETRAINED_SHA256
        ):
            raise ValueError("Use the unchanged pinned pretrained YOLO26n checkpoint")
        model = YOLO(str(checkpoint))
        verify_coco_names(model.names)
        model.train(
            data=str(data),
            freeze=backbone_boundary(model.model),
            project=str(directory),
            name="train",
            exist_ok=True,
            **settings,
        )
    if not best.is_file() or not history.is_file():
        raise FileNotFoundError("Native training did not publish best.pt/results.csv")
    table = pd.read_csv(history)
    if table.epoch.tolist() != list(range(1, settings["epochs"] + 1)):
        raise ValueError("Native training has not completed the epoch budget")
    return table
