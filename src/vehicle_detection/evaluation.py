"""Evaluate a detector on its own offline policy's shared development population."""

import json
from importlib.metadata import version
from pathlib import Path

import yaml
from ultralytics import YOLO

from .baseline import VAL_SETTINGS
from .common import file_hash, read_json, write_json
from .experiment_data import verify_data
from .five_class import FIVE_CLASSES


def evaluate_experiment(root, experiment, *, allow_inference=False, settings=None):
    root = Path(root).resolve()
    prepared = verify_data(root, experiment)
    directory = root / f"runs/{experiment}"
    checkpoint = directory / "train/weights/best.pt"
    if any(
        p.resolve() != p
        for p in (checkpoint, directory / "train/args.yaml", directory / "train.yaml")
    ):
        raise ValueError("Unexpected detector/YAML symlink")
    args = yaml.safe_load((directory / "train/args.yaml").read_text())
    if args["data"] != str(directory / "train.yaml"):
        raise ValueError("Detector belongs to another experiment")
    training = yaml.safe_load((directory / "train.yaml").read_text())
    raw = verify_data(root, "v0")
    if training != {
        **prepared,
        "val": str(Path(raw["path"]) / "images/val"),
        "selection_files_sha256": raw["files_sha256"],
    }:
        raise ValueError("Detector's prepared inputs/policy differ from evaluation")
    config = {**VAL_SETTINGS, "half": False, **(settings or {})}
    if config["nms"] is not False or config["augment"] or config["half"]:
        raise ValueError("Use native FP32 NMS-free validation without augmentation")
    identity = {
        "experiment": experiment,
        "data_sha256": file_hash(directory / "data/data.yaml"),
        "checkpoint_sha256": file_hash(checkpoint),
        "settings": config,
        "ultralytics": version("ultralytics"),
    }
    output = directory / "eval"
    saved = output / "metrics.json"
    if output.resolve() != output or saved.is_symlink():
        raise ValueError("Unexpected evaluation/checkpoint symlink")
    if saved.exists():
        result = read_json(saved)
        if result["inputs"] != identity:
            raise ValueError("Evaluation inputs/settings changed; use a new experiment ID")
        return result
    if not allow_inference:
        raise FileNotFoundError("No saved metrics; explicitly set ALLOW_INFERENCE=True")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Incomplete/unknown evaluation directory; inspect it before retrying")
    model = YOLO(str(checkpoint))
    if model.names != dict(enumerate(FIVE_CLASSES)):
        raise ValueError("Detector taxonomy differs from shared five-class labels")
    if model.model.yaml.get("yaml_file") != "yolo26n.yaml":
        raise ValueError("Expected YOLO26n detector")
    if model.ckpt["train_args"]["data"] != args["data"]:
        raise ValueError("Checkpoint belongs to another experiment")
    metrics = model.val(
        data=str(directory / "data/data.yaml"),
        split="val",
        project=str(directory),
        name="eval",
        exist_ok=True,
        **config,
    )
    precision, recall, map50, map5095 = map(float, metrics.mean_results())
    result = {
        "inputs": identity,
        "overall": {
            "precision": precision,
            "recall": recall,
            "mAP50": map50,
            "mAP50_95": map5095,
        },
        "classes": json.loads(metrics.to_json(decimals=8)),
        "speed_ms": metrics.speed,
        "role": "development_validation",
        "final_test_accessed": False,
    }
    write_json(saved, result)
    return result


def historical_result(root, branch):
    """Read committed historical measurements without relabeling them as new runs."""
    if branch not in {"raw", "adaptive"}:
        raise ValueError("Expected a historical Raw/Adaptive branch")
    root = Path(root)
    training = read_json(root / "docs/issue21/s10_results.json")
    matrix = read_json(root / "docs/development-five-v1/matrix.json")
    result = next(r for r in matrix["evaluations"] if r["evaluation"] == f"{branch}-{branch}")
    identity = result["identity"]
    if (
        training["final_test_accessed"]
        or matrix["final_test_accessed"]
        or identity["class_names"] != FIVE_CLASSES
        or identity["checkpoint_sha256"]
        != training["branches"][branch]["checkpoints"]["best"]["sha256"]
        or identity["protocol_id"] != training["protocol_id"]
        or identity["policy_id"] != training["policy_id"]
    ):
        raise ValueError("Historical measurement provenance differs")
    return result
