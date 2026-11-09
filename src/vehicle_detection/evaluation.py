"""Evaluate one run's selected detector on its own policy's development (test) split."""

import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator

from .common import file_hash, read_json, relative, write_json
from .data import FIVE_CLASSES
from .evaluation_stats import MatchingStats, class_ap, load_stats
from .experiment_data import load_experiment, run_dir, verify_data
from .training import completed_training, training_settings

DARK_BELOW = 0.12


class ExperimentValidator(MatchingStats, DetectionValidator):
    """Retain native matches for subgroup/error analysis of new evaluations."""


def cohort_metrics(stats, metadata):
    """Per-class AP for raw-pixel dark/bright and, for routed policies, Apply/Skip images."""
    test = metadata[metadata.split == "test"]
    cohorts = {
        "dark": test.id[test.brightness < DARK_BELOW],
        "bright": test.id[test.brightness >= DARK_BELOW],
    }
    applied = test.applied.astype(str)
    if applied.ne("").any() and applied.eq("").any():
        cohorts |= {"apply": test.id[applied.ne("")], "skip": test.id[applied.eq("")]}
    result = {}
    for name, ids in cohorts.items():
        values = class_ap(stats, list(ids), detailed=True)
        result[name] = {
            "images": len(ids),
            "mAP50_95": float(np.mean([v["AP50_95"] for v in values.values()])) if values else None,
            "classes": {FIVE_CLASSES[c]: v for c, v in values.items()},
        }
    return result


def evaluate_experiment(root, experiment, date=None, *, allow_inference=False, settings=None):
    root = Path(root).resolve()
    config = load_experiment(root, experiment)
    run = run_dir(root, experiment, date)
    prepared = verify_data(root, experiment, run.name.removeprefix(f"{experiment}-"))
    if completed_training(root, run, training_settings(config)) is None:
        raise FileNotFoundError("Training has not completed for this run; run 03_train first")
    checkpoint = run / "train/weights/best.pt"
    eval_settings = {**config["evaluation"], **(settings or {})}
    if eval_settings.get("augment") or eval_settings.get("half"):
        raise ValueError("Report native FP32 evaluation without test-time augmentation")
    identity = {
        "run": relative(run, root),
        "files_sha256": prepared["files_sha256"],
        "checkpoint_sha256": file_hash(checkpoint),
        "settings": eval_settings,
        "ultralytics": version("ultralytics"),
    }
    output = run / "eval"
    saved = output / "metrics.json"
    if saved.exists():
        result = read_json(saved)
        if result["inputs"] != identity:
            raise ValueError("Evaluation inputs/settings changed; delete eval/ or use a new run")
        return result
    if not allow_inference:
        raise FileNotFoundError("No saved metrics; explicitly set ALLOW_INFERENCE=True")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Incomplete/unknown evaluation directory; inspect it before retrying")
    model = YOLO(str(checkpoint))
    if model.names != dict(enumerate(FIVE_CLASSES)):
        raise ValueError("Detector taxonomy differs from shared five-class labels")
    data = root / prepared["path"] / "data.yaml"
    # Ultralytics needs absolute runtime paths; metrics.json stores only relative ones.
    metrics = model.val(
        validator=ExperimentValidator,
        data=str(data),
        split="test",
        project=str(run),
        name="eval",
        exist_ok=True,
        seed=config["seed"],
        **eval_settings,
    )
    precision, recall, map50, map5095 = map(float, metrics.mean_results())
    stats = load_stats(output / "paired_stats.npz")
    metadata = pd.read_csv(root / prepared["path"] / "metadata.csv", keep_default_na=False)
    result = {
        "inputs": identity,
        "overall": {"precision": precision, "recall": recall, "mAP50": map50, "mAP50_95": map5095},
        "classes": json.loads(metrics.to_json(decimals=8)),
        "cohorts": cohort_metrics(stats, metadata),
        "speed_ms": metrics.speed,
        "role": "development_validation",
        "final_test_accessed": False,
    }
    write_json(saved, result)
    return result


def saved_result(root, experiment, date=None):
    """Newest (or dated) run of a version that has saved development metrics, else None."""
    runs = Path(root).resolve() / "runs"
    if date is not None:
        candidates = [run_dir(root, experiment, date)]
    else:
        candidates = sorted(runs.glob(f"{experiment}-????-??-??"), reverse=True)
    for run in candidates:
        if (run / "eval/metrics.json").is_file():
            return run, read_json(run / "eval/metrics.json")
    return None
