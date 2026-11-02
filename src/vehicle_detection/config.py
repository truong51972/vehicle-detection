"""Load presets and reject typos before an expensive run starts."""

import copy
import math
from pathlib import Path

import yaml

PRESETS = Path(__file__).parent / "presets"
CONDITIONS = {"clean", "dark", "blur", "noise", "haze"}


def merge(base, updates):
    for key, value in updates.items():
        if key not in base:
            raise ValueError(f"Unknown configuration key: {key}")
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"Expected a mapping for {key}")
            merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def number(value, name, minimum, maximum, integer=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not minimum <= value <= maximum
        or (integer and not isinstance(value, int))
    ):
        raise ValueError(
            f"{name} must be {'an integer' if integer else 'a number'} in [{minimum}, {maximum}]"
        )


def load_config(preset="pilot", path=None, overrides=None):
    config = yaml.safe_load((PRESETS / "pilot.yaml").read_text())
    if preset == "full":
        merge(config, yaml.safe_load((PRESETS / "full.yaml").read_text()))
    if path:
        updates = yaml.safe_load(Path(path).read_text())
        if not isinstance(updates, dict):
            raise ValueError("Configuration must be a YAML mapping")
        merge(config, updates)
    if overrides:
        merge(config, overrides)
    validate(config)
    return config


def validate(config):
    from .preprocess import validate_operators

    number(config["seed"], "seed", 0, 2**32 - 1, True)
    data, training, policy, bench = (config[k] for k in ("data", "training", "policy", "benchmark"))
    number(data["tune_fraction"], "tune_fraction", 0.01, 0.99)
    for key in ("train_limit", "tune_limit", "holdout_limit"):
        if data[key] is not None:
            number(data[key], key, 1, 10**9, True)
    number(data["quality_samples"], "quality_samples", 1, 10**9, True)
    for key in ("epochs", "batch", "imgsz"):
        number(training[key], key, 1, 10**6, True)
    if training["imgsz"] % 32:
        raise ValueError("imgsz must be a multiple of 32")
    if not isinstance(training["model"], str) or not training["model"]:
        raise ValueError("training.model must name a checkpoint or model YAML")
    if not isinstance(training["device"], (str, int)) or isinstance(training["device"], bool):
        raise ValueError("training.device must be cpu, mps, or a single CUDA device")
    number(policy["max_filters"], "max_filters", 0, 2, True)
    for key in ("brightness_below", "contrast_below", "noise_above"):
        number(policy[key], key, 0, 1)
    number(policy["sharpness_min"], "sharpness_min", 0, 10**6)
    number(policy["sharpness_below"], "sharpness_below", policy["sharpness_min"], 10**6)
    validate_operators(config["operators"])
    enabled = {op["name"] for op in config["operators"] if op["enabled"]}
    for key, allowed in (("conditions", CONDITIONS), ("treatments", enabled | {"raw", "adaptive"})):
        values = bench[key]
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(v, str) for v in values)
            or len(set(values)) != len(values)
            or not set(values) <= allowed
        ):
            raise ValueError(f"Invalid/duplicate benchmark.{key}: {values}")
    if "raw" not in bench["treatments"]:
        raise ValueError("benchmark.treatments must include the raw control")
    for key in ("warmup", "samples"):
        number(bench[key], key, 1, 1000, True)
    number(bench["min_stratum"], "min_stratum", 1, 10**9, True)
    number(bench["max_det"], "max_det", 1, 10000, True)
    for key in ("confidence", "nms_iou", "precision_confidence"):
        number(bench[key], key, 0.000001, 1)
    if bench["confidence"] > bench["precision_confidence"]:
        raise ValueError("prediction confidence must not exceed precision_confidence")
    for key, lo, hi in (
        ("dark_factor", 0.01, 1),
        ("blur_sigma", 0.1, 20),
        ("noise_sigma", 0, 100),
        ("haze_alpha", 0, 1),
    ):
        number(config["stress"][key], key, lo, hi)
