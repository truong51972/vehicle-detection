"""Native YOLO26n calibration metrics and paired group uncertainty; no final-test API."""

import importlib.metadata as md
import platform
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from .baseline import PRETRAINED_SHA256, VAL_SETTINGS, VehicleValidator, verify_coco_names
from .common import digest, file_hash, read_json, reuse_evaluation, write_json
from .data import CLASSES
from .datasets import verify_label
from .evaluation_stats import (  # noqa: F401 -- historical research re-exports
    STAT_KEYS,
    MatchingStats,
    bootstrap_support,
    class_ap,
    load_stats,
    native_ap,
    paired_interval,
)
from .five_class import FIVE_CLASSES, SIX_TO_FIVE, support_audit
from .stress import freeze, validate_pilot, verify_stress

SETTINGS = {**VAL_SETTINGS, "workers": 0}


class CalibrationValidator(MatchingStats, VehicleValidator):
    pass


def evaluate_calibration(protocol, pilot, dataset, checkpoint, output, *, allow_inference=True):
    validate_pilot(pilot, protocol)
    verify_stress(pilot, Path(dataset).resolve())
    return _evaluate_pretrained(
        protocol, pilot, dataset, checkpoint, output, allow_inference=allow_inference
    )


def evaluate_transfer(
    protocol, manifest, sample, dataset, checkpoint, output, *, allow_inference=False
):
    """One fixed development transfer check with the same pinned native calibration detector."""
    from .preprocessing_experiments import validate_sample, verify_candidate_export

    if sample["role"] != "development_validation":
        raise ValueError("Transfer requires the fixed development role")
    validate_sample(protocol, manifest, sample)
    verify_candidate_export(sample, dataset)
    return _evaluate_pretrained(
        protocol, sample, dataset, checkpoint, output, allow_inference=allow_inference
    )


def _evaluate_pretrained(protocol, pilot, dataset, checkpoint, output, *, allow_inference):
    dataset, checkpoint, output = map(lambda p: Path(p).resolve(), (dataset, checkpoint, output))
    export = read_json(dataset / "export.json")
    names = export["conversion"].get("classes", CLASSES)
    if names not in (CLASSES, FIVE_CLASSES):
        raise ValueError("Expected five/six-class calibration labels")
    for row in pilot["rows"]:
        verify_label(
            dataset / "labels/tune" / f"{row['id']}.txt",
            row,
            SIX_TO_FIVE if names == FIVE_CLASSES else None,
        )
    if file_hash(checkpoint) != PRETRAINED_SHA256:
        raise ValueError("Use the pinned unchanged pretrained YOLO26n checkpoint")
    identity = {
        "version": "issue11-native-v1",
        "protocol_id": protocol["manifest_id"],
        "pilot_id": pilot["pilot_id"],
        "role": pilot["role"],
        "export": export,
        "files_identity": read_json(dataset / "verified.json")["files_identity"],
        "checkpoint_sha256": PRETRAINED_SHA256,
        "settings": SETTINGS,
        "code_sha256": file_hash(__file__),
        "adapter_sha256": file_hash(Path(__file__).with_name("baseline.py")),
        "versions": {p: md.version(p) for p in ["ultralytics", "torch", "numpy", "opencv-python"]},
        "hardware": {
            "platform": platform.platform(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda_runtime": torch.version.cuda,
        },
    }
    cached = reuse_evaluation(output / "native_result.json", identity)
    if cached is not None:
        return cached
    if not allow_inference:
        raise FileNotFoundError(f"No verified result at {output}; set ALLOW_INFERENCE=True")
    if not (output / "identity.json").exists() and output.exists() and any(output.iterdir()):
        raise ValueError("Unknown stress evaluation directory")
    freeze(output / "identity.json", identity)
    model = YOLO(str(checkpoint))
    verify_coco_names(model.names)
    if model.model.yaml.get("yaml_file") != "yolo26n.yaml":
        raise ValueError("Wrong pretrained architecture")
    expected = {r["id"] for r in pilot["rows"]}
    runtime = {}

    def check_loader(validator):
        files = validator.dataloader.dataset.im_files
        if len(files) != len(expected) or {Path(f).stem for f in files} != expected:
            raise ValueError("Native loader changed calibration IDs")
        runtime.update(images=len(files), device=str(validator.device))

    def check_mode(validator):
        if not validator.end2end or validator.args.nms is not False:
            raise ValueError("NMS-free inference contract changed")
        runtime.update(end2end=True)

    model.add_callback("on_val_start", check_loader)
    model.add_callback("on_val_end", check_mode)
    metrics = model.val(
        validator=CalibrationValidator,
        data=str(dataset / "data.yaml"),
        split="val",
        project=str(output.parent),
        name=output.name,
        exist_ok=True,
        **SETTINGS,
    )
    if file_hash(checkpoint) != PRETRAINED_SHA256:
        raise ValueError("Checkpoint changed during the study")
    overall = dict(
        zip(
            ["mAP50_95", "mAP50", "precision", "recall"],
            map(float, [metrics.box.map, metrics.box.map50, metrics.box.mp, metrics.box.mr]),
            strict=True,
        )
    )
    if not np.isfinite(list(overall.values())).all():
        raise ValueError("Nonfinite native metrics")
    evaluated = {int(c): i for i, c in enumerate(metrics.box.ap_class_index)}
    classes = []
    support = support_audit(pilot["rows"]) if names == FIVE_CLASSES else pilot["summary"]
    prefix = "new_" if names == FIVE_CLASSES else ""
    for c, name in enumerate(names):
        i = evaluated.get(c)
        classes.append(
            {
                "class": name,
                "images": support.get(f"{prefix}presence:{name}", 0),
                "instances": support.get(f"{prefix}instances:{name}", 0),
                "AP50_95": float(metrics.box.ap[i]) if i is not None else None,
                "AP50": float(metrics.box.ap50[i]) if i is not None else None,
                "recall": float(metrics.box.r[i]) if i is not None else None,
            }
        )
    artifacts = [*output.glob("*.png"), output / "paired_stats.npz"]
    result = {
        "overall": overall,
        "classes": classes,
        "runtime": runtime,
        "model_speed_ms": metrics.speed,
        "artifacts": {p.name: file_hash(p) for p in artifacts},
        "role": pilot["role"],
        "final_test_accessed": False,
    }
    write_json(
        output / "native_result.json",
        {"identity": identity, "result": result, "checksum": digest(result)},
    )
    return result


def load_recorded_calibration(
    directory, expected, *, pilot, protocol_id, checkpoint, settings, code
):
    """Verify original measured artifacts for analysis; never migrate inference identities."""
    directory = Path(directory)
    path = directory / "native_result.json"
    if file_hash(path) != expected["native_result_sha256"]:
        raise ValueError("Recorded native result changed")
    cached = read_json(path)
    identity = cached["identity"]
    required = {
        "role": "calibration",
        "pilot_id": pilot["pilot_id"],
        "protocol_id": protocol_id,
        "checkpoint_sha256": checkpoint,
        "settings": settings,
        "code_sha256": code,
        "files_identity": expected["files_identity"],
    }
    if any(identity.get(k) != v for k, v in required.items()):
        raise ValueError("Recorded inference identity/role differs")
    result = reuse_evaluation(path, identity)
    stats_path = directory / "paired_stats.npz"
    if file_hash(stats_path) != expected["statistics_sha256"]:
        raise ValueError("Recorded prediction statistics changed")
    stats = load_stats(stats_path)
    if set(stats) != {r["id"] for r in pilot["rows"]}:
        raise ValueError("Recorded statistics contain different pilot IDs")
    return result, stats
