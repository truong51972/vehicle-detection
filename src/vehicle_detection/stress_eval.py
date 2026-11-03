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
from .evaluation_stats import (  # Re-export historical research imports.
    MatchingStats,
)
from .evaluation_stats import (
    bootstrap_support as bootstrap_support,
)
from .evaluation_stats import (
    class_ap as class_ap,
)
from .evaluation_stats import (
    load_stats as load_stats,
)
from .evaluation_stats import (
    native_ap as native_ap,
)
from .evaluation_stats import (
    paired_interval as paired_interval,
)
from .stress import freeze, validate_pilot, verify_stress

SETTINGS = {**VAL_SETTINGS, "workers": 0}
STAT_KEYS = ("tp", "conf", "pred_cls", "target_cls")


class CalibrationValidator(MatchingStats, VehicleValidator):
    pass


def evaluate_calibration(protocol, pilot, dataset, checkpoint, output):
    validate_pilot(pilot, protocol)
    dataset, checkpoint, output = map(lambda p: Path(p).resolve(), (dataset, checkpoint, output))
    verify_stress(pilot, dataset)
    if file_hash(checkpoint) != PRETRAINED_SHA256:
        raise ValueError("Use the pinned unchanged pretrained YOLO26n checkpoint")
    identity = {
        "version": "issue11-native-v1",
        "protocol_id": protocol["manifest_id"],
        "pilot_id": pilot["pilot_id"],
        "role": "calibration",
        "export": read_json(dataset / "export.json"),
        "files_identity": read_json(dataset / "verified.json")["files_identity"],
        "checkpoint_sha256": PRETRAINED_SHA256,
        "settings": SETTINGS,
        "code_sha256": file_hash(__file__),
        "adapter_sha256": file_hash(Path(__file__).with_name("baseline.py")),
        "versions": {p: md.version(p) for p in ["ultralytics", "torch", "numpy", "opencv-python"]},
        "hardware": {
            "platform": platform.platform(),
            "gpu": torch.cuda.get_device_name(0),
            "cuda_runtime": torch.version.cuda,
        },
    }
    cached = reuse_evaluation(output / "native_result.json", identity)
    if cached is not None:
        return cached
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
    for c, name in enumerate(CLASSES):
        i = evaluated.get(c)
        classes.append(
            {
                "class": name,
                "images": pilot["summary"].get(f"presence:{name}", 0),
                "instances": pilot["summary"].get(f"instances:{name}", 0),
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
        "role": "calibration",
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
