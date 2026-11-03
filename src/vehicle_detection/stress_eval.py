"""Native YOLO26n calibration metrics and paired group uncertainty; no final-test API."""

import importlib.metadata as md
import platform
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.metrics import ap_per_class

from .baseline import PRETRAINED_SHA256, VAL_SETTINGS, VehicleValidator, verify_coco_names
from .common import digest, file_hash, read_json, reuse_evaluation, write_json
from .data import CLASSES
from .stress import freeze, validate_pilot, verify_stress

SETTINGS = {**VAL_SETTINGS, "workers": 0}
STAT_KEYS = ("tp", "conf", "pred_cls", "target_cls")


class CalibrationValidator(VehicleValidator):
    def init_metrics(self, model):
        self.image_ids = []
        super().init_metrics(model)

    def update_metrics(self, preds, batch):
        self.image_ids.extend(Path(p).stem for p in batch["im_file"])
        super().update_metrics(preds, batch)

    def get_stats(self):
        np.savez_compressed(
            self.save_dir / "paired_stats.npz",
            ids=np.array(self.image_ids),
            **{
                f"{key}_{i}": value
                for key in STAT_KEYS
                for i, value in enumerate(self.metrics.stats[key])
            },
        )
        return super().get_stats()


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


def load_stats(path):
    with np.load(path, allow_pickle=False) as data:
        ids = data["ids"].tolist()
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate prediction statistics IDs")
        return {
            sample_id: {key: data[f"{key}_{i}"] for key in STAT_KEYS}
            for i, sample_id in enumerate(ids)
        }


def native_ap(stats, ids):
    """Reaggregate native per-image IoU matches using Ultralytics' AP implementation."""
    arrays = {k: np.concatenate([stats[i][k] for i in ids], axis=0) for k in STAT_KEYS}
    if not len(arrays["target_cls"]):
        return None
    values = ap_per_class(arrays["tp"], arrays["conf"], arrays["pred_cls"], arrays["target_cls"])
    return float(values[5].mean())


def paired_interval(before, after, pilot, *, resamples=100, seed=2026):
    if set(before) != set(after) or set(before) != {r["id"] for r in pilot["rows"]}:
        raise ValueError("Paired statistics must contain identical calibration IDs")
    if type(resamples) is not int or resamples < 2:
        raise ValueError("At least two bootstrap resamples required")
    for sample_id in before:
        if not np.array_equal(before[sample_id]["target_cls"], after[sample_id]["target_cls"]):
            raise ValueError("Paired ground truth changed")
    groups = {}
    for row in pilot["rows"]:
        groups.setdefault(row["group"], []).append(row["id"])
    keys = sorted(groups)
    rng, deltas = np.random.default_rng(seed), []
    for _ in range(resamples):
        ids = [i for k in rng.choice(keys, len(keys), replace=True) for i in groups[k]]
        a, b = native_ap(before, ids), native_ap(after, ids)
        if a is not None and b is not None:
            deltas.append(b - a)
    if not deltas:
        raise ValueError("No supported classes in bootstrap")
    low, high = np.quantile(deltas, [0.025, 0.975])
    return {
        "recovery_ci_low": float(low),
        "recovery_ci_high": float(high),
        "resamples": resamples,
        "groups": len(keys),
        "interval": "paired group bootstrap 95%; exploratory, no multiplicity correction",
    }
