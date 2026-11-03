"""Native pretrained COCO-80 validation against the frozen six-class development set."""

import importlib.metadata as md
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.metrics import ConfusionMatrix

from .common import digest, file_hash, read_json, reuse_evaluation, write_json
from .data import CLASSES
from .offline import verify_alignment
from .protocol import validate_protocol

COCO_TO_VEHICLE = {2: 0, 7: 1, 5: 2, 6: 3, 3: 4, 1: 5}
PRETRAINED_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt"
PRETRAINED_SHA256 = "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"
VAL_SETTINGS = {
    "imgsz": 640,
    "batch": 32,
    "device": "0",
    "conf": 0.001,
    "iou": 0.7,
    "max_det": 300,
    "quantize": 32,
    "nms": False,
    "agnostic_nms": False,
    "rect": True,
    "augment": False,
    "seed": 2026,
    "deterministic": True,
    "workers": 8,
    "plots": True,
    "verbose": False,
}


def verify_coco_names(names):
    if len(names) != 80 or any(
        names.get(coco) != CLASSES[target] for coco, target in COCO_TO_VEHICLE.items()
    ):
        raise ValueError("Expected COCO-80 names/IDs; pretrained class semantics differ")


def remap_predictions(pred):
    """Filter native postprocessed predictions; keep boxes, scores and auxiliary rows aligned."""
    classes = pred["cls"]
    mapped = torch.full_like(classes, -1)
    for source, target in COCO_TO_VEHICLE.items():
        mapped[classes == source] = target
    keep = mapped >= 0
    return {**{k: v[keep] for k, v in pred.items()}, "cls": mapped[keep]}


class VehicleValidator(DetectionValidator):
    """Keep native matching/AP/plots, changing only prediction labels and metric names."""

    def init_metrics(self, model):
        verify_coco_names(model.names)
        if (
            self.training
            or self.args.split != "val"
            or self.data["names"] != dict(enumerate(CLASSES))
        ):
            raise ValueError("Pretrained adapter requires six-class development validation only")
        super().init_metrics(model)
        self.names = dict(enumerate(CLASSES))
        self.nc = len(CLASSES)
        self.class_map = list(range(1, self.nc + 1))
        self.metrics.names = self.names
        self.confusion_matrix = ConfusionMatrix(names=self.names)

    def postprocess(self, preds):
        return [remap_predictions(pred) for pred in super().postprocess(preds)]


def evaluate_pretrained(manifest, dataset, checkpoint, output, *, settings=None):
    """One identity-checked native validation, always the frozen development Raw population."""
    validate_protocol(manifest)
    dataset, checkpoint, output = (
        Path(dataset).resolve(),
        Path(checkpoint).resolve(),
        Path(output).resolve(),
    )
    selected = {r["id"]: r for r in manifest["splits"]["development_validation"]}
    meta = pd.read_csv(dataset / "metadata.csv")
    if set(meta.id) != set(selected) or set(meta.split) != {"tune"}:
        raise ValueError("Baseline dataset must contain exactly the frozen development IDs")
    export = read_json(dataset / "export.json")
    if export["variant"] != "raw_reencoded" or export["encoding"]["jpeg_quality"] != 95:
        raise ValueError("Expected Raw JPEG95 with no enhancement")
    for row in meta.to_dict("records"):
        source = selected[row["id"]]
        image = Path(row["image"])
        if image.parent != dataset / "images/tune" or image.stem != row["id"] or image.is_symlink():
            raise ValueError("Development images must be owned Raw export files")
        if row["source_sha256"] != source["sha256"] or (row["width"], row["height"]) != (
            source["width"],
            source["height"],
        ):
            raise ValueError("Development export source identity/geometry differs")
    verify_alignment([dataset], deep=True)
    config = {**VAL_SETTINGS, **(settings or {})}
    if set(config) != set(VAL_SETTINGS) or config["nms"] is not False or not config["plots"]:
        raise ValueError("Use explicit NMS-free validation settings and save native plots")
    if file_hash(checkpoint) != PRETRAINED_SHA256:
        raise ValueError("Checkpoint differs from the pinned untouched YOLO26n release")
    model = YOLO(str(checkpoint))
    verify_coco_names(model.names)
    if model.model.yaml.get("yaml_file") != "yolo26n.yaml":
        raise ValueError("Expected independent pretrained YOLO26n architecture")
    identity = {
        "manifest_id": manifest["manifest_id"],
        "role": "development_validation",
        "input": "Raw JPEG95",
        "checkpoint_sha256": file_hash(checkpoint),
        "checkpoint_url": PRETRAINED_URL,
        "mapping": {str(k): v for k, v in COCO_TO_VEHICLE.items()},
        "settings": config,
        "export": export,
        "files_identity": read_json(dataset / "verified.json")["files_identity"],
        "adapter_sha256": file_hash(__file__),
        "versions": {
            p: md.version(p)
            for p in ["ultralytics", "torch", "torchvision", "numpy", "opencv-python"]
        },
        "hardware": {
            "platform": platform.platform(),
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(int(config["device"]))
            if str(config["device"]).isdigit() and torch.cuda.is_available()
            else None,
        },
    }
    cached = reuse_evaluation(output / "native_result.json", identity)
    if cached is not None:
        return cached
    identity_path = output / "identity.json"
    if identity_path.exists():
        if read_json(identity_path) != identity:
            raise ValueError("Baseline inputs changed; use a new experiment root")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("Unknown baseline output directory")
    write_json(identity_path, identity)
    runtime = {}

    def check_loader(validator):
        files = validator.dataloader.dataset.im_files
        if len(files) != len(selected) or {Path(f).stem for f in files} != set(selected):
            raise ValueError("Native loader changed development IDs")
        runtime.update(device=str(validator.device), images=len(files))

    def check_inference_mode(validator):
        if not validator.end2end or validator.args.nms is not False:
            raise ValueError("Native runtime did not select the NMS-free head")
        if validator.args.max_det != config["max_det"]:
            raise ValueError("Native runtime changed max_det")
        runtime.update(end2end=validator.end2end, max_det=validator.args.max_det)

    model.add_callback("on_val_start", check_loader)
    model.add_callback("on_val_end", check_inference_mode)
    metrics = model.val(
        validator=VehicleValidator,
        data=str(dataset / "data.yaml"),
        split="val",
        project=str(output.parent),
        name=output.name,
        exist_ok=True,
        **config,
    )
    if file_hash(checkpoint) != identity["checkpoint_sha256"]:
        raise ValueError("Pretrained checkpoint changed during evaluation")
    values = [metrics.box.map50, metrics.box.map, metrics.box.mp, metrics.box.mr]
    if not np.isfinite(values).all() or not all(0 <= x <= 1 for x in values):
        raise ValueError("Invalid native metrics")
    evaluated = {int(cls): j for j, cls in enumerate(metrics.box.ap_class_index)}
    per_class = []
    for cls, name in enumerate(CLASSES):
        j = evaluated.get(cls)
        per_class.append(
            {
                "class": name,
                "images": manifest["summaries"]["development_validation"][f"presence:{name}"],
                "instances": manifest["summaries"]["development_validation"][f"instances:{name}"],
                "AP50": float(metrics.box.ap50[j]) if j is not None else None,
                "AP50_95": float(metrics.box.ap[j]) if j is not None else None,
                "precision": float(metrics.box.p[j]) if j is not None else None,
                "recall": float(metrics.box.r[j]) if j is not None else None,
            }
        )
    plots = sorted(output.glob("*.png"))
    if not (output / "confusion_matrix_normalized.png").exists() or not list(
        output.glob("*PR_curve.png")
    ):
        raise ValueError("Missing native metric plots")
    payload = {
        "overall": dict(
            zip(["mAP50", "mAP50_95", "precision", "recall"], map(float, values), strict=True)
        ),
        "classes": per_class,
        "runtime": runtime,
        "speed_ms": metrics.speed,
        "nms_mode": "nms=False: one-to-one end-to-end head, no external NMS",
        "holdout_evaluated": False,
        "artifacts": {p.name: file_hash(p) for p in plots},
    }
    write_json(
        output / "native_result.json",
        {"identity": identity, "result": payload, "checksum": digest(payload)},
    )
    return payload
