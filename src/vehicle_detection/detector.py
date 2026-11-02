"""Optional Ultralytics boundary: raw training and frozen image prediction."""

import csv
from pathlib import Path

import numpy as np
import yaml

from .common import environment, file_hash, read_json, write_json
from .data import CLASSES, load_manifest, yolo_labels


def require_yolo():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ValueError("Detector dependencies missing. Run: uv sync --extra detector") from exc
    return YOLO


def train(prepared, output, config):
    prepared, output = Path(prepared).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError(f"Training output already exists: {output}; choose a new directory")
    manifest = load_manifest(prepared)
    expected_dataset = {
        "path": str(prepared),
        "train": "images/train",
        "val": "images/tune",
        "names": dict(enumerate(CLASSES)),
    }
    if yaml.safe_load((prepared / "dataset.yaml").read_text()) != expected_dataset:
        raise ValueError("Prepared dataset.yaml changed; refusing possible holdout leakage")
    # Verify exports too: training must consume the exact raw images and labels in the manifest.
    for split in ("train", "tune"):
        if not any(r["boxes"] for r in manifest["partitions"][split]):
            raise ValueError(f"No vehicle boxes in {split}; increase sample limits")
        for r in manifest["partitions"][split]:
            exported = prepared / "images" / split / (r["id"] + Path(r["path"]).suffix)
            labels = prepared / "labels" / split / f"{r['id']}.txt"
            if exported.resolve() != Path(r["path"]) or labels.read_text() != yolo_labels(
                r["boxes"], r["width"], r["height"]
            ):
                raise ValueError(f"Prepared export changed: {r['id']}")
        expected_images = {r["id"] + Path(r["path"]).suffix for r in manifest["partitions"][split]}
        if {p.name for p in (prepared / "images" / split).iterdir()} != expected_images:
            raise ValueError(f"Unexpected image files in {split}; prepare a new export")
    model = require_yolo()(config["training"]["model"])
    settings = dict(config["training"])
    settings.pop("model")
    initial_weights = Path(getattr(model, "ckpt_path", "") or config["training"]["model"])
    initial_hash = file_hash(initial_weights) if initial_weights.is_file() else None
    result = model.train(
        **settings,
        data=str(prepared / "dataset.yaml"),
        seed=config["seed"],
        deterministic=True,
        project=str(output.parent),
        name=output.name,
        exist_ok=False,
        workers=0,
        plots=True,
        pretrained=True,
        val=True,
    )
    actual = Path(model.trainer.save_dir)
    best, last = actual / "weights/best.pt", actual / "weights/last.pt"
    if not best.is_file() or not last.is_file():
        raise ValueError(f"Training did not produce best/last checkpoints in {actual}")
    # Keep the unmodified training CSV, including last-epoch tuning metrics.
    with (actual / "results.csv").open(newline="") as stream:
        epochs = [{k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(stream)]
    provenance = {
        "manifest_identity": manifest["identity"],
        "config": config,
        "environment": environment(),
        "initial_model": {
            "source": config["training"]["model"],
            "sha256": initial_hash,
            "kind": "checkpoint" if initial_weights.suffix == ".pt" else "architecture",
        },
        "best": {"path": str(best), "sha256": file_hash(best)},
        "last": {"path": str(last), "sha256": file_hash(last)},
        "best_tuning_metrics": {k: float(v) for k, v in result.results_dict.items()},
        "last_epoch": epochs[-1] if epochs else {},
        "selection": "Best checkpoint selected on train-derived tune only; official val not used",
    }
    write_json(actual / "training.json", provenance)
    return best


class YoloDetector:
    def __init__(self, weights, config):
        self.weights = Path(weights).resolve()
        if not self.weights.is_file():
            raise ValueError(f"Frozen checkpoint not found: {weights}; run vd train first")
        self.model = require_yolo()(str(self.weights))
        names = self.model.names
        ordered = [names[i] for i in range(len(names))] if isinstance(names, dict) else list(names)
        if ordered != CLASSES:
            raise ValueError(
                f"Checkpoint classes {ordered} do not match {CLASSES}; "
                "fine-tune on the prepared vehicle dataset first"
            )
        self.config = config
        self.identity = {"weights_sha256": file_hash(self.weights), "classes": ordered}
        training_record = self.weights.parent.parent / "training.json"
        self.training_provenance = read_json(training_record) if training_record.is_file() else None

    def predict(self, image):
        training, bench = self.config["training"], self.config["benchmark"]
        result = self.model.predict(
            source=image,
            imgsz=training["imgsz"],
            device=training["device"],
            conf=bench["confidence"],
            iou=bench["nms_iou"],
            max_det=bench["max_det"],
            batch=1,
            half=False,
            augment=False,
            rect=False,
            verbose=False,
            save=False,
        )[0]
        return np.column_stack(
            (
                result.boxes.cls.cpu().numpy(),
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.conf.cpu().numpy(),
            )
        ).tolist()

    def synchronize(self):
        import torch

        device = self.model.predictor.device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()

    def hardware(self):
        import torch

        device = self.model.predictor.device
        return {
            "device": str(device),
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch_threads": torch.get_num_threads(),
        }
