"""Historical six-class COCO→BDD prediction adapter; native YOLO computes metrics."""

import torch
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.metrics import ConfusionMatrix

from .data import CLASSES

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
