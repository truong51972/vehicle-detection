"""Detector identities and trusted, local-only checkpoint validation for the demo."""

import re
from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from .common import file_hash
from .data import FIVE_CLASSES

SIGN_MODELS = ("traffic-sign-hieu", "traffic-sign-phat")


@dataclass(frozen=True)
class DetectorSpec:
    version: str
    checkpoint: Path | None = None
    sha256: str = ""
    policy_id: str = ""
    imgsz: int = 640
    error: str = ""
    rejected: tuple[str, ...] = ()
    category: str = "vehicle"
    names: tuple[str, ...] = tuple(FIVE_CLASSES)
    preprocessing: str = "vehicle-policy-jpeg95"
    label: str = ""
    source_repository: str = "truong51972/vehicle-detection"
    source_revision: str = ""
    source_checkpoint: str = ""
    task: str = "detect"

    def identity(self):
        return {
            "model_id": self.version,
            "version": self.version,
            "category": self.category,
            "checkpoint": str(self.checkpoint) if self.checkpoint else None,
            "sha256": self.sha256,
            "training_policy_id": self.policy_id,
            "imgsz": self.imgsz,
            "task": self.task,
            "names": {str(i): name for i, name in enumerate(self.names)},
            "preprocessing": self.preprocessing,
            "source_repository": self.source_repository,
            "source_revision": self.source_revision,
            "source_checkpoint": self.source_checkpoint,
            "unavailable_reason": self.error,
            "rejected_runs": list(self.rejected),
        }


def load_detector(checkpoint, *, names=FIVE_CLASSES):
    """Only load trusted local checkpoints; never trigger an automatic weight download."""
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise ValueError(f"Missing checkpoint: {checkpoint}; restore the trusted local checkpoint")
    try:
        from ultralytics import YOLO

        model = YOLO(str(checkpoint))
        if model.task != "detect" or model.names != dict(enumerate(names)):
            raise ValueError(f"Expected detection taxonomy {names}")
        return model
    except Exception as exc:
        raise ValueError(f"Cannot load {checkpoint}: {exc}") from exc


def discover_sign_models(root):
    """Validate each sign independently, including its hash before deserializing weights."""
    root = Path(root).resolve()
    try:
        manifest = yaml.safe_load((root / "configs/external_models.yaml").read_text())
        if not isinstance(manifest, dict):
            raise ValueError("External model manifest must be a mapping")
    except Exception as exc:
        return {
            key: DetectorSpec(
                key,
                category="traffic-sign",
                names=(),
                preprocessing="native",
                error=f"{exc}; see docs/demo.md",
            )
            for key in SIGN_MODELS
        }
    registry = {}
    for key in SIGN_MODELS:
        spec = DetectorSpec(key, category="traffic-sign", names=(), preprocessing="native")
        try:
            config = dict(manifest[key])
            checkpoint = Path(config.pop("checkpoint"))
            if checkpoint.is_absolute() or not (root / checkpoint).resolve().is_relative_to(root):
                raise ValueError("Checkpoint path must stay inside the project")
            config["names"] = tuple(config["names"])
            spec = DetectorSpec(key, checkpoint=root / checkpoint, **config)
            if (
                spec.task != "detect"
                or spec.category != "traffic-sign"
                or spec.preprocessing != "native"
                or len(spec.names) != 6
                or len(set(spec.names)) != 6
                or any(not isinstance(name, str) or not name for name in spec.names)
                or type(spec.imgsz) is not int
                or not 32 <= spec.imgsz <= 4096
                or not re.fullmatch(r"[0-9a-f]{64}", spec.sha256)
                or not re.fullmatch(r"[0-9a-f]{40}", spec.source_revision)
            ):
                raise ValueError("Invalid sign task/taxonomy/preprocessing/imgsz/hash/revision")
            if not spec.checkpoint.is_file():
                raise ValueError(f"Missing checkpoint: {checkpoint}")
            if file_hash(spec.checkpoint) != spec.sha256:
                raise ValueError("Checkpoint SHA256 differs from the pinned source bytes")
            load_detector(spec.checkpoint, names=spec.names)
        except Exception as exc:
            spec = replace(spec, error=f"{exc}; install the trusted checkpoint per docs/demo.md")
        registry[key] = spec
    return registry
