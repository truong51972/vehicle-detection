"""Development-only YOLO.val and verified cache replay. No training orchestration."""

import json
from importlib.metadata import version
from pathlib import Path

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator

from .adaptive import validate_policy
from .baseline import PRETRAINED_SHA256, VAL_SETTINGS, VehicleValidator, verify_coco_names
from .common import digest, file_hash, freeze_json, read_json, require_paths, reuse_evaluation, write_json
from .data import CLASSES
from .datasets import verify_role
from .evaluation_stats import MatchingStats
from .protocol import validate_protocol


class PretrainedValidator(MatchingStats, VehicleValidator):
    """COCO→BDD class remapping plus native matching and native metrics."""


class TrainedValidator(MatchingStats, DetectionValidator):
    """Native validation with optional saved per-image matches for grouped analysis."""


def evaluate_development(
    protocol,
    policy,
    dataset,
    checkpoint,
    output,
    *,
    pretrained=False,
    allow_inference=False,
    settings=None,
    class_names=None,
):
    """Validate development only; native best.pt is sufficient without training receipts."""
    validate_protocol(protocol)
    validate_policy(policy, protocol)
    dataset, checkpoint, output = (Path(p).resolve() for p in (dataset, checkpoint, output))
    variant = read_json(dataset / "export.json")["variant"]
    if variant not in ("raw_reencoded", "adaptive"):
        raise ValueError("Expected Raw/Adaptive development export, not final-test data")
    rows = protocol["splits"]["development_validation"]
    bound = verify_role(
        dataset,
        rows,
        "tune",
        variant,
        policy["recipes"] if variant == "adaptive" else None,
        deep=False,
    )
    names = list(CLASSES if class_names is None else class_names)
    if not names or (pretrained and names != CLASSES):
        raise ValueError("Pretrained COCO-80 mapping requires the historical six-class taxonomy")
    config = {**VAL_SETTINGS, "half": False, **(settings or {})}
    if config["nms"] is not False or config["augment"] or config["half"] or not config["plots"]:
        raise ValueError("Use fixed FP32 NMS-free native validation and plots")
    require_paths(checkpoint=checkpoint)
    weights_hash = file_hash(checkpoint)
    if pretrained and weights_hash != PRETRAINED_SHA256:
        raise ValueError("Expected the unchanged pinned pretrained YOLO26n checkpoint")

    identity = {
        "version": "development-native-v3",
        "protocol_id": protocol["manifest_id"],
        "policy_id": policy["policy_id"],
        "role": "development_validation",
        "dataset": bound,
        "checkpoint_sha256": weights_hash,
        "mapping": "COCO-80 to six vehicles" if pretrained else f"native {len(names)} vehicles",
        "class_names": names,
        "settings": config,
        "ultralytics": version("ultralytics"),
    }
    saved = output / "native_result.json"
    if saved.is_file():
        prior = read_json(saved)["identity"]
        if prior.get("version") in {"development-native-v1", "development-native-v2"}:
            # Read-only old six-class measurements; never rewrite their source hashes.
            comparable = (
                "protocol_id",
                "policy_id",
                "role",
                "dataset",
                "checkpoint_sha256",
                "mapping",
                "settings",
            )
            if names != CLASSES or any(prior.get(k) != identity[k] for k in comparable):
                raise ValueError("Historical evaluation cache inputs/roles changed")
            return reuse_evaluation(saved, prior)
    cached = reuse_evaluation(saved, identity)
    if cached is not None:
        return cached
    if not allow_inference:
        raise FileNotFoundError(f"No verified result at {saved}; set ALLOW_INFERENCE=True")

    freeze_json(output / "identity.json", identity)
    # Check actual exported file hashes before creating the model / doing new inference.
    verify_role(
        dataset,
        rows,
        "tune",
        variant,
        policy["recipes"] if variant == "adaptive" else None,
        deep=True,
    )
    model = YOLO(str(checkpoint))
    if pretrained:
        verify_coco_names(model.names)
    elif model.names != dict(enumerate(names)):
        raise ValueError("Checkpoint taxonomy differs from development labels")
    if model.model.yaml.get("yaml_file") != "yolo26n.yaml":
        raise ValueError("Expected YOLO26n checkpoint")

    metrics = model.val(
        validator=PretrainedValidator if pretrained else TrainedValidator,
        data=str(dataset / "data.yaml"),
        split="val",
        project=str(output.parent),
        name=output.name,
        exist_ok=True,
        **config,
    )
    plot = output / "confusion_matrix_normalized.png"
    stats = output / "paired_stats.npz"
    if not plot.is_file() or not stats.is_file():
        raise ValueError("Native plots/per-image matching statistics missing")
    precision, recall, map50, map5095 = map(float, metrics.mean_results())
    result = {
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
        "artifacts": {
            p.name: file_hash(p) for p in [*output.glob("*.png"), *output.glob("*.jpg"), stats]
        },
    }
    write_json(saved, {"identity": identity, "result": result, "checksum": digest(result)})
    return result
