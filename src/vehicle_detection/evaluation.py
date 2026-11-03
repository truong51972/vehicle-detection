"""Development-only native validation and saved-result reuse; never imports training."""

import importlib.metadata as md
import platform
from pathlib import Path

import numpy as np
import torch
import yaml
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator

from .adaptive import validate_policy
from .baseline import PRETRAINED_SHA256, VAL_SETTINGS, VehicleValidator, verify_coco_names
from .common import (
    digest,
    file_hash,
    freeze_json,
    read_json,
    require_paths,
    reuse_evaluation,
    write_json,
)
from .data import CLASSES
from .datasets import verify_role
from .evaluation_stats import MatchingStats
from .protocol import validate_protocol


class PretrainedValidator(MatchingStats, VehicleValidator):
    pass


class TrainedValidator(MatchingStats, DetectionValidator):
    def init_metrics(self, model):
        if (
            self.training
            or self.args.split != "val"
            or model.names != self.data["names"]
        ):
            raise ValueError("Native checkpoint classes differ from development dataset")
        super().init_metrics(model)


def completed_checkpoints(directory, protocol):
    """Load original training receipts, including historical code identity, without a trainer."""
    validate_protocol(protocol)
    directory = Path(directory).resolve()
    require_paths(training=directory / "pair.json")
    pair = read_json(directory / "pair.json")
    if pair.get("version") == "paired-native-v1":
        # New studies use Ultralytics args.yaml / best.pt, not custom completion receipts.
        expected_train = {r["id"] for r in protocol["splits"]["training_pool"]}
        expected_dev = sorted(r["id"] for r in protocol["splits"]["development_validation"])
        if (
            pair["protocol_id"] != protocol["manifest_id"]
            or pair.get("final_test_accessed")
            or pair["training"]["raw"]["ids"] != pair["training"]["adaptive"]["ids"]
            or not set(pair["training"]["raw"]["ids"]) <= expected_train
            or pair["development"]["ids"] != expected_dev
        ):
            raise ValueError("Native study changed protocol/role/paired IDs")
        checkpoints = {}
        for branch in ("raw", "adaptive"):
            run = directory / branch
            best = run / "weights/best.pt"
            args_path = run / "args.yaml"
            data_path = directory / f"{branch}.yaml"
            require_paths(checkpoint=best, args=args_path, data=data_path)
            args = yaml.safe_load(args_path.read_text())
            data = yaml.safe_load(data_path.read_text())
            if (
                args.get("data") != str(data_path)
                or args.get("freeze") != pair["freeze"]
                or any(args.get(k) != pair["settings"][k] for k in ("epochs", "batch", "imgsz"))
                or data["train"] != str(Path(pair["training"][branch]["path"]) / "images/train")
                or data["val"] != str(Path(pair["development"]["path"]) / "images/tune")
                or data["names"] != dict(enumerate(CLASSES))
            ):
                raise ValueError("Native checkpoint run/dataset identity changed")
            checkpoints[branch] = best
        return checkpoints
    if (
        pair["experiment_id"] != digest({k: v for k, v in pair.items() if k != "experiment_id"})
        or pair["protocol_id"] != protocol["manifest_id"]
        or pair["final_test_accessed"]
        or pair["training"]["raw"]["ids"] != pair["training"]["adaptive"]["ids"]
        or not set(pair["training"]["raw"]["ids"])
        <= {r["id"] for r in protocol["splits"]["training_pool"]}
        or pair["development"]["ids"]
        != sorted(r["id"] for r in protocol["splits"]["development_validation"])
    ):
        raise ValueError("Stale training manifest, roles or paired IDs")
    checkpoints, budgets = {}, []
    for branch in ("raw", "adaptive"):
        run = directory / branch
        require_paths(completion=run / "complete.json", identity=run / "identity.json")
        identity, receipt = read_json(run / "identity.json"), read_json(run / "complete.json")
        if (
            identity["pair"] != pair
            or identity["branch"] != branch
            or identity["run_id"] != digest({k: v for k, v in identity.items() if k != "run_id"})
            or receipt["run_id"] != identity["run_id"]
            or receipt["checksum"] != digest({k: v for k, v in receipt.items() if k != "checksum"})
            or receipt["final_test_accessed"]
            or not {"weights/best.pt", "weights/last.pt", "results.csv"}
            <= receipt["artifacts"].keys()
        ):
            raise ValueError("Stale training completion/identity")
        for name, expected in receipt["artifacts"].items():
            path = run / name
            if not path.resolve().is_relative_to(run) or file_hash(path) != expected:
                raise ValueError("Completed checkpoint/training artifact changed")
        budgets.append(receipt["budget"])
        checkpoints[branch] = run / "weights/best.pt"
    if budgets[0] != budgets[1]:
        raise ValueError("Paired training budgets differ")
    return checkpoints


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
    """Reuse verified results; missing inference requires an explicit opt-in."""
    validate_protocol(protocol)
    validate_policy(policy, protocol)
    dataset, checkpoint, output = map(lambda p: Path(p).resolve(), (dataset, checkpoint, output))
    require_paths(checkpoint=checkpoint, dataset=dataset / "metadata.csv")
    export = read_json(dataset / "export.json")
    variant = export["variant"]
    if variant not in {"raw_reencoded", "adaptive"}:
        raise ValueError("Expected frozen Raw or Adaptive development inputs")
    bound = verify_role(
        dataset,
        protocol["splits"]["development_validation"],
        "tune",
        variant,
        policy["recipes"] if variant == "adaptive" else None,
        deep=False,
    )
    target_classes = tuple(CLASSES if class_names is None else class_names)
    if not target_classes or (pretrained and target_classes != tuple(CLASSES)):
        raise ValueError("Pretrained COCO mapping is frozen to the historical six-class taxonomy")
    config = {**VAL_SETTINGS, "half": False, **(settings or {})}
    if config["nms"] is not False or config["augment"] or not config["plots"] or config["half"]:
        raise ValueError("Use fixed FP32 NMS-free validation with native plots")
    checkpoint_hash = file_hash(checkpoint)
    if pretrained and checkpoint_hash != PRETRAINED_SHA256:
        raise ValueError("Use the pinned untouched pretrained YOLO26n checkpoint")
    identity = {
        "version": "development-native-v2",
        "protocol_id": protocol["manifest_id"],
        "role": "development_validation",
        "policy_id": policy["policy_id"],
        "dataset": bound,
        "checkpoint_sha256": checkpoint_hash,
        "mapping": "COCO-80 to six vehicles" if pretrained else "native six vehicles",
        "class_names": list(target_classes),
        "settings": config,
        "code": {
            name: file_hash(Path(__file__).with_name(name))
            for name in (
                "evaluation.py",
                "evaluation_stats.py",
                "datasets.py",
                "baseline.py",
            )
        },
        "hardware": {
            "platform": platform.platform(),
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(int(config["device"]))
            if str(config["device"]).isdigit() and torch.cuda.is_available()
            else None,
        },
        "versions": {p: md.version(p) for p in ("ultralytics", "torch", "numpy", "opencv-python")},
    }
    result_path = output / "native_result.json"
    if result_path.is_file():
        saved = read_json(result_path)
        prior = saved["identity"]
        if prior.get("version") == "development-native-v1":
            # Read-only replay of original six-class caches, even if source code changed.
            comparable = ("protocol_id", "role", "policy_id", "dataset",
                          "checkpoint_sha256", "mapping", "settings")
            if target_classes != tuple(CLASSES) or any(
                prior.get(k) != identity.get(k) for k in comparable
            ):
                raise ValueError("Historical evaluation cache inputs/roles changed")
            return reuse_evaluation(result_path, prior)
    cached = reuse_evaluation(result_path, identity)
    if cached is not None:
        return cached
    if not allow_inference:
        raise FileNotFoundError(
            f"No valid saved evaluation at {output}; explicitly set ALLOW_INFERENCE=True "
            "in notebook 04 to evaluate existing inputs/checkpoints"
        )
    if not (output / "identity.json").exists() and output.exists() and any(output.iterdir()):
        raise ValueError("Unknown evaluation directory; choose a new version/root")
    freeze_json(output / "identity.json", identity)
    verify_role(
        dataset,
        protocol["splits"]["development_validation"],
        "tune",
        variant,
        policy["recipes"] if variant == "adaptive" else None,
        deep=True,
    )
    model = YOLO(str(checkpoint))
    if pretrained:
        verify_coco_names(model.names)
    elif model.names != dict(enumerate(target_classes)):
        raise ValueError("Trained checkpoint has the wrong class mapping")
    if model.model.yaml.get("yaml_file") != "yolo26n.yaml" or not model.model.yaml.get("end2end"):
        raise ValueError("Expected the native end-to-end YOLO26n architecture")
    expected = set(bound["ids"])
    runtime = {}

    def check_loader(validator):
        ids = [Path(p).stem for p in validator.dataloader.dataset.im_files]
        if len(ids) != len(expected) or set(ids) != expected:
            raise ValueError("Native loader changed frozen development IDs")
        runtime.update(images=len(ids), device=str(validator.device))

    def check_mode(validator):
        if not validator.end2end or validator.args.nms is not False:
            raise ValueError("NMS-free inference contract changed")
        runtime.update(end2end=True)

    model.add_callback("on_val_start", check_loader)
    model.add_callback("on_val_end", check_mode)
    metrics = model.val(
        validator=PretrainedValidator if pretrained else TrainedValidator,
        data=str(dataset / "data.yaml"),
        split="val",
        project=str(output.parent),
        name=output.name,
        exist_ok=True,
        **config,
    )
    if file_hash(checkpoint) != checkpoint_hash:
        raise ValueError("Checkpoint changed during evaluation")
    values = list(map(float, (metrics.box.map, metrics.box.map50, metrics.box.mp, metrics.box.mr)))
    if not np.isfinite(values).all() or not all(0 <= v <= 1 for v in values):
        raise ValueError("Invalid native metrics")
    indices = {int(c): i for i, c in enumerate(metrics.box.ap_class_index)}
    classes = []
    for cls, name in enumerate(target_classes):
        i = indices.get(cls)
        classes.append(
            {
                "class": name,
                "AP50_95": float(metrics.box.ap[i]) if i is not None else None,
                "AP50": float(metrics.box.ap50[i]) if i is not None else None,
                "precision": float(metrics.box.p[i]) if i is not None else None,
                "recall": float(metrics.box.r[i]) if i is not None else None,
            }
        )
    artifacts = [*output.glob("*.png"), *output.glob("*.jpg"), output / "paired_stats.npz"]
    if not (output / "confusion_matrix_normalized.png").is_file():
        raise ValueError("Missing native evaluation plots")
    result = {
        "overall": dict(zip(("mAP50_95", "mAP50", "precision", "recall"), values, strict=True)),
        "classes": classes,
        "speed_ms": metrics.speed,
        "runtime": runtime,
        "checkpoint_sha256": checkpoint_hash,
        "role": "development_validation",
        "final_test_accessed": False,
        "artifacts": {p.name: file_hash(p) for p in artifacts},
    }
    write_json(
        output / "native_result.json",
        {
            "identity": identity,
            "result": result,
            "checksum": digest(result),
        },
    )
    return result
