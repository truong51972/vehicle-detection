"""Native paired YOLO26n S10 training. Data integrity belongs to the export phase."""

from pathlib import Path

import pandas as pd
import yaml
from ultralytics import YOLO

from .adaptive import validate_policy
from .baseline import PRETRAINED_SHA256, VAL_SETTINGS, verify_coco_names
from .common import digest, file_hash, freeze_json, read_json
from .data import CLASSES
from .datasets import verify_role
from .offline import verify_alignment
from .subsets import validate_subsets

# Keep the historical S10 budget and disabled augmentations; both branches share it.
TRAIN_SETTINGS = {
    **VAL_SETTINGS,
    "epochs": 20,
    "patience": 0,
    "batch": 16,
    "amp": False,
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.01,
    "cos_lr": True,
    "warmup_epochs": 3.0,
    "warmup_bias_lr": 0.0,
    "weight_decay": 0.0005,
    "nbs": 64,
    "cls_remap": True,
    "rect": False,
    "cache": False,
    "fraction": 1.0,
    "compile": False,
    "channels_last": False,
    "save": True,
    "val": True,
    "augmentations": [],
    **dict.fromkeys(
        (
            "hsv_h",
            "hsv_s",
            "hsv_v",
            "degrees",
            "translate",
            "scale",
            "shear",
            "perspective",
            "flipud",
            "fliplr",
            "bgr",
            "mosaic",
            "mixup",
            "cutmix",
            "copy_paste",
            "multi_scale",
            "close_mosaic",
        ),
        0,
    ),
}


def backbone_boundary(model):
    """Verify the pinned YOLO26n layout before passing its boundary to native freeze."""
    if model.yaml.get("yaml_file") != "yolo26n.yaml" or not model.yaml.get("end2end"):
        raise ValueError("Expected the installed end-to-end YOLO26n architecture")
    boundary = len(model.yaml["backbone"])
    if boundary != 11 or not 0 < boundary < len(model.model):
        raise ValueError("Pinned YOLO26n backbone changed; review native freeze boundary")
    return boundary


def prepare_pair(protocol, subsets, policy, decision, pair, development, checkpoint, output):
    """One-time lightweight preflight; deep export verification occurs in notebook 02."""
    validate_subsets(subsets, protocol)
    validate_policy(policy, protocol)
    if (
        decision["decision_id"] != digest({k: v for k, v in decision.items() if k != "decision_id"})
        or decision["status"] != "v3_not_deployed"
        or decision["retained_policy_id"] != policy["policy_id"]
    ):
        raise ValueError("Expected the frozen decision retaining the issue #6 policy")
    pair, development, checkpoint, output = (
        Path(p).resolve() for p in (pair, development, checkpoint, output)
    )
    binding = read_json(pair / "pair.json")
    if (
        binding["protocol_id"] != protocol["manifest_id"]
        or binding["subsets_id"] != subsets["manifest_id"]
        or binding["policy_id"] != policy["policy_id"]
        or binding["subset"] != "S10"
        or binding["records_sha256"] != subsets["subsets"]["S10"]["records_sha256"]
    ):
        raise ValueError("Expected frozen S10 exports; expansion requires a new study")
    ids = set(subsets["subsets"]["S10"]["ids"])
    rows = [r for r in protocol["splits"]["training_pool"] if r["id"] in ids]
    training = {
        branch: verify_role(
            pair / branch,
            rows,
            "train",
            variant,
            policy["recipes"] if branch == "adaptive" else None,
            deep=False,
        )
        for branch, variant in (("raw", "raw_reencoded"), ("adaptive", "adaptive"))
    }
    # This one cross-variant check compares IDs, label digests, dimensions and splits.
    verify_alignment([pair / "raw", pair / "adaptive"], deep=False)
    development_export = verify_role(
        development,
        protocol["splits"]["development_validation"],
        "tune",
        "raw_reencoded",
        deep=False,
    )
    if file_hash(checkpoint) != PRETRAINED_SHA256:
        raise ValueError("Use the unchanged pinned pretrained YOLO26n checkpoint")
    model = YOLO(str(checkpoint))
    verify_coco_names(model.names)
    experiment = {
        "version": "paired-native-v1",
        "protocol_id": protocol["manifest_id"],
        "subsets_id": subsets["manifest_id"],
        "policy_id": policy["policy_id"],
        "policy_decision_id": decision["decision_id"],
        "training": training,
        "development": development_export,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": PRETRAINED_SHA256,
        "freeze": backbone_boundary(model.model),
        "settings": TRAIN_SETTINGS,
        "final_test_accessed": False,
    }
    if not (output / "pair.json").exists() and output.exists() and any(output.iterdir()):
        raise ValueError("Unknown training directory; choose an empty experiment root")
    freeze_json(output / "pair.json", experiment)
    for branch, dataset in training.items():
        config = {
            "path": str(output),
            "train": str(Path(dataset["path"]) / "images/train"),
            "val": str(development / "images/tune"),
            "names": dict(enumerate(CLASSES)),
        }
        path = output / f"{branch}.yaml"
        if path.exists() and yaml.safe_load(path.read_text()) != config:
            raise ValueError("Training YAML changed; use a new experiment root")
        if not path.exists():
            path.write_text(yaml.safe_dump(config))
    return experiment


def train_branch(experiment, branch, output, *, resume=False):
    """Run/reuse one branch using native YOLO.train and native last.pt resume."""
    output = Path(output).resolve()
    if branch not in ("raw", "adaptive") or read_json(output / "pair.json") != experiment:
        raise ValueError("Unknown branch or changed paired experiment")
    data = output / f"{branch}.yaml"
    expected = {
        "path": str(output),
        "train": str(Path(experiment["training"][branch]["path"]) / "images/train"),
        "val": str(Path(experiment["development"]["path"]) / "images/tune"),
        "names": dict(enumerate(CLASSES)),
    }
    if yaml.safe_load(data.read_text()) != expected:
        raise ValueError("Training YAML changed")
    # One cheap manifest/inventory check, never rehash/decode every training image.
    root = Path(experiment["training"][branch]["path"])
    verify_alignment([root], deep=False)
    if (
        read_json(root / "verified.json")["files_identity"]
        != experiment["training"][branch]["files_identity"]
    ):
        raise ValueError("Frozen training export identity changed")
    run = output / branch
    args_path = run / "args.yaml"
    if args_path.is_file():
        args = yaml.safe_load(args_path.read_text())
        if (
            args.get("data") != str(data)
            or args.get("freeze") != experiment["freeze"]
            or any(args.get(k) != experiment["settings"][k] for k in ("batch", "epochs", "imgsz"))
        ):
            raise ValueError("Native run settings changed; select a new output root")
    elif run.exists() and any(run.iterdir()):
        raise ValueError("Unknown native run directory; select a new output root")
    best, last, history = (run / "weights/best.pt", run / "weights/last.pt", run / "results.csv")
    if resume:
        if not args_path.is_file() or not last.is_file():
            raise FileNotFoundError("Native resume requires this branch's last.pt and args.yaml")
        YOLO(str(last)).train(resume=True)
    elif not (best.is_file() and last.is_file() and history.is_file()):
        if args_path.is_file():
            raise ValueError("Interrupted native run; explicitly resume=True or use a new root")
        if file_hash(experiment["checkpoint"]) != experiment["checkpoint_sha256"]:
            raise ValueError("Starting pretrained checkpoint changed")
        YOLO(experiment["checkpoint"]).train(
            data=str(data),
            freeze=experiment["freeze"],
            project=str(output),
            name=branch,
            exist_ok=True,
            **experiment["settings"],
        )
    for path in (best, last, history):
        if not path.is_file():
            raise FileNotFoundError(f"Native training did not publish {path}")
    table = pd.read_csv(history)
    if len(table) != experiment["settings"]["epochs"]:
        raise ValueError("Native training has not completed the declared epoch budget")
    metric = "metrics/mAP50-95(B)"
    result = {
        "budget": {
            "epochs": len(table),
            "images_per_epoch": len(experiment["training"][branch]["ids"]),
        },
        "best_epoch": int(table[metric].idxmax()) + 1 if metric in table else None,
        "best_checkpoint": str(best),
        "last_checkpoint": str(last),
    }
    return result
