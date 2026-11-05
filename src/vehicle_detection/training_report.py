"""P5 provenance checks for paired native training and the issue #35 handoff."""

from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO

from .common import digest, file_hash, read_json
from .finetuning import backbone_boundary
from .five_class import FIVE_CLASSES


def verify_historical_reuse(root, historical):
    """Authenticate the actual pair, args, histories and weights against #21 evidence."""
    root = Path(root).resolve()
    if file_hash(root / "pair.json") != historical["pair_manifest_sha256"]:
        raise ValueError("Historical pair provenance differs; use a fresh training root")
    for branch in ("raw", "adaptive"):
        evidence = historical["branches"][branch]
        files = {
            "args.yaml": evidence["args_sha256"],
            "results.csv": evidence["history_sha256"],
            **{
                f"weights/{name}.pt": entry["sha256"]
                for name, entry in evidence["checkpoints"].items()
            },
        }
        for relative, expected in files.items():
            if file_hash(root / branch / relative) != expected:
                raise ValueError(f"Historical {branch}/{relative} differs; cannot claim reuse")


def training_evidence(experiment, preparation, completed, root, *, historical=None):
    """Verify native checkpoints and publish identities; never construct a trainer/evaluator."""
    root = Path(root).resolve()
    if (
        experiment["settings"] != read_json(root / "pair.json")["settings"]
        or any(experiment[k] != preparation[k] for k in ("protocol_id", "subsets_id", "policy_id"))
        or experiment["training"] != {b: preparation["exports"][b] for b in ("raw", "adaptive")}
        or experiment["development"] != preparation["exports"]["development"]
    ):
        raise ValueError("Training provenance differs from verified P4 inputs")
    if historical is not None:
        verify_historical_reuse(root, historical)
        if (
            historical["settings"] != experiment["settings"]
            or historical["classes"] != FIVE_CLASSES
            or historical["freeze"] != experiment["freeze"]
            or historical["starting_weights"]["sha256"] != experiment["checkpoint_sha256"]
            or any(
                historical[k] != experiment[k] for k in ("protocol_id", "subsets_id", "policy_id")
            )
        ):
            raise ValueError("Historical training protocol/settings differ")
    branches = {}
    for branch in ("raw", "adaptive"):
        native = root / branch
        history = pd.read_csv(native / "results.csv")
        args = yaml.safe_load((native / "args.yaml").read_text())
        if (
            args.get("data") != str(root / f"{branch}.yaml")
            or args.get("freeze") != experiment["freeze"]
            or any(args.get(k) != v for k, v in experiment["settings"].items())
        ):
            raise ValueError("Native training args differ from paired settings")
        checkpoints = {}
        for name, epoch in (
            ("best", completed[branch]["best_epoch"]),
            ("last", completed[branch]["final_epoch"]),
        ):
            path = native / f"weights/{name}.pt"
            model = YOLO(str(path))
            ckpt = model.ckpt
            saved = ckpt["train_args"]
            # Native checkpoint serialization can represent an empty augmentation list as None.
            if saved.get("augmentations") is None and experiment["settings"]["augmentations"] == []:
                saved = {**saved, "augmentations": []}
            if (
                model.names != dict(enumerate(FIVE_CLASSES))
                or backbone_boundary(model.model) != experiment["freeze"]
                or ckpt["version"] != version("ultralytics")
                or saved.get("data") != args["data"]
                or saved.get("freeze") != experiment["freeze"]
                or any(saved.get(k) != v for k, v in experiment["settings"].items())
            ):
                raise ValueError("Native checkpoint head/version/settings differ")
            expected = history.loc[history.epoch == epoch].iloc[0]
            for metric in ("metrics/mAP50-95(B)", "metrics/mAP50(B)"):
                if not np.isclose(
                    ckpt["train_metrics"][metric], expected[metric], atol=1e-5, rtol=0
                ):
                    raise ValueError("Checkpoint metrics differ from native history")
            checkpoints[name] = {"path": str(path), "sha256": file_hash(path), "epoch": epoch}
        branches[branch] = {
            **completed[branch],
            "args_sha256": file_hash(native / "args.yaml"),
            "history_sha256": file_hash(native / "results.csv"),
            "checkpoints": checkpoints,
            "interruption": historical["branches"][branch]["interruption"] if historical else None,
        }
    result = {
        "version": "round2-training-handoff-v1",
        "status": "verified_historical_reuse" if historical else "completed_native_training",
        "preparation_id": preparation["preparation_id"],
        "pair_sha256": file_hash(root / "pair.json"),
        "protocol_id": experiment["protocol_id"],
        "subsets_id": experiment["subsets_id"],
        "policy_id": experiment["policy_id"],
        "classes": FIVE_CLASSES,
        "initialization_sha256": experiment["checkpoint_sha256"],
        "freeze": experiment["freeze"],
        "settings": experiment["settings"],
        "branches": branches,
        "historical_evidence_sha256": digest(historical) if historical else None,
        "historical_runtime": historical["runtime"] if historical else None,
        "timing_scope": (
            "Native CSV training/validation sessions; excludes setup, downtime, "
            "partial epochs and final validation."
        ),
        "memory_peak": "not recorded by historical native runs",
        "final_test_accessed": False,
        "next_issue": 35,
        "s20_run": False,
        "v3_released": False,
    }
    result["handoff_id"] = digest(result)
    return result
