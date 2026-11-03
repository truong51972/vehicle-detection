"""Paired YOLO26n training on frozen issue #6 exports; development validation only."""

import hashlib
import importlib.metadata as md
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import torch_safe_load

from .adaptive import validate_policy
from .baseline import PRETRAINED_SHA256, VAL_SETTINGS, verify_coco_names
from .common import digest, file_hash, read_json, write_json
from .data import CLASSES
from .offline import verify_alignment
from .subsets import validate_subsets

TRAIN_SETTINGS = {
    **VAL_SETTINGS,
    "epochs": 20,
    "patience": 0,  # Fixed budgets; best-epoch selection is reported separately.
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


def _freeze_json(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError("Frozen training identity changed; use a new experiment root")
    else:
        write_json(path, value)


def backbone_boundary(model):
    """Derive the boundary from the loaded architecture, never another YOLO generation."""
    if model.yaml.get("yaml_file") != "yolo26n.yaml" or not model.yaml.get("end2end"):
        raise ValueError("Expected the installed end-to-end YOLO26n architecture")
    boundary = len(model.yaml["backbone"])
    if not 0 < boundary < len(model.model) or type(model.model[-1]).__name__ != "Detect":
        raise ValueError("Unsupported backbone/detection layout")
    return boundary


def _verify_role(directory, records, split, variant, recipes=None):
    """Reject wrong roles before decoding pixels, then verify owned files and frozen boxes."""
    directory = Path(directory).resolve()
    meta = pd.read_csv(directory / "metadata.csv")
    selected = {r["id"]: r for r in records}
    if meta.id.duplicated().any() or set(meta.id) != set(selected) or set(meta.split) != {split}:
        raise ValueError("Export must contain exactly the frozen role IDs, with no duplicates")
    export = read_json(directory / "export.json")
    if (
        export["variant"] != variant
        or export["encoding"]["format"] != "jpg"
        or export["encoding"]["jpeg_quality"] != 95
        or (variant == "raw_reencoded" and export["recipe"] != {})
        or (recipes is not None and export["recipe"] != recipes)
    ):
        raise ValueError("Export treatment/encoding differs from the frozen policy")
    for row in meta.to_dict("records"):
        source = selected[row["id"]]
        image = directory / "images" / split / f"{row['id']}.jpg"
        label = directory / "labels" / split / f"{row['id']}.txt"
        if (
            Path(row["image"]) != image
            or image.resolve() != image
            or label.resolve() != label
            or row["source_sha256"] != source["sha256"]
            or (row["width"], row["height"]) != (source["width"], source["height"])
        ):
            raise ValueError(
                "Export path, source hash or geometry changed; re-export after relocation"
            )
        actual = [list(map(float, line.split())) for line in label.read_text().splitlines()]
        expected = [
            [
                c,
                (x1 + x2) / 2 / source["width"],
                (y1 + y2) / 2 / source["height"],
                (x2 - x1) / source["width"],
                (y2 - y1) / source["height"],
            ]
            for c, x1, y1, x2, y2 in source["boxes"]
        ]
        if len(actual) != len(expected) or any(len(box) != 5 for box in actual):
            raise ValueError("Export labels differ from the frozen boxes")
        # Native labels preserve source order; sorting rounded centers can swap tied boxes.
        if actual and not np.allclose(actual, expected, atol=1e-6, rtol=0):
            raise ValueError("Export labels differ from the frozen boxes")
    verify_alignment([directory], deep=True)
    return {
        "path": str(directory),
        "export": export,
        "files_identity": read_json(directory / "verified.json")["files_identity"],
        "ids": sorted(selected),
    }


def prepare_pair(protocol, subsets, policy, decision, pair, development, checkpoint, output):
    """Freeze an S10 pair before optimizer steps; both branches select on the same Raw dev."""
    validate_subsets(subsets, protocol)
    validate_policy(policy, protocol)
    if (
        decision["decision_id"] != digest({k: v for k, v in decision.items() if k != "decision_id"})
        or decision["status"] != "v3_not_deployed"
        or decision["retained_policy_id"] != policy["policy_id"]
    ):
        raise ValueError("Expected the issue #11 frozen decision retaining the issue #6 policy")
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
        raise ValueError(
            "Expected the frozen S10 pair; expansion needs a separate declared decision"
        )
    ids = set(subsets["subsets"]["S10"]["ids"])
    rows = [r for r in protocol["splits"]["training_pool"] if r["id"] in ids]
    datasets = {
        b: _verify_role(pair / b, rows, "train", v, policy["recipes"] if b == "adaptive" else None)
        for b, v in [("raw", "raw_reencoded"), ("adaptive", "adaptive")]
    }
    verify_alignment([pair / b for b in datasets], deep=False)
    dev = _verify_role(
        development, protocol["splits"]["development_validation"], "tune", "raw_reencoded"
    )
    if file_hash(checkpoint) != PRETRAINED_SHA256:
        raise ValueError("Use the pinned unchanged pretrained YOLO26n checkpoint")
    model = YOLO(str(checkpoint))
    verify_coco_names(model.names)
    identity = {
        "version": "issue7-s10-v1",
        "protocol_id": protocol["manifest_id"],
        "pair": binding,
        "policy_decision_id": decision["decision_id"],
        "training": datasets,
        "development": dev,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": PRETRAINED_SHA256,
        "freeze": backbone_boundary(model.model),
        "settings": TRAIN_SETTINGS,
        "native_defaults_sha256": digest(DEFAULT_CFG_DICT),
        "code_sha256": file_hash(__file__),
        "versions": {p: md.version(p) for p in ("ultralytics", "torch", "torchvision", "numpy")},
        "hardware": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda_runtime": torch.version.cuda,
        },
        "initialization": "independent-from-pretrained; semantic COCO-to-six-class head transfer",
        "final_test_accessed": False,
    }
    identity["experiment_id"] = digest(identity)
    if not (output / "pair.json").exists() and output.exists() and any(output.iterdir()):
        raise ValueError("Unknown training directory; choose an empty experiment root")
    _freeze_json(output / "pair.json", identity)
    for branch, dataset in datasets.items():
        config = {
            "path": str(output),
            "train": str(Path(dataset["path"]) / "images/train"),
            "val": str(development / "images/tune"),
            "names": dict(enumerate(CLASSES)),
        }
        path = output / f"{branch}.yaml"
        if path.exists() and yaml.safe_load(path.read_text()) != config:
            raise ValueError("Training YAML changed; use a new experiment root")
        path.write_text(yaml.safe_dump(config))
    return identity


def _state_hash(state):
    h = hashlib.sha256()
    for name, value in state.items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class FreezeAudit:
    """Check all frozen parameters/buffers and actual neck/head gradients and updates."""

    def __init__(self, model, boundary):
        self.boundary = boundary
        self.before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        self.frozen = tuple(f"model.{i}." for i in range(boundary))
        self.report = {
            "backbone_modules": boundary,
            "initial_state_sha256": _state_hash(self.before),
            "parameters": [
                {"name": k, "count": v.numel(), "trainable": v.requires_grad}
                for k, v in model.named_parameters()
            ],
            "modules": [
                {"name": f"model.{i}", "type": type(m).__name__, "frozen": i < boundary}
                for i, m in enumerate(model.model)
            ],
            "optimizer_steps_checked": 0,
        }
        if any(p.requires_grad for k, p in model.named_parameters() if k.startswith(self.frozen)):
            raise ValueError("Backbone parameters must be frozen")
        if not any(p.requires_grad for p in model.parameters()):
            raise ValueError("No trainable neck/head parameters")
        for trainable in (False, True):
            self.report["trainable_count" if trainable else "frozen_count"] = sum(
                p["count"] for p in self.report["parameters"] if p["trainable"] == trainable
            )

    def check_gradients(self, model):
        nonzero = []
        for name, param in model.named_parameters():
            if name.startswith(self.frozen) and param.grad is not None:
                raise ValueError("Frozen backbone received a gradient")
            if param.grad is not None:
                if not torch.isfinite(param.grad).all():
                    raise ValueError("Nonfinite training gradient")
                if param.grad.count_nonzero():
                    nonzero.append(name)
        detection = f"model.{len(model.model) - 1}."
        if not any(k.startswith(detection) for k in nonzero) or not any(
            not k.startswith((*self.frozen, detection)) for k in nonzero
        ):
            raise ValueError("Expected nonzero gradients in both neck and detection head")
        self.report["nonzero_gradient_parameters"] = nonzero

    def check_weights(self, model, *, updated=False):
        changed = []
        for name, value in model.state_dict().items():
            if not torch.equal(self.before[name], value.detach().cpu()):
                if name.startswith(self.frozen):
                    raise ValueError(f"Frozen backbone weight/buffer changed: {name}")
                changed.append(name)
        if updated:
            trainable = {k for k, v in model.named_parameters() if v.requires_grad}
            changed = sorted(trainable.intersection(changed))
            detection = f"model.{len(model.model) - 1}."
            if not any(k.startswith(detection) for k in changed) or not any(
                not k.startswith(detection) for k in changed
            ):
                raise ValueError("Expected actual weight updates in both neck and detection head")
            self.report["changed_trainable_parameters"] = changed
            self.report["optimizer_steps_checked"] += 1
        return self.report


class PairedTrainer(DetectionTrainer):
    """Native loop, with checks at setup/optimizer steps and measured epoch costs."""

    def _setup_train(self):
        super()._setup_train()
        boundary = backbone_boundary(self.model)
        if self.args.freeze != boundary or self.model.names != dict(enumerate(CLASSES)):
            raise ValueError("Trainable protocol or six-class head changed")
        if self.batch_size != self.run_identity["pair"]["settings"]["batch"]:
            raise ValueError("Native OOM retry changed paired batch size; choose a new protocol")
        for loader, expected in [
            (self.train_loader, self.run_identity["pair"]["training"][self.branch]["ids"]),
            (self.test_loader, self.run_identity["pair"]["development"]["ids"]),
        ]:
            if sorted(Path(p).stem for p in loader.dataset.im_files) != expected:
                raise ValueError("Native loader changed frozen image membership")
        self.audit = FreezeAudit(self.model, boundary)
        self.optimizer_steps = 0
        if not self.resume:
            _freeze_json(
                self.save_dir.parent / "initial_model.json",
                {"initial_state_sha256": self.audit.report["initial_state_sha256"]},
            )
        self.add_callback("on_train_epoch_start", self._epoch_start)
        self.add_callback("on_train_epoch_end", self._training_epoch_end)
        self.add_callback("on_fit_epoch_end", self._epoch_end)
        self.add_callback("on_model_save", self._checkpoint_receipt)
        write_json(self.save_dir / "freeze_audit.json", self.audit.report)

    def optimizer_step(self):
        if self.batch_size != self.run_identity["pair"]["settings"]["batch"]:
            raise ValueError("Native OOM retry changed paired batch size; choose a new protocol")
        first = self.audit.report["optimizer_steps_checked"] == 0
        if first:
            self.audit.check_gradients(self.model)
        updating = any(group["lr"] > 0 for group in self.optimizer.param_groups)
        super().optimizer_step()
        self.optimizer_steps += 1
        if first and updating:
            self.audit.check_weights(self.model, updated=True)
            write_json(self.save_dir / "freeze_audit.json", self.audit.report)

    def _epoch_start(self, trainer):
        self.epoch_started = time.perf_counter()
        self.epoch_start_steps = self.optimizer_steps
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def _training_epoch_end(self, trainer):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.training_seconds = time.perf_counter() - self.epoch_started

    def _epoch_end(self, trainer):
        self.audit.check_weights(self.model)
        if self.epoch >= self.epochs:  # Native final best-checkpoint validation callback.
            return
        row = {
            "epoch": self.epoch + 1,
            "wall_seconds_including_validation": time.perf_counter() - self.epoch_started,
            "training_seconds": self.training_seconds,
            "optimizer_steps": self.optimizer_steps - self.epoch_start_steps,
            "native_resume": bool(self.resume),
            "training_images_per_second": len(self.train_loader.dataset) / self.training_seconds,
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device)
            if self.device.type == "cuda"
            else None,
            "metrics": {k: float(v) for k, v in self.metrics.items()},
        }
        write_json(self.save_dir / f"epoch-{self.epoch + 1:03}.json", row)

    def _checkpoint_receipt(self, trainer):
        receipt = self.save_dir / "resume.json"
        previous = read_json(receipt)["checkpoint"] if receipt.exists() else None
        target = self.wdir / f"resume-{self.epoch:03}.pt"
        temporary = target.with_suffix(".tmp")
        shutil.copyfile(self.last, temporary)
        os.replace(temporary, target)
        write_json(
            receipt,
            {
                "run_id": self.run_identity["run_id"],
                "checkpoint": target.name,
                "sha256": file_hash(target),
            },
        )
        if previous and previous != target.name and Path(previous).name == previous:
            (self.wdir / previous).unlink(missing_ok=True)


def check_resume(checkpoint, identity):
    """Only this run's complete, unstripped native optimizer checkpoint may resume."""
    # shortcut: native EMA/FP16 resume, add exact state saves for bitwise continuation.
    checkpoint = Path(checkpoint)
    receipt = read_json(checkpoint.parent.parent / "resume.json")
    name = receipt["checkpoint"]
    if Path(name).name != name:
        raise ValueError("Unsafe resume checkpoint path")
    checkpoint = checkpoint.parent / name
    if receipt != {
        "run_id": identity["run_id"],
        "checkpoint": name,
        "sha256": file_hash(checkpoint),
    }:
        raise ValueError("Resume checkpoint identity/hash changed")
    saved, _ = torch_safe_load(checkpoint)
    pair = identity["pair"]
    if (
        saved.get("optimizer") is None
        or saved.get("epoch", -1) < 0
        or saved.get("version") != pair["versions"]["ultralytics"]
        or backbone_boundary(saved["ema"]) != pair["freeze"]
        or any(saved["train_args"].get(k) != v for k, v in pair["settings"].items())
        or saved["train_args"].get("freeze") != pair["freeze"]
        or saved["train_args"].get("data") != identity["data"]
    ):
        raise ValueError(
            "Resume requires identical architecture, settings and native optimizer state"
        )
    return checkpoint


def train_branch(experiment, branch, output, *, resume=False):
    """Run one explicitly selected S10 branch. Re-run prepare_pair first to verify inputs."""
    output = Path(output).resolve()
    if branch not in {"raw", "adaptive"} or read_json(output / "pair.json") != experiment:
        raise ValueError("Unknown branch or changed paired experiment")
    if experiment["experiment_id"] != digest(
        {k: v for k, v in experiment.items() if k != "experiment_id"}
    ) or experiment["code_sha256"] != file_hash(__file__):
        raise ValueError("Training implementation/identity changed; prepare a new root")
    if (
        any(md.version(p) != version for p, version in experiment["versions"].items())
        or digest(DEFAULT_CFG_DICT) != experiment["native_defaults_sha256"]
        or experiment["hardware"]
        != {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda_runtime": torch.version.cuda,
        }
    ):
        raise ValueError("Training software/hardware changed; prepare a new experiment")
    for dataset in [*experiment["training"].values(), experiment["development"]]:
        root = Path(dataset["path"])
        if read_json(root / "export.json") != dataset["export"]:
            raise ValueError("Frozen export recipe/encoding changed")
        meta = pd.read_csv(root / "metadata.csv")
        for row in meta.to_dict("records"):
            image = root / "images" / row["split"] / f"{row['id']}.jpg"
            label = root / "labels" / row["split"] / f"{row['id']}.txt"
            if Path(row["image"]) != image or image.resolve() != image or label.resolve() != label:
                raise ValueError("Export paths/symlinks changed")
        verify_alignment([dataset["path"]], deep=True)
        if (
            read_json(Path(dataset["path"]) / "verified.json")["files_identity"]
            != dataset["files_identity"]
        ):
            raise ValueError("Training/development export changed")
    config = output / f"{branch}.yaml"
    expected = {
        "path": str(output),
        "train": str(Path(experiment["training"][branch]["path"]) / "images/train"),
        "val": str(Path(experiment["development"]["path"]) / "images/tune"),
        "names": dict(enumerate(CLASSES)),
    }
    if yaml.safe_load(config.read_text()) != expected:
        raise ValueError("Training YAML changed")
    if file_hash(experiment["checkpoint"]) != PRETRAINED_SHA256:
        raise ValueError("Starting checkpoint changed")
    identity = {"pair": experiment, "branch": branch, "data": str(config)}
    identity["run_id"] = digest(identity)
    run = output / branch
    if run.exists() and any(run.iterdir()) and not (run / "identity.json").exists():
        raise ValueError("Unknown native run directory")
    _freeze_json(run / "identity.json", identity)
    if (run / "complete.json").exists():
        completed = read_json(run / "complete.json")
        if (
            completed["run_id"] != identity["run_id"]
            or completed["checksum"]
            != digest({k: v for k, v in completed.items() if k != "checksum"})
            or any(file_hash(run / name) != sha for name, sha in completed["artifacts"].items())
        ):
            raise ValueError("Completed training artifacts changed")
        return completed
    checkpoint = run / "weights/last.pt"
    if resume:
        checkpoint = check_resume(checkpoint, identity)
    elif (run / "args.yaml").exists():
        raise ValueError("Existing training attempt; resume its checkpoint or choose a new root")
    trainer = PairedTrainer(
        overrides={
            **experiment["settings"],
            "model": str(checkpoint) if resume else experiment["checkpoint"],
            "data": str(config),
            "freeze": experiment["freeze"],
            "project": str(output),
            "name": branch,
            "exist_ok": True,
            "resume": str(checkpoint) if resume else False,
        }
    )
    trainer.run_identity, trainer.branch = identity, branch
    trainer.train()
    report = read_json(run / "freeze_audit.json")
    if not report["optimizer_steps_checked"]:
        raise ValueError("Training finished without a verified optimizer step")
    epochs = len(pd.read_csv(run / "results.csv"))
    resources = sorted(run.glob("epoch-*.json"))
    if epochs != experiment["settings"]["epochs"] or len(resources) != epochs:
        raise ValueError("Training did not complete the declared epoch budget")
    budget = {
        "epochs": epochs,
        "images_per_epoch": len(experiment["training"][branch]["ids"]),
        "optimizer_steps": sum(read_json(p)["optimizer_steps"] for p in resources),
    }
    other = output / ("adaptive" if branch == "raw" else "raw") / "complete.json"
    if other.exists() and read_json(other)["budget"] != budget:
        raise ValueError("Paired runs consumed unequal training budgets")
    completed = {
        "run_id": identity["run_id"],
        "artifacts": {
            name: file_hash(run / name)
            for name in [
                "weights/best.pt",
                "weights/last.pt",
                "results.csv",
                "freeze_audit.json",
                *[p.name for p in resources],
            ]
        },
        "budget": budget,
        "best_epoch": trainer.stopper.best_epoch,
        "final_test_accessed": False,
    }
    completed["checksum"] = digest(completed)
    write_json(run / "complete.json", completed)
    return completed
