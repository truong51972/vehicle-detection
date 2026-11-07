"""Issue #51's fixed S20 resolution pair; native training and explicit acquisition gates."""

import importlib.metadata as md
import os
import shutil
import subprocess
import time
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .common import digest, file_hash, freeze_json, read_image, read_json

STUDY = "issue47-s20-resolution-20e"
PLAN = "docs/ISSUE47_FULL_TRAINING_PLAN.md"
NOTEBOOK = "notebooks/research/issue47/03_full_resolution_training.ipynb"
MODULE = "src/vehicle_detection/full_resolution_training.py"
RECEIPT = "docs/issue51/implementation.json"
RESAMPLES = 1000
MATRIX = ((640, 640), (960, 960), (640, 960), (960, 640))


def check_recovery(trainer, batch, started, seconds):
    """Fail before another batch can turn native recovery into an unmatched arm."""
    if (
        trainer.batch_size != batch
        or trainer.args.resume
        or trainer.nan_recovery_attempts
        or getattr(trainer, "_oom_retries", 0)
    ):
        raise ValueError("Recovery/resume or batch reduction is forbidden")
    if time.perf_counter() - started > seconds:
        raise TimeoutError("Training time budget exhausted; preserve this attempt")
    if trainer.device.type == "cuda":
        import torch

        if torch.cuda.max_memory_allocated() > 14 * 1024**3:
            raise MemoryError("Allocated VRAM exceeds 14 GiB; do not reduce the batch")
    if trainer.loss is not None and not bool(trainer.loss.isfinite().all()):
        raise ValueError("Nonfinite loss; native NaN recovery is forbidden")


def install_selection(model, train_ids, val_ids, train_size, directory):
    """Configure native 640 selection once; observe every real tensor and native save."""
    import torch
    from ultralytics.data import YOLODataset, build_dataloader

    from .baseline import VAL_SETTINGS
    from .resolution_training import check_mode

    directory = Path(directory)
    observed = {"train_shapes": set(), "passes": [], "best": None}
    started = time.perf_counter()
    seconds = {640: 90 * 60, 960: 150 * 60}[train_size]
    state = {}

    def guard(trainer):
        check_recovery(trainer, state["batch"], started, seconds)

    def configure(trainer):
        if trainer.args.imgsz != train_size or trainer.amp:
            raise ValueError("Training size/FP32 contract changed")
        state.update(trainer=trainer, batch=trainer.batch_size)
        guard(trainer)
        if {Path(p).stem for p in trainer.train_loader.dataset.im_files} != set(train_ids):
            raise ValueError("Training loader IDs changed")
        args = copy(trainer.validator.args)
        for key, value in {**VAL_SETTINGS, "imgsz": 640, "batch": 32, "workers": 0}.items():
            setattr(args, key, value)
        args.device = trainer.args.device
        # Native rect validation defaults to pad=.5 (672 pixels); select at actual 640.
        dataset = YOLODataset(
            img_path=trainer.data["val"],
            imgsz=640,
            batch_size=32,
            augment=False,
            hyp=args,
            rect=True,
            cache=None,
            stride=int(trainer.stride),
            pad=0.0,
            data=trainer.data,
        )
        loader = build_dataloader(dataset, batch=32, workers=0, shuffle=False, rank=-1)
        trainer.test_loader = loader
        trainer.validator.dataloader = loader
        trainer.validator.args = args
        preprocess = trainer.preprocess_batch

        def train_batch(batch):
            guard(trainer)
            batch = preprocess(batch)
            shape = tuple(batch["img"].shape[-2:])
            if shape != (train_size, train_size):
                raise ValueError("Actual training tensors differ from training resolution")
            observed["train_shapes"].add(shape)
            return batch

        trainer.preprocess_batch = train_batch
        val_preprocess = trainer.validator.preprocess

        def val_batch(batch):
            guard(trainer)
            batch = val_preprocess(batch)
            shape = tuple(batch["img"].shape[-2:])
            if (
                max(shape) != 640
                or any(side % int(trainer.stride) for side in shape)
                or batch["img"].dtype != torch.float32
            ):
                raise ValueError("Actual selection tensors are not fixed 640")
            state["shapes"].add(shape)
            state["ids"].extend(Path(p).stem for p in batch["im_file"])
            return batch

        trainer.validator.preprocess = val_batch
        observed["trainable_parameters"] = [
            name for name, value in trainer.model.named_parameters() if value.requires_grad
        ]

    def val_start(validator):
        dataset = validator.dataloader.dataset
        if (
            validator.args.imgsz != 640
            or validator.args.batch != 32
            or not validator.args.rect
            or dataset.imgsz != 640
            or dataset.pad != 0
            or not dataset.rect
            or {Path(p).stem for p in dataset.im_files} != set(val_ids)
            or any(
                getattr(validator.args, k) != value
                for k, value in {
                    "conf": 0.001,
                    "iou": 0.7,
                    "max_det": 300,
                    "nms": False,
                    "agnostic_nms": False,
                    "augment": False,
                    "workers": 0,
                }.items()
            )
        ):
            raise ValueError("Selection args or loader differ from fixed-640 development")
        state.update(shapes=set(), ids=[])

    def val_end(validator):
        check_mode(validator)
        if sorted(state["ids"]) != sorted(val_ids):
            raise ValueError("Actual selection pass IDs changed")
        metrics = validator.metrics.results_dict
        if not all(np.isfinite(value) for value in metrics.values()):
            raise ValueError("Nonfinite fitness; native NaN recovery is forbidden")
        record = {
            "epoch": state["trainer"].epoch + 1,
            "terminal": not validator.training,
            "imgsz": 640,
            "pad": 0.0,
            "tensor_shapes": sorted(state["shapes"]),
            "images": len(state["ids"]),
            "metrics": metrics,
        }
        observed["passes"].append(record)
        freeze_json(directory / f"pass{len(observed['passes']):02}.json", record)

    def saved(trainer):
        checkpoint = torch.load(trainer.best, map_location="cpu", weights_only=False)
        record = {
            "selected_epoch": int(checkpoint["epoch"]) + 1,
            "train_metrics": checkpoint["train_metrics"],
            "best_fitness": checkpoint["best_fitness"],
        }
        observed["best"] = record
        freeze_json(directory / f"epoch{trainer.epoch + 1:02}.json", record)

    model.add_callback("on_pretrain_routine_end", configure)
    for event in ("on_train_batch_start", "on_train_batch_end", "on_train_epoch_end"):
        model.add_callback(event, guard)
    model.add_callback("on_val_start", val_start)
    model.add_callback("on_val_end", val_end)
    model.add_callback("on_model_save", saved)
    return observed


def selection_provenance(history, checkpoint, observed):
    """Use the native pre-strip epoch receipt, including ties, then verify saved metrics."""
    best = observed["best"]
    epoch = best["selected_epoch"]
    row = history.loc[history.epoch == epoch]
    if len(row) != 1:
        raise ValueError("Selected epoch is missing from native history")
    for key, value in best["train_metrics"].items():
        if key in history and not np.isclose(row.iloc[0][key], value, rtol=5e-6, atol=5e-6):
            raise ValueError("Native best checkpoint metrics disagree with selected history row")
    if checkpoint["train_metrics"] != best["train_metrics"]:
        raise ValueError("Stripped best checkpoint differs from native selection receipt")
    passes = observed["passes"]
    if (
        len(passes) != len(history) + 1
        or any(p["terminal"] for p in passes[:-1])
        or not passes[-1]["terminal"]
        or [p["epoch"] for p in passes[:-1]] != history.epoch.tolist()
    ):
        raise ValueError("Expected every fixed-640 selection pass and one terminal validation")
    # Native training validation returns five-decimal fitness before best.pt selection.
    fitness = [round(p["metrics"]["fitness"], 5) for p in passes[:-1]]
    if not np.isclose(fitness[epoch - 1], max(fitness), rtol=0, atol=1e-12):
        raise ValueError("Native best selection did not maximize observed fitness")
    return best


def require_new(path):
    path = Path(path).absolute()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Attempt exists; never overwrite/retry: {path}")
    if path.resolve() != path:
        raise ValueError("Unexpected output symlink")


def read_report(root):
    """Hash-verified defaults read only the frozen implementation and compact evidence."""
    root = Path(root).resolve()
    receipt = read_json(root / RECEIPT)
    if set(receipt["inputs"]) != {PLAN, MODULE, NOTEBOOK}:
        raise ValueError("Unexpected implementation receipt")
    for name, expected in receipt["inputs"].items():
        if file_hash(root / name) != expected:
            raise ValueError(f"Frozen implementation changed: {name}")
    report = root / f"docs/{STUDY}"
    if not (report / "evidence.json").exists():
        return {"status": "incomplete" if report.exists() else "not acquired", "issue47_open": True}
    evidence = read_json(report / "evidence.json")
    if "result.json" not in evidence:
        raise ValueError("Missing result hash")
    for name, expected in evidence.items():
        if Path(name).name != name or file_hash(report / name) != expected:
            raise ValueError("Full-resolution evidence changed")
    return read_json(report / "result.json")


def frozen_inputs(root):
    read_report(root)
    if read_json(root / RECEIPT).get("verified") is not True:
        raise ValueError("Synthetic safeguard verification is required before acquisition")
    paths = [
        PLAN,
        MODULE,
        NOTEBOOK,
        RECEIPT,
        "configs",
        "src",
        "tests",
        "uv.lock",
        "pyproject.toml",
    ]
    names = subprocess.check_output(
        ["git", "ls-files", "--", *paths], cwd=root, text=True
    ).splitlines()
    if not {PLAN, MODULE, NOTEBOOK, RECEIPT, "tests/test_full_resolution_training.py"} <= set(
        names
    ):
        raise ValueError("Commit protocol, implementation and verification before acquisition")
    changed = subprocess.check_output(
        ["git", "diff", "HEAD", "--name-only", "--", *paths], cwd=root, text=True
    )
    if changed:
        raise ValueError("Commit protocol and implementation changes before acquisition")
    if md.version("ultralytics") != "8.4.198":
        raise ValueError("Pinned Ultralytics 8.4.198 is required")
    return {name: file_hash(root / name) for name in names}


def historical_inventory(root):
    """Hash tracked evidence; stat historical runs/weights without opening their pixels."""
    names = subprocess.check_output(["git", "ls-files"], cwd=root, text=True).splitlines()
    tracked = {name: file_hash(root / name) for name in names}
    local = {}
    for directory in (root / "runs", root / "weights"):
        for base, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if Path(base) / d != root / f"runs/{STUDY}"]
            for name in files:
                path = Path(base) / name
                stat = path.stat()
                local[str(path.relative_to(root))] = [stat.st_size, stat.st_mtime_ns]
    return {"tracked": tracked, "local": local}


def check_preservation(root):
    before = read_json(root / f"runs/{STUDY}/preservation.json")
    if before != historical_inventory(root):
        raise ValueError("Historical file hashes or size/mtime inventory changed")
    return {"tracked_hashes": len(before["tracked"]), "local_size_mtimes": len(before["local"])}


def prepare(root, *, allow_preparation=False):
    if not allow_preparation:
        return None
    from .datasets import verify_label
    from .experiment_data import populations, source_rows, verify_data
    from .preprocess import quality
    from .resolution_training import check_populations

    root = Path(root).resolve()
    output, report = root / f"runs/{STUDY}", root / f"docs/{STUDY}"
    require_new(output)
    require_new(report)
    inputs = frozen_inputs(root)
    if shutil.disk_usage(root).free < 20 * 1024**3:
        raise OSError("Require 20 GiB free disk before preparation")
    preservation = historical_inventory(root)
    shared, rows = source_rows(root)  # Metadata only, including the existing cross-role audit.
    selected = populations(rows, "S20")
    for split, count, groups, ids_hash in (
        ("train", 9781, 5915, "ca35e8665e4b7961379b75bf96c697210f29f0fd17c6a3a172c9de6903bea1bc"),
        ("val", 13975, 8258, "bf2957e6f237c3c2373f9913117933e6de0ae3c7d3bcf5aa700362efa0e71ff3"),
    ):
        rs = selected[split]
        if (len(rs), len({r["group"] for r in rs}), digest(sorted(r["id"] for r in rs))) != (
            count,
            groups,
            ids_hash,
        ):
            raise ValueError("Frozen full S20/development membership changed")
    if (
        shared["splits_sha256"]
        != "b7a6aba1d1f72e1b1e3eceda8e8427d41f44408067fad63ee68479137714cb96"
    ):
        raise ValueError("Frozen split changed")
    check_populations(selected["train"], selected["val"])
    raw = verify_data(root, "v0")  # Read-only historical byte/receipt verification.
    if raw["files_sha256"] != "4e1a5ce060ef4775910f9bb7227b7fced7cb7b4b2afe3f928479b39cb49e43a6":
        raise ValueError("Historical Raw export receipt changed")
    output.mkdir(parents=True)
    report.mkdir(parents=True)
    freeze_json(output / "preservation.json", preservation)
    plan = {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "inputs": inputs,
        "raw_files_sha256": raw["files_sha256"],
        "versions": {p: md.version(p) for p in ("ultralytics", "torch", "numpy", "opencv-python")},
        "bootstrap_draws": RESAMPLES,
        "matrix": MATRIX,
    }
    freeze_json(output / "plan.json", plan)
    freeze_json(report / "plan.json", plan)
    # Preserve a partial attempt on any subsequent preparation failure.
    features, files = {}, []
    for split, rs in selected.items():
        for row in rs:
            hashes = []
            for kind, suffix in (("images", ".jpg"), ("labels", ".txt")):
                source = root / f"runs/v0/data/{kind}/{split}/{row['id']}{suffix}"
                target = output / f"data/{kind}/{split}/{row['id']}{suffix}"
                if source.resolve() != source:
                    raise ValueError("Unexpected historical export symlink")
                target.parent.mkdir(parents=True, exist_ok=True)
                expected = file_hash(source)
                shutil.copyfile(source, target)
                if file_hash(target) != expected:
                    raise ValueError("Copied export bytes changed")
                hashes.append(expected)
            verify_label(output / f"data/labels/{split}/{row['id']}.txt", row)
            image = read_image(output / f"data/images/{split}/{row['id']}.jpg")
            if image.shape != (row["height"], row["width"], 3):
                raise ValueError("Copied export geometry changed")
            if split == "val":
                features[row["id"]] = quality(image)
            files.append((split, row["id"], *hashes))
    freeze_json(output / "exports.json", files)
    freeze_json(output / "rows.json", selected)
    freeze_json(output / "features.json", features)
    data = {
        "path": str(output / "data"),
        "train": "images/train",
        "val": "images/val",
        "names": dict(enumerate(shared["classes"])),
    }
    (output / "data/data.yaml").write_text(yaml.safe_dump(data))
    freeze_json(
        output / "prepared.json",
        {
            "sha256": {
                name: file_hash(output / name)
                for name in (
                    "exports.json",
                    "rows.json",
                    "features.json",
                    "data/data.yaml",
                    "plan.json",
                )
            },
        },
    )
    pd.DataFrame(selected["train"] + selected["val"]).drop(columns="boxes").to_csv(
        report / "membership.csv", index=False
    )
    freeze_json(
        report / "data.json",
        {
            "files_sha256": digest(files),
            "owned_bytes": sum(
                p.stat().st_size for p in (output / "data").rglob("*") if p.is_file()
            ),
            "preservation": check_preservation(root),
        },
    )
    return plan


def prepared_inputs(root):
    inputs = frozen_inputs(root)
    output = root / f"runs/{STUDY}"
    if output.resolve() != output:
        raise ValueError("Unexpected study directory symlink")
    hashes = read_json(output / "prepared.json")["sha256"]
    if set(hashes) != {"exports.json", "rows.json", "features.json", "data/data.yaml", "plan.json"}:
        raise ValueError("Unexpected prepared receipt paths")
    for name, expected in hashes.items():
        if (output / name).resolve() != output / name or file_hash(output / name) != expected:
            raise ValueError("Prepared inputs changed")
    if read_json(output / "plan.json")["inputs"] != inputs:
        raise ValueError("Prepared protocol/code identity changed")
    for split, sample_id, ih, lh in read_json(output / "exports.json"):
        for kind, suffix, expected in (("images", ".jpg", ih), ("labels", ".txt", lh)):
            path = output / f"data/{kind}/{split}/{sample_id}{suffix}"
            if path.resolve() != path or file_hash(path) != expected:
                raise ValueError("Owned image/label bytes changed")
    check_preservation(root)
    return read_json(output / "rows.json")


def train(root, *, allow_training=False):
    if not allow_training:
        return None
    import torch
    from ultralytics import YOLO

    from .baseline import PRETRAINED_SHA256, verify_coco_names
    from .finetuning import backbone_boundary
    from .training_settings import TRAIN_SETTINGS, validate_training_args

    root = Path(root).resolve()
    output, report = root / f"runs/{STUDY}", root / f"docs/{STUDY}"
    require_new(output / "training-attempt.json")
    for size in (640, 960):
        require_new(output / f"train{size}")
    selected = prepared_inputs(root)
    checkpoint = root / "weights/yolo26n.pt"
    if file_hash(checkpoint) != PRETRAINED_SHA256 or not torch.cuda.is_available():
        raise ValueError("Pinned pretrained checkpoint and CUDA required; no substitute")
    freeze_json(
        output / "training-attempt.json",
        {
            "gpu": torch.cuda.get_device_name(0),
            "cuda": torch.version.cuda,
            "checkpoint_sha256": PRETRAINED_SHA256,
        },
    )
    settings = {**TRAIN_SETTINGS, "workers": 0}
    args_pair, trainable_pair = [], []
    try:
        for size in (640, 960):
            model = YOLO(str(checkpoint))
            verify_coco_names(model.names)
            observed = install_selection(
                model,
                [r["id"] for r in selected["train"]],
                [r["id"] for r in selected["val"]],
                size,
                output / f"train{size}/selection",
            )
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            measurement = {"status": "incomplete", "imgsz": size}
            try:
                model.train(
                    data=str(output / "data/data.yaml"),
                    freeze=backbone_boundary(model.model),
                    project=str(output),
                    name=f"train{size}",
                    exist_ok=False,
                    **{**settings, "imgsz": size},
                )
                measurement["status"] = "completed"
            finally:
                # Persist cost before any checkpoint/history/report parsing can fail.
                measurement.update(
                    wall_seconds=time.perf_counter() - started,
                    peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
                    peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3,
                )
                freeze_json(report / f"train{size}_measurement.json", measurement)
            directory = output / f"train{size}"
            args = yaml.safe_load((directory / "args.yaml").read_text())
            validate_training_args(args, output / "data/data.yaml", {**settings, "imgsz": size})
            args_pair.append(
                {k: v for k, v in args.items() if k not in ("imgsz", "name", "save_dir")}
            )
            trainable_pair.append(observed["trainable_parameters"])
            history = pd.read_csv(directory / "results.csv")
            if history.epoch.tolist() != list(range(1, 21)):
                raise ValueError("Incomplete 20-epoch budget")
            best = torch.load(directory / "weights/best.pt", map_location="cpu", weights_only=False)
            provenance = selection_provenance(history, best, observed)
            freeze_json(
                report / f"train{size}_checkpoint.json",
                {
                    **provenance,
                    "selection_imgsz": 640,
                    "selection_pad": 0.0,
                    "checkpoint_hashes": {
                        n: file_hash(directory / f"weights/{n}.pt") for n in ("best", "last")
                    },
                    "train_shapes": sorted(observed["train_shapes"]),
                    "selection_passes": observed["passes"],
                    "trainable_parameters": observed["trainable_parameters"],
                },
            )
            shutil.copyfile(directory / "results.csv", report / f"train{size}_results.csv")
            shutil.copyfile(directory / "args.yaml", report / f"train{size}_args.yaml")
            del model, best
        if args_pair[0] != args_pair[1] or trainable_pair[0] != trainable_pair[1]:
            raise ValueError("Arm settings or frozen parameters differ beyond resolution")
        freeze_json(
            output / "trained.json",
            {str(s): file_hash(report / f"train{s}_checkpoint.json") for s in (640, 960)},
        )
    except Exception as error:
        freeze_json(
            report / "training_failure.json", {"type": type(error).__name__, "error": str(error)}
        )
        raise
    finally:
        check_preservation(root)
    return {str(s): read_json(report / f"train{s}_checkpoint.json") for s in (640, 960)}


def assess(result):
    """Preserve failed gates and distinguish uncertain/contradictory evidence from a pass."""
    from .five_class import FIVE_CLASSES

    before, after = (result["metrics"][key] for key in ("640/960", "960/960"))
    interval = result["paired_interval"]
    failed, inconclusive = [], []
    if after["all"]["mAP50_95"] - before["all"]["mAP50_95"] < 0.002:
        failed.append("Incremental primary gain below 0.2 pp")
    if interval["estimand"]["class_ids"] != list(range(5)) or interval["recovery_ci_low"] <= 0:
        inconclusive.append("Positive five-class primary interval is not established")
    gains = [
        after["all"]["classes"][c]["AP50_95"] - before["all"]["classes"][c]["AP50_95"]
        for c in ("motorcycle", "bicycle")
    ]
    if min(gains) < 0 or np.mean(gains) < 0.005:
        failed.append("Two-wheeler point-estimate criterion failed")
    for cohort in ("all", "dark", "bright"):
        for name in FIVE_CLASSES:
            a, b = (x[cohort]["classes"].get(name) for x in (before, after))
            if a is None or b is None or b["source_groups"] < 5:
                inconclusive.append(f"Sparse or absent class support: {cohort}/{name}")
                continue
            if b["AP50_95"] - a["AP50_95"] < (-0.02 if cohort == "all" else -0.01):
                failed.append(f"Supported class regression: {cohort}/{name}")
            if (cohort == "all" and name in ("motorcycle", "bicycle")) or (
                cohort == "dark" and name == "bus"
            ):
                ca, cb = (x[cohort]["counts"][name] for x in (before, after))
                if b["AP50_95"] > a["AP50_95"] and cb["FN50"] > ca["FN50"]:
                    inconclusive.append(
                        f"AP vs fixed-confidence TP/FN contradiction: {cohort}/{name}"
                    )
    for entry in interval["per_class"]:
        if entry["class"] in ("motorcycle", "bicycle") and (
            entry["ci_low"] is None or entry["ci_low"] <= 0 <= entry["ci_high"]
        ):
            inconclusive.append(f"Individual two-wheeler gain uncertain: {entry['class']}")
    training, timings = result["training"], result["timings"]
    if training["960"]["wall_seconds"] > 3 * training["640"]["wall_seconds"]:
        failed.append("Training wall time exceeds 3x")
    if max(t["peak_allocated_gib"] for t in training.values()) > 14:
        failed.append("Allocated VRAM exceeds 14 GiB")
    if timings["960/960"]["native_ms_per_image"]["inference"] > (
        1.5 * timings["640/960"]["native_ms_per_image"]["inference"]
    ):
        failed.append("Matched 960 inference cost exceeds 1.5x")
    return {
        "status": "negative"
        if failed
        else "inconclusive"
        if inconclusive
        else "follow-up warranted",
        "failed_criteria": failed,
        "inconclusive": inconclusive,
        "policy_selected": False,
        "independent_confirmation": False,
        "issue47_open": True,
    }


def evaluate(root, *, allow_inference=False):
    if not allow_inference:
        return None
    import torch
    from ultralytics import YOLO

    from .baseline import VAL_SETTINGS
    from .evaluation import ExperimentValidator
    from .evaluation_stats import load_stats
    from .five_class import FIVE_CLASSES
    from .method_discovery import cohorts, outcome
    from .resolution_training import check_mode

    root = Path(root).resolve()
    output, report = root / f"runs/{STUDY}", root / f"docs/{STUDY}"
    require_new(output / "evaluation-attempt.json")
    for a, b in MATRIX:
        require_new(output / f"eval{a}_{b}")
    selected = prepared_inputs(root)
    for size, expected in read_json(output / "trained.json").items():
        if file_hash(report / f"train{size}_checkpoint.json") != expected:
            raise ValueError("Frozen checkpoint receipt changed")
    if not torch.cuda.is_available():
        raise ValueError("CUDA required; do not substitute evaluation hardware")
    settings = {**VAL_SETTINGS, "workers": 0, "half": False}
    freeze_json(output / "evaluation-attempt.json", {"matrix": MATRIX, "settings": settings})
    features = read_json(output / "features.json")
    groups = {
        k: v
        for k, v in cohorts(selected["val"], features).items()
        if k in ("all", "dark", "bright", "small_two_wheeler")
    }
    ids = {r["id"] for r in selected["val"]}
    metrics, timings, statistics = {}, {}, {}
    try:
        for train_size, size in MATRIX:
            key = f"{train_size}/{size}"
            checkpoint = output / f"train{train_size}/weights/best.pt"
            expected = read_json(report / f"train{train_size}_checkpoint.json")[
                "checkpoint_hashes"
            ]["best"]
            if file_hash(checkpoint) != expected:
                raise ValueError("Frozen best checkpoint changed")
            model = YOLO(str(checkpoint))
            if model.names != dict(enumerate(FIVE_CLASSES)):
                raise ValueError("Trained taxonomy changed")
            batches, shapes = [], set()
            clock = {"start": 0}

            def val_start(validator, size=size):
                dataset = validator.dataloader.dataset
                if (
                    len(dataset.im_files) != len(ids)
                    or {Path(p).stem for p in dataset.im_files} != ids
                    or dataset.imgsz != size
                    or dataset.pad != 0.5
                ):
                    raise ValueError("Matrix loader resolution/padding/IDs changed")

            def batch_start(validator, clock=clock):
                torch.cuda.synchronize()
                clock["start"] = time.perf_counter()

            def batch_end(validator, clock=clock, batches=batches):
                torch.cuda.synchronize()
                batches.append((time.perf_counter() - clock["start"]) * 1000)

            model.add_callback("on_val_start", val_start)
            model.add_callback("on_val_end", check_mode)
            model.add_callback("on_val_batch_start", batch_start)
            model.add_callback("on_val_batch_end", batch_end)

            # Reuse the native validator and retain observed tensor shapes alongside matches.
            def observe_shape(validator, shapes=shapes):
                preprocess = validator.preprocess

                def observed_batch(batch):
                    result = preprocess(batch)
                    shapes.add(tuple(result["img"].shape[-2:]))
                    return result

                validator.preprocess = observed_batch

            model.add_callback("on_val_start", observe_shape)
            started = time.perf_counter()
            native = model.val(
                validator=ExperimentValidator,
                data=str(output / "data/data.yaml"),
                split="val",
                project=str(output),
                name=f"eval{train_size}_{size}",
                exist_ok=False,
                **{**settings, "imgsz": size},
            )
            measurement = {
                "wall_seconds": time.perf_counter() - started,
                "native_ms_per_image": native.speed,
                "batch_mean_ms": float(np.mean(batches)),
                "batch_p95_ms": float(np.quantile(batches, 0.95)),
                "batches": len(batches),
                "tensor_shapes": sorted(shapes),
                "checkpoint_sha256": expected,
            }
            freeze_json(report / f"eval{train_size}_{size}_measurement.json", measurement)
            if file_hash(checkpoint) != expected:
                raise ValueError("Evaluation modified a frozen checkpoint")
            path = output / f"eval{train_size}_{size}/paired_stats.npz"
            stats = load_stats(path)
            if set(stats) != ids:
                raise ValueError("Matrix saved matching IDs changed")
            metrics[key] = {name: outcome(stats, rs) for name, rs in groups.items()}
            for cohort, rs in groups.items():
                for cls, name in enumerate(FIVE_CLASSES):
                    if name in metrics[key][cohort]["classes"]:
                        metrics[key][cohort]["classes"][name]["images"] = sum(
                            any(b[0] == cls for b in row["boxes"]) for row in rs
                        )
            if not np.allclose(
                [native.box.map, native.box.map50],
                [metrics[key]["all"]["mAP50_95"], metrics[key]["all"]["mAP50"]],
                atol=1e-8,
                rtol=0,
            ):
                raise ValueError("Native AP disagrees with saved matches")
            timings[key] = measurement
            statistics[key] = {"path": str(path.relative_to(root)), "sha256": file_hash(path)}
            freeze_json(report / f"eval{train_size}_{size}_metrics.json", metrics[key])
            del model, stats
        result = summarize(root, metrics, timings, statistics, selected["val"], groups)
        freeze_json(report / "preservation.json", check_preservation(root))
        freeze_json(report / "result.json", result)
        plot_results(report, result)
        freeze_json(
            report / "evidence.json",
            {p.name: file_hash(p) for p in sorted(report.iterdir()) if p.is_file()},
        )
        return result
    except Exception as error:
        freeze_json(
            report / "evaluation_failure.json", {"type": type(error).__name__, "error": str(error)}
        )
        raise
    finally:
        check_preservation(root)


def summarize(root, metrics, timings, statistics, rows, groups):
    from .evaluation_stats import load_stats, paired_interval
    from .five_class import FIVE_CLASSES
    from .resolution_training import examples

    before, after = [load_stats(root / statistics[k]["path"]) for k in ("640/960", "960/960")]
    intervals = {}
    for name in ("all", "dark", "bright"):
        rs = [{**r, "group": r["source_group"]} for r in groups[name]]
        ids = {r["id"] for r in rs}
        if any(c["source_groups"] >= 5 for c in metrics["640/960"][name]["classes"].values()):
            intervals[name] = paired_interval(
                {k: v for k, v in before.items() if k in ids},
                {k: v for k, v in after.items() if k in ids},
                {"rows": rs},
                resamples=RESAMPLES,
                seed=2026,
                class_names=FIVE_CLASSES,
            )
        else:
            intervals[name] = None
    if intervals["all"] is None:
        raise ValueError("Primary interval lacks support")
    values = {k: v["all"]["mAP50_95"] for k, v in metrics.items()}
    a, b, c, d = (values[k] for k in ("640/640", "640/960", "960/640", "960/960"))
    report = root / f"docs/{STUDY}"
    result = {
        "status": "completed",
        "population": "Full S20; reused Raw development; single seed",
        "metrics": metrics,
        "timings": timings,
        "statistics": statistics,
        "training": {str(s): read_json(report / f"train{s}_measurement.json") for s in (640, 960)},
        "contrasts": {
            "D-B": d - b,
            "C-A": c - a,
            "B-A": b - a,
            "D-C": d - c,
            "interaction": (d - b) - (c - a),
        },
        "paired_interval": intervals["all"],
        "cohort_intervals": intervals,
        "examples": examples(before, after),
        "development_images": len(rows),
        "confirmation_final_accessed": False,
    }
    result["decision"] = assess(result)
    return result


def plot_results(report, result):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    for size in (640, 960):
        history = pd.read_csv(report / f"train{size}_results.csv")
        for ax, x in zip(axes[:2], ("epoch", "time"), strict=True):
            ax.plot(history[x], 100 * history["metrics/mAP50-95(B)"], label=f"Train {size}")
            ax.set(xlabel=x, ylabel="Selection mAP50:95 (%)", title="Fixed 640, pad 0")
        axes[2].plot(
            [640, 960],
            [100 * result["metrics"][f"{size}/{s}"]["all"]["mAP50_95"] for s in (640, 960)],
            marker="o",
            label=f"Train {size}",
        )
    axes[2].set(xlabel="Inference imgsz", ylabel="Development mAP50:95 (%)", xticks=[640, 960])
    for ax in axes:
        ax.legend()
    fig.savefig(report / "convergence_matrix.png", dpi=140)
    plt.close(fig)
