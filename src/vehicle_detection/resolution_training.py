"""One fixed paired training-resolution diagnostic; no policy search or S20 promotion."""

import importlib.metadata as md
import subprocess
import time
from pathlib import Path
from shutil import copyfile

import numpy as np
import pandas as pd
import yaml

from .common import digest, file_hash, freeze_json, read_image, read_json
from .datasets import verify_label
from .experiment_data import source_rows
from .five_class import FIVE_CLASSES
from .method_discovery import cohorts, outcome
from .v3_pilot import run as read_pilot

STUDY = "issue47-resolution-training-repeat"


def select_training(rows, minimum=2048, maximum=2176):
    groups = {}
    for row in rows:
        if row["role"] == "training_pool" and row["s20"]:
            groups.setdefault(row["group"], []).append(row)
    selected = []
    for group in sorted(
        groups, key=lambda g: (digest([2026, "issue47-resolution-training", g]), g)
    ):
        selected.extend(groups[group])
        if len(selected) >= minimum:
            break
    if not minimum <= len(selected) <= maximum:
        raise ValueError("Training prefix exceeds fixed budget or lacks images")
    return sorted(selected, key=lambda r: r["id"])


def check_populations(train, evaluation):
    for key in ("id", "group", "source_group", "sha256"):
        if {r[key] for r in train} & {r[key] for r in evaluation}:
            raise ValueError(f"Training/evaluation overlap: {key}")
    for rows in (train, evaluation):
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError("Duplicate population IDs")
        if any(
            len({r["source_group"] for r in rows if any(b[0] == c for b in r["boxes"])}) < 5
            for c in range(5)
        ):
            raise ValueError("Insufficient five-class source-group support")


def assess(result):
    before, after = (result["metrics"][key] for key in ("640/960", "960/960"))
    reasons = []
    if after["all"]["mAP50_95"] - before["all"]["mAP50_95"] < 0.002:
        reasons.append("Incremental primary gain below 0.2 pp")
    interval = result["paired_interval"]
    if interval["estimand"]["class_ids"] != list(range(5)) or interval["recovery_ci_low"] <= 0:
        reasons.append("Five-class incremental interval does not establish positive benefit")
    changes = [
        after["all"]["classes"][c]["AP50_95"] - before["all"]["classes"][c]["AP50_95"]
        for c in ("motorcycle", "bicycle")
    ]
    if min(after["all"]["classes"][c]["source_groups"] for c in ("motorcycle", "bicycle")) < 5:
        reasons.append("Insufficient two-wheeler support")
    if min(changes) < 0 or np.mean(changes) < 0.005:
        reasons.append("Two-wheeler incremental gain insufficient or a class declines")
    for cohort in ("all", "dark", "bright"):
        for name, values in after[cohort]["classes"].items():
            if values["source_groups"] >= 5 and values["AP50_95"] - before[cohort]["classes"][name][
                "AP50_95"
            ] < (-0.02 if cohort == "all" else -0.01):
                reasons.append(f"Supported class regression: {cohort}/{name}")
    training = result["training"]
    if training["960"]["wall_seconds"] > 3 * training["640"]["wall_seconds"]:
        reasons.append("Training wall time exceeds 3x")
    if max(v["peak_allocated_gib"] for v in training.values()) > 14:
        reasons.append("Peak allocated VRAM exceeds 14 GiB")
    if (
        result["timings"]["960/960"]["native_ms_per_image"]["inference"]
        > 1.5 * result["timings"]["640/960"]["native_ms_per_image"]["inference"]
    ):
        reasons.append("Matched 960 inference mean exceeds 1.5x")
    return {
        "useful_training_resolution_signal": not reasons,
        "reasons": reasons,
        "policy_selected": False,
        "independent_confirmation": False,
        "s20_promoted": False,
    }


def examples(before, after):
    records = []
    for c, name in enumerate(FIVE_CLASSES):
        changes = []
        for sample_id in sorted(before):
            counts = []
            for stats in (before, after):
                values = stats[sample_id]
                keep = (values["conf"] >= 0.25) & (values["pred_cls"] == c)
                tp = int(values["tp"][keep, 0].sum())
                counts.append({"TP50": tp, "FP50": int(keep.sum()) - tp})
            if counts[0] != counts[1]:
                changes.append(
                    {"id": sample_id, "class": name, "before": counts[0], "after": counts[1]}
                )
        ordered = sorted(
            changes,
            key=lambda r: (
                r["after"]["TP50"] - r["before"]["TP50"],
                r["before"]["FP50"] - r["after"]["FP50"],
                r["id"],
            ),
        )
        records.extend(ordered[:1] + ordered[-1:] if len(ordered) > 1 else ordered)
    return records


def check_mode(validator):
    if not validator.end2end or validator.args.nms is not False:
        raise ValueError("NMS-free evaluation contract changed")


def run(root, *, allow_training=False):
    root = Path(root).resolve()
    report, output = root / f"docs/{STUDY}", root / f"runs/{STUDY}/study"
    if not allow_training:
        for name, expected in read_json(report / "evidence.json").items():
            if Path(name).name != name or file_hash(report / name) != expected:
                raise ValueError("Training diagnostic evidence changed")
        return read_json(report / "result.json")
    if output.exists() or report.exists():
        raise FileExistsError("Training diagnostic attempt exists; never overwrite or retry")
    if output.resolve() != output or report.resolve() != report:
        raise ValueError("Unexpected diagnostic output symlink")
    protected = [
        "docs/ISSUE47_TRAINING_PLAN.md",
        "docs/ISSUE47_TRAINING_REPEAT_PLAN.md",
        "notebooks/research/issue47/02_resolution_training.ipynb",
        "src/vehicle_detection/resolution_training.py",
        "src/vehicle_detection/experiment_data.py",
        "src/vehicle_detection/finetuning.py",
        "src/vehicle_detection/training_settings.py",
        "src/vehicle_detection/evaluation_stats.py",
        "src/vehicle_detection/evaluation.py",
        "src/vehicle_detection/method_discovery.py",
        "configs/data.yaml",
        "configs/splits.csv.gz",
    ]
    for name in protected:
        if (
            subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=root)
            != (root / name).read_bytes()
        ):
            raise ValueError("Commit plan and implementation before training")
    old = read_pilot(root)
    _, rows = source_rows(root)
    train = select_training(rows)
    ids = set(pd.read_csv(root / "docs/issue47/membership.csv").id)
    evaluation = sorted([r for r in rows if r["id"] in ids], key=lambda r: r["id"])
    if (
        len(evaluation) != 1027
        or digest(evaluation) != read_json(root / "docs/issue47/plan.json")["rows_sha256"]
    ):
        raise ValueError("Consumed evaluation population changed")
    check_populations(train, evaluation)

    import torch
    from ultralytics import YOLO

    from .baseline import PRETRAINED_SHA256, VAL_SETTINGS, verify_coco_names
    from .evaluation import ExperimentValidator
    from .evaluation_stats import load_stats, paired_interval
    from .finetuning import backbone_boundary
    from .training_settings import TRAIN_SETTINGS, validate_training_args

    checkpoint = root / "weights/yolo26n.pt"
    if file_hash(checkpoint) != PRETRAINED_SHA256 or not torch.cuda.is_available():
        raise ValueError("Pinned checkpoint and CUDA are required; no substitute run")
    settings = {**TRAIN_SETTINGS, "epochs": 5, "workers": 0, "val": False}
    native_settings = {**VAL_SETTINGS, "workers": 0, "half": False}
    plan = {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "inputs": {name: file_hash(root / name) for name in protected},
        "checkpoint_sha256": PRETRAINED_SHA256,
        "train_ids_sha256": digest([r["id"] for r in train]),
        "evaluation_ids_sha256": digest([r["id"] for r in evaluation]),
        "train_images": len(train),
        "evaluation_images": len(evaluation),
        "training_settings": settings,
        "evaluation_settings": native_settings,
        "versions": {p: md.version(p) for p in ("torch", "ultralytics", "numpy", "opencv-python")},
        "gpu": torch.cuda.get_device_name(0),
        "cuda": torch.version.cuda,
        "independent_confirmation": False,
    }
    freeze_json(output / "plan.json", plan)
    freeze_json(report / "plan.json", plan)
    pd.DataFrame(train + evaluation).drop(columns="boxes").to_csv(
        report / "membership.csv", index=False
    )
    entries = read_json(root / "runs/issue47/pilot/exports.json")
    if digest(entries) != old["exports_sha256"]:
        raise ValueError("Historical evaluation export receipt changed")
    exports = {Path(n).stem: (ih, lh) for n, ih, lh in entries if n.startswith("skip/")}
    if set(exports) != ids:
        raise ValueError("Historical evaluation export IDs changed")
    dataset, files = output / "data", []
    for split, selected, source_root in (
        ("train", train, root / "runs/v0/data"),
        ("val", evaluation, root / "runs/issue47/pilot/skip"),
    ):
        for row in selected:
            hashes = []
            for kind, suffix in (("images", ".jpg"), ("labels", ".txt")):
                source = source_root / kind / split / f"{row['id']}{suffix}"
                if source.resolve() != source:
                    raise ValueError("Unexpected export symlink")
                expected = file_hash(source)
                target = dataset / kind / split / source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                copyfile(source, target)
                if file_hash(target) != expected:
                    raise ValueError("Copied export bytes changed")
                hashes.append(expected)
            if split == "val" and tuple(hashes) != exports[row["id"]]:
                raise ValueError("Historical evaluation bytes changed")
            verify_label(dataset / "labels" / split / f"{row['id']}.txt", row)
            if read_image(dataset / "images" / split / f"{row['id']}.jpg").shape != (
                row["height"],
                row["width"],
                3,
            ):
                raise ValueError("Copied export geometry changed")
            files.append((split, row["id"], *hashes))
    freeze_json(output / "exports.json", files)
    freeze_json(report / "data.json", {"files_sha256": digest(files), "jpeg_quality": 95})
    data = dataset / "data.yaml"
    data.write_text(
        yaml.safe_dump(
            {
                "path": str(dataset),
                "train": "images/train",
                "val": "images/val",
                "names": dict(enumerate(FIVE_CLASSES)),
            }
        )
    )
    training, checkpoints = {}, {}
    for size in (640, 960):
        model = YOLO(str(checkpoint))
        verify_coco_names(model.names)
        tick = time.perf_counter()

        def stop_at_budget(trainer, started=tick):
            if time.perf_counter() - started > 1800:
                raise TimeoutError("Fixed training time budget exhausted; no retry")
            if trainer.batch_size != 16 or trainer.nan_recovery_attempts:
                raise ValueError("Native recovery changes the fixed budget; stop without retry")

        def check_training_loader(trainer):
            paths = trainer.train_loader.dataset.im_files
            if len(paths) != len(train) or {Path(p).stem for p in paths} != {
                r["id"] for r in train
            }:
                raise ValueError("Training loader IDs changed")

        model.add_callback("on_train_batch_end", stop_at_budget)
        model.add_callback("on_train_batch_start", stop_at_budget)
        model.add_callback("on_pretrain_routine_end", check_training_loader)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model.train(
            data=str(data),
            freeze=backbone_boundary(model.model),
            project=str(output),
            name=f"train{size}",
            exist_ok=False,
            **{**settings, "imgsz": size},
        )
        directory = output / f"train{size}"
        history = pd.read_csv(directory / "results.csv")
        validate_training_args(
            yaml.safe_load((directory / "args.yaml").read_text()), data, {**settings, "imgsz": size}
        )
        if history.epoch.tolist() != list(range(1, 6)):
            raise ValueError("Incomplete five-epoch training budget; no retry")
        checkpoints[size] = directory / "weights/last.pt"
        training[str(size)] = {
            "wall_seconds": time.perf_counter() - tick,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "checkpoint_sha256": file_hash(checkpoints[size]),
            "epochs": 5,
        }
        freeze_json(report / f"train{size}_measurement.json", training[str(size)])
        history.to_csv(report / f"train{size}_results.csv", index=False)
        copyfile(directory / "args.yaml", report / f"train{size}_args.yaml")
        del model
    features = read_json(root / "docs/issue47/features.json")
    metrics, timings, statistics = {}, {}, {}
    batch_times, start = [], 0

    def check_loader(validator):
        paths = validator.dataloader.dataset.im_files
        if len(paths) != len(ids) or {Path(p).stem for p in paths} != ids:
            raise ValueError("Evaluation loader IDs changed")

    def start_batch(validator):
        nonlocal start
        torch.cuda.synchronize()
        start = time.perf_counter()

    def end_batch(validator):
        torch.cuda.synchronize()
        batch_times.append((time.perf_counter() - start) * 1000)

    for train_size in (640, 960):
        for size in (640, 960):
            key = f"{train_size}/{size}"
            model = YOLO(str(checkpoints[train_size]))
            if model.names != dict(enumerate(FIVE_CLASSES)):
                raise ValueError("Trained detector taxonomy changed")
            batch_times.clear()
            model.add_callback("on_val_start", check_loader)
            model.add_callback("on_val_end", check_mode)
            model.add_callback("on_val_batch_start", start_batch)
            model.add_callback("on_val_batch_end", end_batch)
            tick = time.perf_counter()
            native = model.val(
                validator=ExperimentValidator,
                data=str(data),
                split="val",
                project=str(output),
                name=f"eval{train_size}_{size}",
                exist_ok=False,
                **{**native_settings, "imgsz": size},
            )
            measurement = {
                "native_ms_per_image": native.speed,
                "wall_seconds": time.perf_counter() - tick,
                "batch_mean_ms": float(np.mean(batch_times)),
                "batch_p95_ms": float(np.quantile(batch_times, 0.95)),
                "batches": len(batch_times),
            }
            freeze_json(report / f"eval{train_size}_{size}_measurement.json", measurement)
            path = output / f"eval{train_size}_{size}/paired_stats.npz"
            stats = load_stats(path)
            if set(stats) != ids:
                raise ValueError("Paired statistics IDs changed")
            metrics[key] = {
                name: outcome(stats, rs) for name, rs in cohorts(evaluation, features).items()
            }
            if not np.allclose(
                [native.box.map, native.box.map50],
                [metrics[key]["all"]["mAP50_95"], metrics[key]["all"]["mAP50"]],
                atol=1e-8,
                rtol=0,
            ):
                raise ValueError("Native AP disagrees with saved matches")
            timings[key] = measurement
            statistics[key] = {"path": str(path.relative_to(root)), "sha256": file_hash(path)}
            del model
    bootstrap_rows = [{**r, "group": r["source_group"]} for r in evaluation]
    before, after = [load_stats(root / statistics[k]["path"]) for k in ("640/960", "960/960")]
    result = {
        "population": "Consumed calibration exploration; short training diagnostic only",
        "training": training,
        "metrics": metrics,
        "timings": timings,
        "statistics": statistics,
        "examples": examples(before, after),
        "paired_interval": paired_interval(
            before, after, {"rows": bootstrap_rows}, class_names=FIVE_CLASSES
        ),
        "confirmation_development_final_pixels_accessed": False,
    }
    result["decision"] = assess(result)
    freeze_json(report / "result.json", result)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for size in (640, 960):
        axes[0].plot(
            [640, 960],
            [100 * metrics[f"{size}/{s}"]["all"]["mAP50_95"] for s in (640, 960)],
            marker="o",
            label=f"Train {size}",
        )
    axes[0].set(
        xlabel="Inference size",
        ylabel="Five-class mAP50:95 (%)",
        title="Consumed exploration; five training epochs",
        xticks=[640, 960],
    )
    axes[0].legend()
    per_class = result["paired_interval"]["per_class"]
    axes[1].barh(FIVE_CLASSES, [100 * r["recovery"] for r in per_class])
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set(
        xlabel="Incremental AP points, inference 960", title="Train 960 minus train 640; one seed"
    )
    fig.savefig(report / "training.png", dpi=140)
    plt.close(fig)
    freeze_json(report / "evidence.json", {p.name: file_hash(p) for p in sorted(report.iterdir())})
    return result
