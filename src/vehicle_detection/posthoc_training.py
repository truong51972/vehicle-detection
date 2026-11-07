"""One explicitly requested post-hoc Adaptive S20/960 run; frozen policy, native trainer."""

import hashlib
import importlib.metadata as md
import os
import shutil
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from .common import digest, file_hash, freeze_json, read_image, read_json
from .five_class import FIVE_CLASSES
from .full_resolution_training import install_selection, require_new, selection_provenance
from .v3_adaptive import BASELINE, CHECKPOINT, apply, archive, route

STUDY = "issue47-posthoc-adaptive-960"
PLAN = "docs/ISSUE47_POSTHOC_TRAINING_PLAN.md"
NOTEBOOK = "notebooks/research/issue47/05_posthoc_adaptive_training.ipynb"
MODULE = "src/vehicle_detection/posthoc_training.py"
MATRIX = (("raw", "raw"), ("raw", "adaptive"), ("adaptive", "raw"), ("adaptive", "adaptive"))
ADMIN_ARGS = {"data", "project", "name", "save_dir"}


def read_report(root):
    report = Path(root) / f"docs/{STUDY}"
    if not (report / "evidence.json").exists():
        return {"status": "incomplete" if (report / "plan.json").exists() else "not acquired"}
    evidence = read_json(report / "evidence.json")
    if "result.json" not in evidence:
        raise ValueError("Missing post-hoc result hash")
    for name, expected in evidence.items():
        if Path(name).name != name or file_hash(report / name) != expected:
            raise ValueError("Post-hoc evidence changed")
    return read_json(report / "result.json")


def frozen_inputs(root):
    paths = [PLAN, MODULE, NOTEBOOK, "tests/test_posthoc_training.py", "uv.lock"]
    paths += [
        f"src/vehicle_detection/{n}.py"
        for n in (
            "common",
            "baseline",
            "training_settings",
            "full_resolution_training",
            "v3_adaptive",
            "preprocess",
            "experiment_data",
            "datasets",
            "evaluation",
            "evaluation_stats",
            "five_class",
            "method_discovery",
            "resolution_training",
            "research_round3",
        )
    ]
    for name in paths:
        if (
            subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=root)
            != (root / name).read_bytes()
        ):
            raise ValueError("Commit post-hoc protocol, code, notebook and safeguards first")
    return {name: file_hash(root / name) for name in paths}


def inventory(root):
    names = subprocess.check_output(["git", "ls-files"], cwd=root, text=True).splitlines()
    mutable = {
        "README.md",
        "notebooks/experiments/compare.ipynb",
        "notebooks/experiments/v3/README.md",
    }
    tracked = {
        n: file_hash(root / n)
        for n in names
        if n not in mutable and not n.startswith(f"docs/{STUDY}/")
    }
    local = {}
    for directory in (root / "runs", root / "weights"):
        for base, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if Path(base) / d != root / f"runs/{STUDY}"]
            for name in files:
                path = Path(base) / name
                stat = path.stat()
                local[str(path.relative_to(root))] = [stat.st_size, stat.st_mtime_ns]
    return {"tracked": tracked, "local": local}


def preservation(root):
    before = read_json(root / f"runs/{STUDY}/preservation.json")
    for name, expected in before["tracked"].items():
        if file_hash(root / name) != expected:
            raise ValueError(f"Historical tracked file changed: {name}")
    for name, expected in before["local"].items():
        stat = (root / name).stat()
        if [stat.st_size, stat.st_mtime_ns] != expected:
            raise ValueError(f"Historical local file changed: {name}")
    return {"tracked_hashes": len(before["tracked"]), "local_size_mtimes": len(before["local"])}


def export_row(row, split, source_root, raw_root, output, expected):
    """Original pixels → frozen action → JPEG95 once; exact original label bytes."""
    from .datasets import verify_label
    from .preprocess import quality

    sample_id = row["id"]
    original = source_root / row["source_path"]
    if original.resolve() != original or file_hash(original) != row["sha256"]:
        raise ValueError("Original source path/bytes changed")
    raw_image = raw_root / f"images/{split}/{sample_id}.jpg"
    raw_label = raw_root / f"labels/{split}/{sample_id}.txt"
    if any(p.resolve() != p for p in (raw_image, raw_label)) or [
        file_hash(raw_image),
        file_hash(raw_label),
    ] != list(expected):
        raise ValueError("Historical Raw export changed")
    verify_label(raw_label, row)
    labels = raw_label.read_bytes()
    image = read_image(original)
    if image.shape != (row["height"], row["width"], 3):
        raise ValueError("Original geometry changed")
    ok, raw_encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok or hashlib.sha256(raw_encoded.tobytes()).hexdigest() != expected[0]:
        raise ValueError("Raw JPEG95 reproduction differs from PR #52")
    features = quality(image)
    transformed = apply(image, "adaptive_local")
    ok, encoded = cv2.imencode(".jpg", transformed, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok or cv2.imdecode(encoded, cv2.IMREAD_COLOR).shape != image.shape:
        raise ValueError("Adaptive JPEG95 export failed/changed geometry")
    content = encoded.tobytes()
    adaptive_hash = hashlib.sha256(content).hexdigest()
    if not route(features) and adaptive_hash != expected[0]:
        raise ValueError("Skipped Adaptive bytes differ from Raw")
    (output / f"data/adaptive/images/{split}/{sample_id}.jpg").write_bytes(content)
    (output / f"data/adaptive/labels/{split}/{sample_id}.txt").write_bytes(labels)
    if split == "val":
        (output / f"data/raw/images/val/{sample_id}.jpg").write_bytes(raw_encoded.tobytes())
        (output / f"data/raw/labels/val/{sample_id}.txt").write_bytes(labels)
    return [split, sample_id, expected[0], adaptive_hash, expected[1], route(features)], features


def prepare(root, *, allow_preparation=False):
    if not allow_preparation:
        return None
    from .experiment_data import populations, shared_splits
    from .full_resolution_evidence import verify
    from .research_round3 import exploration_access

    root = Path(root).resolve()
    output, report = root / f"runs/{STUDY}", root / f"docs/{STUDY}"
    require_new(output)
    require_new(report)
    inputs = frozen_inputs(root)
    verify(root)
    old = root / f"runs/{BASELINE}"
    old_receipt = read_json(old / "prepared.json")["sha256"]
    for name, expected in old_receipt.items():
        if file_hash(old / name) != expected:
            raise ValueError("PR #52 prepared metadata changed")
    if any(md.version(p) != v for p, v in read_json(old / "plan.json")["versions"].items()):
        raise ValueError("Use PR #52's recorded package versions")
    shared, canonical = shared_splits(root)
    canonical = populations(canonical, "S20")
    selected = read_json(old / "rows.json")
    if (
        digest(read_json(old / "exports.json"))
        != read_json(root / f"docs/{BASELINE}/data.json")["files_sha256"]
    ):
        raise ValueError("Raw export receipt differs from PR #52's committed identity")
    for split, count in (("train", 9781), ("val", 13975)):
        if (
            len(selected[split]) != count
            or [{k: v for k, v in r.items() if k != "boxes"} for r in selected[split]]
            != canonical[split]
        ):
            raise ValueError("Frozen S20/development metadata changed")
    if shutil.disk_usage(root).free < 20 * 1024**3:
        raise OSError("Require 20 GiB free disk")
    before = inventory(root)
    freeze_json(output / "preservation.json", before)
    plan = {
        "inputs": inputs,
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "matrix": MATRIX,
        "scope": "Post-hoc exploration; not independent confirmation",
        "baseline_prepared_sha256": file_hash(old / "prepared.json"),
        "versions": read_json(old / "plan.json")["versions"],
    }
    freeze_json(output / "plan.json", plan)
    freeze_json(report / "plan.json", plan)
    for domain, splits in (("raw", ("val",)), ("adaptive", ("train", "val"))):
        for split in splits:
            for kind in ("images", "labels"):
                (output / f"data/{domain}/{kind}/{split}").mkdir(parents=True)
    lookup = {(split, i): (ih, lh) for split, i, ih, lh in read_json(old / "exports.json")}
    source_root = (root / shared["data_root"]).resolve()
    previous_threads = cv2.getNumThreads()
    started = time.perf_counter()
    all_rows = [(split, row) for split, rows in selected.items() for row in rows]

    def exporting(item):
        split, row = item
        return export_row(row, split, source_root, old / "data", output, lookup[split, row["id"]])

    try:
        cv2.setNumThreads(1)
        with (
            exploration_access(source_root, {"rows": [r for _, r in all_rows]}),
            ThreadPoolExecutor(max_workers=4) as pool,
        ):
            records, features = [], {}
            for index, (record, scores) in enumerate(pool.map(exporting, all_rows)):
                records.append(record)
                features[record[1]] = scores
                if (index + 1) % 1000 == 0:
                    print(f"Prepared {index + 1}/{len(all_rows)} original IDs", flush=True)
    finally:
        cv2.setNumThreads(previous_threads)
    freeze_json(output / "exports.json", records)
    archive(output / "routing_features.json.gz", features)
    shutil.copyfile(old / "rows.json", output / "rows.json")
    shutil.copyfile(old / "features.json", output / "features.json")
    for domain in ("raw", "adaptive"):
        config = {
            "path": str(output / f"data/{domain}"),
            "train": "images/train" if domain == "adaptive" else "images/val",
            "val": "images/val",
            "names": dict(enumerate(FIVE_CLASSES)),
        }
        (output / f"data/{domain}/data.yaml").write_text(yaml.safe_dump(config))
    training = {
        "path": str(output / "data/adaptive"),
        "train": "images/train",
        "val": str(output / "data/raw/images/val"),
        "names": dict(enumerate(FIVE_CLASSES)),
    }
    (output / "train.yaml").write_text(yaml.safe_dump(training))
    names = [
        "exports.json",
        "routing_features.json.gz",
        "rows.json",
        "features.json",
        "plan.json",
        "train.yaml",
        "data/raw/data.yaml",
        "data/adaptive/data.yaml",
    ]
    freeze_json(output / "prepared.json", {"sha256": {n: file_hash(output / n) for n in names}})
    data = {
        "preparation_wall_seconds": time.perf_counter() - started,
        "images": {s: len(rs) for s, rs in selected.items()},
        "ids_sha256": {s: digest(sorted(r["id"] for r in rs)) for s, rs in selected.items()},
        "routing": {
            s: dict(Counter("apply" if r[5] else "skip" for r in records if r[0] == s))
            for s in selected
        },
        "negative_images": {s: sum(not r["boxes"] for r in rs) for s, rs in selected.items()},
        "prepared_sha256": file_hash(output / "prepared.json"),
        "preservation": preservation(root),
    }
    freeze_json(report / "data.json", data)
    return data


def prepared_inputs(root):
    output = root / f"runs/{STUDY}"
    if read_json(output / "plan.json")["inputs"] != frozen_inputs(root):
        raise ValueError("Frozen post-hoc implementation changed")
    for name, expected in read_json(output / "prepared.json")["sha256"].items():
        if file_hash(output / name) != expected:
            raise ValueError("Post-hoc prepared metadata changed")
    for split, i, raw, adaptive, label, _ in read_json(output / "exports.json"):
        pairs = [("adaptive", split, adaptive)] + ([("raw", "val", raw)] if split == "val" else [])
        for domain, subset, image in pairs:
            if (
                file_hash(output / f"data/{domain}/images/{subset}/{i}.jpg") != image
                or file_hash(output / f"data/{domain}/labels/{subset}/{i}.txt") != label
            ):
                raise ValueError("Post-hoc owned image/label bytes changed")
    return read_json(output / "rows.json")


def compare_args(actual, reference):
    if {k: v for k, v in actual.items() if k not in ADMIN_ARGS} != {
        k: v for k, v in reference.items() if k not in ADMIN_ARGS
    }:
        raise ValueError("Resolved training arguments differ from PR #52 beyond data/output paths")


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
    require_new(output / "train")
    selected = prepared_inputs(root)
    checkpoint = root / "weights/yolo26n.pt"
    if file_hash(checkpoint) != PRETRAINED_SHA256 or not torch.cuda.is_available():
        raise ValueError("Pinned COCO checkpoint and CUDA required")
    freeze_json(
        output / "training-attempt.json",
        {
            "checkpoint_sha256": PRETRAINED_SHA256,
            "gpu": torch.cuda.get_device_name(0),
            "cuda": torch.version.cuda,
        },
    )
    model = YOLO(str(checkpoint))
    verify_coco_names(model.names)
    observed = install_selection(
        model,
        [r["id"] for r in selected["train"]],
        [r["id"] for r in selected["val"]],
        960,
        output / "train/selection",
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    measurement = {"status": "incomplete", "imgsz": 960}
    try:
        model.train(
            data=str(output / "train.yaml"),
            freeze=backbone_boundary(model.model),
            project=str(output),
            name="train",
            exist_ok=False,
            **{**TRAIN_SETTINGS, "imgsz": 960, "workers": 0},
        )
        measurement["status"] = "completed"
    finally:
        measurement.update(
            wall_seconds=time.perf_counter() - started,
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3,
        )
        freeze_json(report / "training_measurement.json", measurement)
    args = yaml.safe_load((output / "train/args.yaml").read_text())
    validate_training_args(
        args, output / "train.yaml", {**TRAIN_SETTINGS, "imgsz": 960, "workers": 0}
    )
    compare_args(args, yaml.safe_load((root / f"docs/{BASELINE}/train960_args.yaml").read_text()))
    raw = read_json(root / f"docs/{BASELINE}/train960_checkpoint.json")
    if observed["trainable_parameters"] != raw["trainable_parameters"]:
        raise ValueError("Initialized trainable parameters differ from Raw")
    history = pd.read_csv(output / "train/results.csv")
    if history.epoch.tolist() != list(range(1, 21)):
        raise ValueError("Incomplete 20-epoch training")
    best = torch.load(output / "train/weights/best.pt", weights_only=False, map_location="cpu")
    provenance = selection_provenance(history, best, observed)
    result = {
        **provenance,
        "selection_passes": observed["passes"],
        "train_shapes": sorted(observed["train_shapes"]),
        "trainable_parameters": observed["trainable_parameters"],
        "checkpoint_hashes": {
            n: file_hash(output / f"train/weights/{n}.pt") for n in ("best", "last")
        },
    }
    freeze_json(report / "checkpoint.json", result)
    shutil.copyfile(output / "train/args.yaml", report / "training_args.yaml")
    shutil.copyfile(output / "train/results.csv", report / "training_results.csv")
    freeze_json(
        output / "trained.json",
        {"checkpoint_receipt_sha256": file_hash(report / "checkpoint.json")},
    )
    preservation(root)
    return result


def contrasts(metrics):
    values = {k: v["all"]["mAP50_95"] for k, v in metrics.items()}
    a, b, c, d = (
        values[k] for k in ("raw/raw", "raw/adaptive", "adaptive/raw", "adaptive/adaptive")
    )
    return {
        "adaptive_training_on_adaptive": d - b,
        "adaptive_training_on_raw": c - a,
        "preprocessing_raw_model": b - a,
        "preprocessing_adaptive_model": d - c,
        "joint_pipeline": d - a,
        "interaction": (d - b) - (c - a),
    }


def evaluate(root, *, allow_inference=False):
    if not allow_inference:
        return None
    import torch
    from ultralytics import YOLO

    from .baseline import VAL_SETTINGS
    from .evaluation import ExperimentValidator
    from .evaluation_stats import load_stats
    from .method_discovery import cohorts, outcome
    from .resolution_training import check_mode

    root = Path(root).resolve()
    output, report = root / f"runs/{STUDY}", root / f"docs/{STUDY}"
    require_new(output / "evaluation-attempt.json")
    selected = prepared_inputs(root)
    if read_json(output / "trained.json")["checkpoint_receipt_sha256"] != file_hash(
        report / "checkpoint.json"
    ):
        raise ValueError("Adaptive checkpoint receipt changed")
    freeze_json(output / "evaluation-attempt.json", {"matrix": MATRIX})
    groups = {
        k: v
        for k, v in cohorts(selected["val"], read_json(output / "features.json")).items()
        if k in ("all", "dark", "bright", "small_two_wheeler")
    }
    ids = {r["id"] for r in selected["val"]}
    models = {
        "raw": (root / f"runs/{BASELINE}/train960/weights/best.pt", CHECKPOINT),
        "adaptive": (
            output / "train/weights/best.pt",
            read_json(report / "checkpoint.json")["checkpoint_hashes"]["best"],
        ),
    }
    metrics, timings, statistics = {}, {}, {}
    reference_stats = None
    for trained, domain in MATRIX:
        key, name = f"{trained}/{domain}", f"eval_{trained}_{domain}"
        require_new(output / name)
        checkpoint, expected = models[trained]
        if file_hash(checkpoint) != expected:
            raise ValueError("Matrix checkpoint changed")
        model = YOLO(str(checkpoint))
        if model.names != dict(enumerate(FIVE_CLASSES)):
            raise ValueError("Matrix taxonomy changed")
        shapes, batches, clock = set(), [], {}

        def check_loader(validator, shapes=shapes):
            dataset = validator.dataloader.dataset
            if (
                len(dataset.im_files) != len(ids)
                or {Path(p).stem for p in dataset.im_files} != ids
                or dataset.imgsz != 960
                or dataset.pad != 0.5
            ):
                raise ValueError("Matrix loader IDs/resolution/padding changed")
            preprocess = validator.preprocess

            def observed(batch):
                batch = preprocess(batch)
                if batch["img"].dtype != torch.float32:
                    raise ValueError("Matrix requires FP32")
                shapes.add(tuple(batch["img"].shape[-2:]))
                return batch

            validator.preprocess = observed

        def batch_start(validator, clock=clock):
            torch.cuda.synchronize()
            clock["start"] = time.perf_counter()

        def batch_end(validator, clock=clock, batches=batches):
            torch.cuda.synchronize()
            batches.append((time.perf_counter() - clock["start"]) * 1000)

        model.add_callback("on_val_start", check_loader)
        model.add_callback("on_val_end", check_mode)
        model.add_callback("on_val_batch_start", batch_start)
        model.add_callback("on_val_batch_end", batch_end)
        started = time.perf_counter()
        native = model.val(
            validator=ExperimentValidator,
            data=str(output / f"data/{domain}/data.yaml"),
            project=str(output),
            name=name,
            exist_ok=False,
            **{**VAL_SETTINGS, "imgsz": 960, "workers": 0, "half": False},
        )
        timings[key] = {
            "wall_seconds": time.perf_counter() - started,
            "native_ms_per_image": native.speed,
            "batch_mean_ms": float(np.mean(batches)),
            "batch_p95_ms": float(np.quantile(batches, 0.95)),
            "batches": len(batches),
            "tensor_shapes": sorted(shapes),
            "checkpoint_sha256": expected,
        }
        freeze_json(report / f"{name}_measurement.json", timings[key])
        path = output / name / "paired_stats.npz"
        stats = load_stats(path)
        if reference_stats is None:
            reference_stats = {i: v["target_cls"] for i, v in stats.items()}
        if set(stats) != ids or any(
            not np.array_equal(stats[i]["target_cls"], reference_stats[i]) for i in ids
        ):
            raise ValueError("Matrix original GT/IDs changed")
        metrics[key] = {k: outcome(stats, rs) for k, rs in groups.items()}
        if not np.allclose(
            [native.box.map, native.box.map50],
            [metrics[key]["all"]["mAP50_95"], metrics[key]["all"]["mAP50"]],
            atol=1e-8,
            rtol=0,
        ):
            raise ValueError("Matrix native/saved AP disagree")
        freeze_json(report / f"{name}_metrics.json", metrics[key])
        if key == "raw/raw":
            prior = read_json(root / f"docs/{BASELINE}/eval960_960_metrics.json")
            for group, record in metrics[key].items():
                for cls, values in record["classes"].items():
                    if any(
                        abs(values[k] - prior[group]["classes"][cls][k]) > 1e-6
                        for k in ("AP50", "AP50_95")
                    ):
                        raise ValueError("Raw/Raw control does not reproduce PR #52")
                if record["counts"] != prior[group]["counts"]:
                    raise ValueError("Raw/Raw native counts do not reproduce PR #52")
        if file_hash(checkpoint) != expected:
            raise ValueError("Matrix evaluation modified checkpoint")
        statistics[key] = {"path": str(path.relative_to(root)), "sha256": file_hash(path)}
        print(f"Completed {key}: {native.box.map:.8f} AP50:95", flush=True)
        del model, stats
    result = {
        "status": "completed",
        "scope": "Post-hoc exploration, not independent confirmation or policy release",
        "metrics": metrics,
        "contrasts": contrasts(metrics),
        "timings": timings,
        "statistics": statistics,
        "data": read_json(report / "data.json"),
        "training": {
            "raw": read_json(root / f"docs/{BASELINE}/train960_measurement.json"),
            "adaptive": read_json(report / "training_measurement.json"),
        },
        "selected_epoch": {
            "raw": 10,
            "adaptive": read_json(report / "checkpoint.json")["selected_epoch"],
        },
        "new_training_exposures": 195620,
        "new_explicit_evaluation_image_passes": 55900,
        "new_selection_and_terminal_image_passes": 293475,
        "confirmation_accessed": False,
        "final_test_accessed": False,
        "preservation": preservation(root),
    }
    freeze_json(report / "result.json", result)
    freeze_json(
        report / "evidence.json",
        {p.name: file_hash(p) for p in sorted(report.iterdir()) if p.is_file()},
    )
    return result
