"""One bounded local-contrast study against the frozen, trained Raw S20/960 detector."""

import gzip
import importlib.metadata as md
import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from .common import digest, file_hash, freeze_json, read_image, read_json
from .experiment_data import shared_splits, source_rows
from .five_class import FIVE_CLASSES
from .preprocess import quality, validate_image

STUDY = "issue47-v3-960"
PLAN = "docs/ISSUE47_V3_960_PLAN.md"
MODULE = "src/vehicle_detection/v3_adaptive.py"
NOTEBOOK = "notebooks/experiments/v3/01_experiments.ipynb"
BASELINE = "issue47-s20-resolution-20e"
CHECKPOINT = "b13ec9339f7a58d55dd53c46f7e0e24330a889a761f3ee8cc4a3e9764573b2cb"
ARMS = ("raw", "always_local", "adaptive_local")
EXPOSURES = (
    "docs/issue6/policy.json",
    "docs/issue6-v2/policy.json",
    "docs/issue11-v1/pilot_membership.csv.gz",
    "docs/round2-preprocessing-v2/membership.csv.gz",
    "docs/round2-research-v3/membership.csv.gz",
    "docs/issue45/membership.csv",
    "docs/issue47/membership.csv",
)


def route(features):
    return (
        features["contrast"] < 0.20
        and 0.04 < features["brightness"] < 0.80
        and features["noise"] <= 0.01
        and features["black_fraction"] < 0.30
        and features["white_fraction"] < 0.10
    )


def local_contrast(image):
    validate_image(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.createCLAHE(1.5, (8, 8)).apply(gray)
    luminance = gray.astype(np.float32)
    protection = np.clip(np.minimum(luminance - 8, 247 - luminance) / 32, 0, 1)
    offset = np.rint(np.clip((equalized - luminance) * 0.5, -12, 12) * protection)
    return np.clip(image.astype(np.int16) + offset.astype(np.int16)[:, :, None], 0, 255).astype(
        np.uint8
    )


def apply(image, arm):
    if arm not in ARMS:
        raise ValueError("Unknown v3 contrast arm")
    features = quality(image)
    if arm == "always_local" or (arm == "adaptive_local" and route(features)):
        return local_contrast(image)
    return image


def archive(path, value):
    """Keep complete machine evidence losslessly, without a large text diff."""
    with Path(path).open("xb") as stream:
        stream.write(
            gzip.compress(json.dumps(value, sort_keys=True, allow_nan=False).encode(), mtime=0)
        )


def population(root, role="exploration"):
    """Metadata only: exclude all recorded exposures by whole leakage/content group."""
    root = Path(root)
    shared, rows = shared_splits(root)
    manifest = read_json(root / "docs/round2-evaluation-v2/evaluation_sets.json")
    if (
        manifest["manifest_id"] != digest({k: v for k, v in manifest.items() if k != "manifest_id"})
        or manifest["protocol_id"] != shared["source_protocol_id"]
        or role not in ("exploration", "confirmation")
    ):
        raise ValueError("Calibration partition identity changed")
    calibration = {r["id"]: r for r in rows if r["role"] == "calibration"}
    sets = [set(manifest["sets"][f"calibration_{r}"]) for r in ("exploration", "confirmation")]
    if sets[0] & sets[1] or sets[0] | sets[1] != set(calibration):
        raise ValueError("Calibration partition changed")
    exposed = set()
    for name in EXPOSURES:
        if name.endswith(".json"):
            ids = read_json(root / name)["calibration_ids"]
            if not set(ids) <= calibration.keys():
                raise ValueError("Historical calibration IDs changed")
            exposed.update(ids)
        else:
            for record in pd.read_csv(root / name, keep_default_na=False).to_dict("records"):
                if record["id"] not in calibration:
                    continue
                original = calibration[record["id"]]
                if record["group"] != original["group"]:
                    raise ValueError("Historical exposure group changed")
                for key in ("source_group", "sha256", "source_sha256"):
                    canonical = "sha256" if key == "source_sha256" else key
                    if key in record and record[key] != original[canonical]:
                        raise ValueError("Historical exposure metadata changed")
                exposed.add(record["id"])
    keys = ("group", "source_group", "sha256")
    forbidden = {(k, calibration[i][k]) for i in exposed for k in keys}
    allowed = set(manifest["sets"][f"calibration_{role}"])
    other = set(calibration) - allowed
    # Reject partition leakage, not just ID overlap with the opposite partition.
    other_tokens = {(k, calibration[i][k]) for i in other for k in keys}
    selected = []
    for sample_id in sorted(allowed):
        row = calibration[sample_id]
        if any((k, row[k]) in other_tokens for k in keys):
            raise ValueError("Calibration partitions share source/content groups")
        if not any((k, row[k]) in forbidden for k in keys):
            selected.append(row)
    if role == "confirmation":
        groups = {}
        for row in selected:
            groups.setdefault(row["group"], []).append(row)
        selected = []
        for group in sorted(groups, key=lambda g: (digest([2026, "v3-960-confirmation", g]), g)):
            selected.extend(groups[group])
            if len(selected) >= 1024:
                break
        if not 1024 <= len(selected) <= 1040:
            raise ValueError("Insufficient whole-group confirmation budget")
    return sorted(selected, key=lambda r: r["id"])


def read_report(root, phase="exploration"):
    """Read only compact committed evidence; no sources, models or statistics replay."""
    if phase not in ("exploration", "confirmation"):
        raise ValueError("Unknown study phase")
    report = Path(root) / f"docs/{STUDY}"
    if phase == "confirmation":
        report = report / "confirmation"
    if not (report / "evidence.json").exists():
        status = (
            "incomplete"
            if (report / "plan.json").exists() or (report / "failure.json").exists()
            else "not acquired"
        )
        return {"status": status, "policy_frozen": False, "v3_trained": False}
    evidence = read_json(report / "evidence.json")
    if "result.json" not in evidence:
        raise ValueError("Missing v3 result hash")
    for name, expected in evidence.items():
        if Path(name).name != name or file_hash(report / name) != expected:
            raise ValueError("V3 adaptive evidence changed")
    result = read_json(report / "result.json")
    for name, expected in result["inputs"].items():
        if file_hash(Path(root) / name) != expected:
            raise ValueError(f"Frozen v3 study input changed: {name}")
    return result


def assess(metrics, runtime, intervals):
    """Apply the prospective full-route, rare-class, regression and predictor gates."""
    before, after = metrics["raw"], metrics["adaptive_local"]
    reasons = []
    primary = intervals["all"]
    if after["all"]["mAP50_95"] - before["all"]["mAP50_95"] < 0.002:
        reasons.append("Primary gain below 0.2 AP points")
    if after["all"]["mAP50"] - before["all"]["mAP50"] < -0.001:
        reasons.append("mAP50 regression exceeds 0.1 AP points")
    if (
        primary["estimand"]["class_ids"] != list(range(5))
        or min(r["source_groups"] for r in primary["per_class"]) < 20
        or primary["valid_draws"] < 950
        or primary["recovery_ci_low"] <= 0
    ):
        reasons.append("Primary support/paired uncertainty gate failed")
    rare = [r for r in primary["per_class"] if r["class"] in ("motorcycle", "bicycle")]
    if len(rare) != 2 or any(
        r["recovery"] is None or r["ci_low"] is None or r["recovery"] < 0 or r["ci_low"] < -0.01
        for r in rare
    ):
        reasons.append("Individual two-wheeler AP/uncertainty gate failed")
    if (
        len(rare) != 2
        or any(r["recovery"] is None for r in rare)
        or np.mean([r["recovery"] for r in rare]) < 0.005
    ):
        reasons.append("Two-wheeler mean gain below 0.5 AP points")
    if any(intervals[name]["valid_draws"] < 950 for name in ("dark", "bright")):
        reasons.append("Cohort paired interval has fewer than 950 valid draws")
    for cohort in ("all", "dark", "bright", "apply", "skip"):
        for name, values in after[cohort]["classes"].items():
            delta = values["AP50_95"] - before[cohort]["classes"][name]["AP50_95"]
            if values["source_groups"] >= 5 and delta < -(0.02 if cohort == "all" else 0.01):
                reasons.append(f"Supported AP regression: {cohort}/{name}")
            if cohort not in ("dark", "bright") or values["source_groups"] < 5:
                continue
            a, b = before[cohort]["counts"][name], after[cohort]["counts"][name]
            ci = next(r for r in intervals[cohort]["per_class"] if r["class"] == name)
            if b["FN50"] - a["FN50"] >= 2 and b["FN50"] > a["FN50"] * 1.05:
                if ci["ci_low"] is None or ci["ci_low"] <= 0:
                    reasons.append(f"Inconclusive AP/FN contradiction: {cohort}/{name}")
    always = metrics["always_local"]
    common = [
        name
        for name in FIVE_CLASSES
        if all(
            before[s]["classes"].get(name, {}).get("source_groups", 0) >= 5
            for s in ("apply", "skip")
        )
    ]
    separation = None
    if len(common) < 3:
        reasons.append("Insufficient common route-class support for predictor test")
    else:
        gains = {
            s: float(
                np.mean(
                    [
                        always[s]["classes"][c]["AP50_95"] - before[s]["classes"][c]["AP50_95"]
                        for c in common
                    ]
                )
            )
            for s in ("apply", "skip")
        }
        separation = {
            "classes": common,
            "gains": gains,
            "difference": gains["apply"] - gains["skip"],
        }
        if gains["apply"] < 0.002 or separation["difference"] < 0.002:
            reasons.append("Routing signal does not predict required detection benefit")
    if after["all"]["mAP50_95"] < always["all"]["mAP50_95"] - 0.001:
        reasons.append("Adaptive AP worse than always-on by more than 0.1 points")
    if runtime["adaptive_local"]["mean_ms"] > 10 or runtime["adaptive_local"]["p95_ms"] > 25:
        reasons.append("CPU cost exceeds mean 10 ms / p95 25 ms")
    return {"advance": not reasons, "reasons": reasons, "routing_predictor": separation}


def inventory(root):
    """Preserve historical evidence bytes and local size/mtime without pixel rereads."""
    names = subprocess.check_output(["git", "ls-files"], cwd=root, text=True).splitlines()
    tracked = {
        n: file_hash(root / n)
        for n in names
        if (
            n.startswith(
                (
                    "docs/",
                    "configs/",
                    "notebooks/experiments/v0/",
                    "notebooks/experiments/v1/",
                    "notebooks/experiments/v2/",
                    "notebooks/research/",
                )
            )
        )
        and not n.startswith(f"docs/{STUDY}/")
        and n != PLAN
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


def run(root, *, allow_inference=False):
    """Acquire exactly one fresh three-arm exploration; default only verifies results."""
    root = Path(root).resolve()
    if not allow_inference:
        return read_report(root)
    output, report = root / f"runs/{STUDY}", root / f"docs/{STUDY}"
    if output.resolve() != output or report.resolve() != report:
        raise ValueError("Unexpected study output symlink")
    if output.exists() or output.is_symlink() or (report / "result.json").exists():
        raise FileExistsError("V3 acquisition attempt exists; preserve it and do not retry")
    paths = [
        PLAN,
        NOTEBOOK,
        "tests/test_v3_adaptive.py",
        f"docs/{STUDY}/membership.csv.gz",
        f"docs/{STUDY}/declaration.json",
    ]
    paths += subprocess.check_output(["git", "ls-files", "src"], cwd=root, text=True).splitlines()
    for name in paths:
        if (
            subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=root)
            != (root / name).read_bytes()
        ):
            raise ValueError("Commit the frozen protocol, membership and implementation first")
    if md.version("ultralytics") != "8.4.198":
        raise ValueError("Pinned Ultralytics version required")
    from .full_resolution_evidence import verify

    verify(root)
    checkpoint = root / f"runs/{BASELINE}/train960/weights/best.pt"
    if file_hash(checkpoint) != CHECKPOINT:
        raise ValueError("PR #52 Raw960 checkpoint changed")
    selected = population(root)
    declared = pd.read_csv(report / "membership.csv.gz", keep_default_na=False).to_dict("records")
    if selected != declared:
        raise ValueError("Frozen fresh exploration membership changed")
    declaration = read_json(report / "declaration.json")
    if digest(selected) != declaration["rows_sha256"]:
        raise ValueError("Premeasurement declaration changed")
    inputs = {
        name: file_hash(root / name)
        for name in paths
        + list(EXPOSURES)
        + [
            "configs/data.yaml",
            "configs/splits.csv.gz",
            "src/vehicle_detection/preprocess.py",
            "src/vehicle_detection/evaluation_stats.py",
            "src/vehicle_detection/method_discovery.py",
            "src/vehicle_detection/evaluation.py",
            f"docs/{BASELINE}/train960_checkpoint.json",
        ]
    }
    preservation = inventory(root)
    freeze_json(output / "preservation.json", preservation)
    freeze_json(output / "attempt.json", {"inputs": inputs, "checkpoint_sha256": CHECKPOINT})
    try:
        # Annotation metadata only; never call audit_sources(), which opens other roles.
        shared, annotated = source_rows(root)
        lookup = {r["id"]: r for r in annotated}
        selected = [lookup[r["id"]] for r in selected]
        support = {
            name: {
                "GT": sum(b[0] == c for r in selected for b in r["boxes"]),
                "source_groups": len(
                    {r["source_group"] for r in selected if any(b[0] == c for b in r["boxes"])}
                ),
            }
            for c, name in enumerate(FIVE_CLASSES)
        }
        if support != declaration["support"] or any(
            v["source_groups"] < 20 for v in support.values()
        ):
            raise ValueError("Frozen primary class support changed or is insufficient")
        freeze_json(output / "rows.json", selected)
        return acquire(root, selected, shared, checkpoint, inputs, preservation)
    except Exception as error:
        freeze_json(report / "failure.json", {"type": type(error).__name__, "error": str(error)})
        raise


def acquire(root, rows, shared, checkpoint, inputs, preservation, *, phase="exploration"):
    acquisition_started = time.perf_counter()
    import torch
    from ultralytics import YOLO

    from .baseline import VAL_SETTINGS
    from .datasets import verify_label
    from .evaluation import ExperimentValidator
    from .evaluation_stats import load_stats, paired_interval
    from .method_discovery import cohorts, outcome
    from .research_round3 import exploration_access
    from .resolution_training import check_mode

    if not torch.cuda.is_available():
        raise ValueError("CUDA required; no substitute acquisition hardware")
    output, report = root / f"runs/{STUDY}", root / f"docs/{STUDY}"
    if phase not in ("exploration", "confirmation"):
        raise ValueError("Unknown acquisition phase")
    if phase == "confirmation":
        output, report = output / "confirmation", report / "confirmation"
    data = (root / shared["data_root"]).resolve()
    if output.is_relative_to(data) or data.is_relative_to(output):
        raise ValueError("Output overlaps original sources")
    settings = {**VAL_SETTINGS, "imgsz": 960, "workers": 0, "half": False}
    plan = {
        "inputs": inputs,
        "checkpoint_sha256": CHECKPOINT,
        "settings": settings,
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "versions": {p: md.version(p) for p in ("ultralytics", "torch", "numpy", "opencv-python")},
        "gpu": torch.cuda.get_device_name(0),
        "cuda": torch.version.cuda,
        "images": len(rows),
        "source_groups": len({r["source_group"] for r in rows}),
        "ids_sha256": digest([r["id"] for r in rows]),
        "maximum_inference_image_passes": 3 * len(rows),
    }
    freeze_json(report / "plan.json", plan)
    features, exported, timings = {}, [], {a: [] for a in ARMS}
    previous = cv2.getNumThreads()
    try:
        cv2.setNumThreads(8)
        for arm in ARMS:
            for kind in ("images", "labels"):
                (output / arm / kind / "val").mkdir(parents=True)
        with exploration_access(data, {"rows": rows}) as reads:
            for index, row in enumerate(rows):
                source = data / row["source_path"]
                if source.resolve() != source or file_hash(source) != row["sha256"]:
                    raise ValueError("Original source bytes/path changed")
                image = read_image(source)
                if image.shape != (row["height"], row["width"], 3):
                    raise ValueError("Original source geometry changed")
                features[row["id"]] = quality(image)
                if index == 0:
                    for arm in ARMS:
                        apply(image, arm)
                for repeat in range(3):
                    offset = (index + repeat) % len(ARMS)
                    for arm in ARMS[offset:] + ARMS[:offset]:
                        started = time.perf_counter()
                        apply(image, arm)
                        timings[arm].append((time.perf_counter() - started) * 1000)
                labels, hashes = [], {}
                for arm in ARMS:
                    target = output / arm / "images/val" / f"{row['id']}.jpg"
                    if not cv2.imwrite(
                        str(target), apply(image, arm), [cv2.IMWRITE_JPEG_QUALITY, 95]
                    ):
                        raise OSError(f"JPEG95 export failed: {target}")
                    if read_image(target).shape != image.shape:
                        raise ValueError("Export geometry changed")
                    label = output / arm / "labels/val" / f"{row['id']}.txt"
                    label.write_text(
                        "".join(
                            f"{c} {(x1 + x2) / 2 / row['width']:g} "
                            f"{(y1 + y2) / 2 / row['height']:g} "
                            f"{(x2 - x1) / row['width']:g} {(y2 - y1) / row['height']:g}\n"
                            for c, x1, y1, x2, y2 in row["boxes"]
                        )
                    )
                    verify_label(label, row)
                    labels.append(file_hash(label))
                    hashes[arm] = file_hash(target)
                if len(set(labels)) != 1:
                    raise ValueError("Paired original label bytes changed")
                peer = "always_local" if route(features[row["id"]]) else "raw"
                if hashes["adaptive_local"] != hashes[peer]:
                    raise ValueError("Adaptive action differs from its exact control")
                exported.append({"id": row["id"], "images": hashes, "label_sha256": labels[0]})
                if (index + 1) % 250 == 0:
                    print(f"Exported/timed {index + 1}/{len(rows)} original IDs", flush=True)
            source_reads = sorted(reads)
    finally:
        cv2.setNumThreads(previous)
    freeze_json(output / "exports.json", exported)
    freeze_json(output / "timings.json", timings)
    archive(report / "features.json.gz", features)
    archive(report / "source_reads.json.gz", source_reads)
    subsets = {k: v for k, v in cohorts(rows, features).items() if not k.startswith("route_")}
    subsets.update(
        {
            "apply": [r for r in rows if route(features[r["id"]])],
            "skip": [r for r in rows if not route(features[r["id"]])],
        }
    )
    runtime = {
        a: {"mean_ms": float(np.mean(v)), "p95_ms": float(np.quantile(v, 0.95)), "calls": len(v)}
        for a, v in timings.items()
    }
    freeze_json(report / "runtime.json", runtime)
    statistics, metrics, measurements = {}, {}, {}
    ids = {r["id"] for r in rows}
    for arm in ARMS:
        directory = output / arm
        (directory / "data.yaml").write_text(
            yaml.safe_dump(
                {
                    "path": str(directory),
                    "train": "images/val",
                    "val": "images/val",
                    "names": dict(enumerate(FIVE_CLASSES)),
                }
            )
        )
        model = YOLO(str(checkpoint))
        if (
            model.names != dict(enumerate(FIVE_CLASSES))
            or model.model.yaml.get("yaml_file") != "yolo26n.yaml"
        ):
            raise ValueError("Raw S20 detector architecture/taxonomy changed")
        shapes, batches, clock = set(), [], {}

        def validate_loader(validator, shapes=shapes):
            dataset = validator.dataloader.dataset
            if (
                len(dataset.im_files) != len(ids)
                or {Path(p).stem for p in dataset.im_files} != ids
                or dataset.imgsz != 960
                or dataset.pad != 0.5
            ):
                raise ValueError("Native paired IDs/resolution/padding changed")
            preprocess = validator.preprocess

            def observed(batch):
                batch = preprocess(batch)
                if batch["img"].dtype != torch.float32:
                    raise ValueError("Native evaluation is not FP32")
                shapes.add(tuple(batch["img"].shape[-2:]))
                return batch

            validator.preprocess = observed

        def batch_start(validator, clock=clock):
            torch.cuda.synchronize()
            clock["start"] = time.perf_counter()

        def batch_end(validator, clock=clock, batches=batches):
            torch.cuda.synchronize()
            batches.append((time.perf_counter() - clock["start"]) * 1000)

        model.add_callback("on_val_start", validate_loader)
        model.add_callback("on_val_end", check_mode)
        model.add_callback("on_val_batch_start", batch_start)
        model.add_callback("on_val_batch_end", batch_end)
        started = time.perf_counter()
        native = model.val(
            validator=ExperimentValidator,
            data=str(directory / "data.yaml"),
            split="val",
            project=str(directory),
            name="eval",
            exist_ok=False,
            **settings,
        )
        measurements[arm] = {
            "wall_seconds": time.perf_counter() - started,
            "native_ms_per_image": native.speed,
            "batch_mean_ms": float(np.mean(batches)),
            "batch_p95_ms": float(np.quantile(batches, 0.95)),
            "batches": len(batches),
            "tensor_shapes": sorted(shapes),
            "native": dict(
                zip(
                    ("precision", "recall", "mAP50", "mAP50_95"),
                    map(float, native.mean_results()),
                    strict=True,
                )
            ),
        }
        freeze_json(report / f"{arm}_measurement.json", measurements[arm])
        path = directory / "eval/paired_stats.npz"
        statistics[arm] = load_stats(path)
        if set(statistics[arm]) != ids or any(
            not np.array_equal(statistics["raw"][i]["target_cls"], statistics[arm][i]["target_cls"])
            for i in ids
        ):
            raise ValueError("Native paired original GT/IDs changed")
        metrics[arm] = {name: outcome(statistics[arm], rs) for name, rs in subsets.items()}
        if not np.allclose(
            [native.box.map, native.box.map50],
            [metrics[arm]["all"]["mAP50_95"], metrics[arm]["all"]["mAP50"]],
            atol=1e-8,
            rtol=0,
        ):
            raise ValueError("Native and saved-match AP disagree")
        freeze_json(report / f"{arm}_metrics.json", metrics[arm])
        if file_hash(checkpoint) != CHECKPOINT:
            raise ValueError("Frozen Raw checkpoint changed")
        print(f"Completed native {arm}: {native.box.map:.8f} AP50:95", flush=True)
        del model
    intervals = {}
    for name in ("all", "dark", "bright"):
        rs = subsets[name]
        sample = {"rows": [{**r, "group": r["source_group"]} for r in rs]}
        subset_ids = {r["id"] for r in rs}
        interval = paired_interval(
            {i: v for i, v in statistics["raw"].items() if i in subset_ids},
            {i: v for i, v in statistics["adaptive_local"].items() if i in subset_ids},
            sample,
            resamples=1000,
            class_names=FIVE_CLASSES,
        )
        archive(report / f"{name}_interval.json.gz", interval)
        intervals[name] = {k: v for k, v in interval.items() if k != "draws"}
        print(f"Completed {name} paired interval", flush=True)
    examples = {}
    for c, name in enumerate(FIVE_CLASSES):
        counts = {}
        for row in rows:
            i = row["id"]
            counts[i] = {}
            for arm in ("raw", "adaptive_local"):
                values = statistics[arm][i]
                keep = (values["conf"] >= 0.25) & (values["pred_cls"] == c)
                tp = int(values["tp"][keep, 0].sum())
                counts[i][arm] = {"TP50": tp, "FP50": int(keep.sum()) - tp}
        ordered = sorted(
            ids,
            key=lambda i: (
                counts[i]["adaptive_local"]["TP50"] - counts[i]["raw"]["TP50"],
                counts[i]["raw"]["FP50"] - counts[i]["adaptive_local"]["FP50"],
                i,
            ),
        )
        examples[name] = {i: counts[i] for i in dict.fromkeys(ordered[:2] + ordered[-2:])}
    freeze_json(report / "examples.json", examples)
    if preservation != inventory(root):
        raise ValueError("Historical evidence or local file size/mtime changed")
    result = {
        "population": f"Fresh calibration {phase}; trained Raw S20/960; not development",
        "inputs": inputs,
        "metrics": metrics,
        "runtime": runtime,
        "intervals": intervals,
        "routing": dict(Counter("apply" if route(f) else "skip" for f in features.values())),
        "decision": assess(metrics, runtime, intervals),
        "measurements": measurements,
        "checkpoint_sha256": CHECKPOINT,
        "source_reads": {"count": len(source_reads), "paths_sha256": digest(source_reads)},
        "statistics": {
            a: {
                "path": str((output / a / "eval/paired_stats.npz").relative_to(root)),
                "sha256": file_hash(output / a / "eval/paired_stats.npz"),
            }
            for a in ARMS
        },
        "exports_sha256": file_hash(output / "exports.json"),
        "preservation": {
            "tracked_hashes": len(preservation["tracked"]),
            "local_size_mtimes": len(preservation["local"]),
        },
        "inference_image_passes": len(rows) * len(ARMS),
        "acquisition_wall_seconds": time.perf_counter() - acquisition_started,
        "cpu_threads": 8,
        "confirmation_accessed": phase == "confirmation",
        "development_accessed": False,
        "final_test_accessed": False,
        "policy_frozen": phase == "confirmation",
        "v3_trained": False,
    }
    freeze_json(report / "result.json", result)
    freeze_json(
        report / "evidence.json",
        {p.name: file_hash(p) for p in sorted(report.iterdir()) if p.is_file()},
    )
    return result
