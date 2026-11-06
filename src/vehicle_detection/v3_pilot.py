"""Fixed natural-image routing hypotheses; no search, training or confirmation."""

import importlib.metadata as md
import subprocess
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from .common import digest, file_hash, freeze_json, read_image, read_json
from .datasets import verify_label
from .experiment_data import shared_splits, source_rows
from .five_class import FIVE_CLASSES
from .preprocess import gamma, quality, sharpen
from .v2_pilot import metric_record, select_sample

ARMS = ("skip", "exposure", "detail", "routed")


def route(features):
    """Original-pixel decisions only; never recompute a route after an operator."""
    exposure = (
        features["brightness"] < 0.18
        and features["contrast"] < 0.20
        and features["black_fraction"] < 0.10
        and features["noise"] <= 0.01
    )
    detail = 50 <= features["sharpness"] < 150 and features["noise"] <= 0.01
    if exposure and detail:
        return "both"
    if exposure:
        return "exposure"
    return "detail" if detail else "skip"


def apply(image, arm):
    if arm not in ARMS:
        raise ValueError("Unknown pilot arm")
    decision = route(quality(image))
    output = image
    if arm in ("exposure", "routed") and decision in ("exposure", "both"):
        output = gamma(output, 0.85)
    if arm in ("detail", "routed") and decision in ("detail", "both"):
        output = sharpen(output, 0.15, 1.0)
    return output


def sample(rows, manifest, consumed):
    """Four successive historical 256-image prefixes, with no label/outcome selection."""
    selected = []
    for _ in range(4):
        chunk = select_sample(rows, manifest, consumed)
        selected.extend(chunk)
        consumed = pd.concat([consumed, pd.DataFrame(chunk)], ignore_index=True)
    return sorted(selected, key=lambda r: r["id"])


def assess(metrics, runtime, interval):
    """Advance only the declared complete route; never promote a component post hoc."""
    reasons = []
    candidate, control = metrics["routed"], metrics["skip"]
    if candidate["all"]["mAP50_95"] - control["all"]["mAP50_95"] < 0.002:
        reasons.append("Primary gain below 0.2 pp vs Skip")
    for subgroup, record in candidate.items():
        for name, values in record["classes"].items():
            limit = 0.02 if subgroup == "all" else 0.01
            if (
                values["source_groups"] >= 5
                and values["AP50_95"] - control[subgroup]["classes"][name]["AP50_95"] < -limit
            ):
                reasons.append(f"Supported class regression: {subgroup}/{name}")
    for subgroup, components in (
        ("exposure", ("skip",)),
        ("detail", ("skip",)),
        ("both", ("exposure", "detail")),
    ):
        record = candidate[subgroup]
        if (
            set(record["classes"]) != set(FIVE_CLASSES)
            or min(v["source_groups"] for v in record["classes"].values()) < 5
        ):
            reasons.append(f"Insufficient five-class route support: {subgroup}")
        elif any(record["mAP50_95"] - metrics[c][subgroup]["mAP50_95"] < 0.002 for c in components):
            reasons.append(f"Route benefit below 0.2 pp: {subgroup}")
    if interval["estimand"]["class_ids"] != list(range(5)) or interval["recovery_ci_low"] <= 0:
        reasons.append("Five-class paired interval does not establish positive benefit")
    if runtime["routed"]["mean_ms"] > 10 or runtime["routed"]["p95_ms"] > 25:
        reasons.append("Preprocessing exceeds mean 10 ms / p95 25 ms")
    return {"advance": not reasons, "reasons": reasons, "policy_frozen": False, "v3_trained": False}


def run(root, *, allow_inference=False):
    """Default verifies committed results only; opt-in consumes one new fixed slice."""
    root = Path(root).resolve()
    report, output = root / "docs/issue47", root / "runs/issue47/pilot"
    if not allow_inference:
        for name, expected in read_json(report / "evidence.json").items():
            if Path(name).name != name or file_hash(report / name) != expected:
                raise ValueError("V3 pilot evidence changed")
        return read_json(report / "summary.json")
    if output.exists() or output.is_symlink() or report.exists():
        raise FileExistsError("V3 attempt exists; never overwrite consumed exploration")
    protected = [
        "docs/ISSUE47_PLAN.md",
        "src/vehicle_detection/v3_pilot.py",
        "src/vehicle_detection/preprocess.py",
        "src/vehicle_detection/v2_pilot.py",
        "src/vehicle_detection/experiment_data.py",
        "src/vehicle_detection/evaluation_stats.py",
        "src/vehicle_detection/stress_eval.py",
        "src/vehicle_detection/baseline.py",
        "notebooks/experiments/v3/01_experiments.ipynb",
    ]
    for name in protected:
        committed = subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=root)
        if committed != (root / name).read_bytes():
            raise ValueError("Commit the declaration and implementation before measuring")
    shared, rows = shared_splits(root)
    manifest = read_json(root / "docs/round2-evaluation-v2/evaluation_sets.json")
    if manifest["protocol_id"] != shared["source_protocol_id"]:
        raise ValueError("Foreign exploration protocol")
    prior = (
        root / "docs/round2-research-v3/membership.csv.gz",
        root / "docs/issue45/membership.csv",
    )
    for path in prior:
        receipt = read_json(path.parent / "evidence.json")
        expected = (
            receipt["artifacts"][str(path.relative_to(root))]
            if "artifacts" in receipt
            else receipt[path.name]
        )
        if file_hash(path) != expected:
            raise ValueError("Consumed exploration membership changed")
    selected = sample(rows, manifest, pd.concat([pd.read_csv(p) for p in prior]))
    _, annotated = source_rows(root)
    lookup = {r["id"]: r for r in annotated}
    selected = [lookup[r["id"]] for r in selected]
    support = {
        name: len({r["source_group"] for r in selected if any(b[0] == c for b in r["boxes"])})
        for c, name in enumerate(FIVE_CLASSES)
    }
    if min(support.values()) < 5:
        raise ValueError(f"Insufficient primary support; no inference: {support}")

    import torch
    from ultralytics import YOLO

    from .baseline import PRETRAINED_SHA256, verify_coco_names
    from .evaluation_stats import load_stats, paired_interval
    from .research_round3 import exploration_access
    from .research_round3_report import errors
    from .stress_eval import SETTINGS, CalibrationValidator

    checkpoint = root / "weights/yolo26n.pt"
    if file_hash(checkpoint) != PRETRAINED_SHA256:
        raise ValueError("Pinned checkpoint changed")
    plan = {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "inputs": {
            name: file_hash(root / name)
            for name in protected
            + ["configs/data.yaml", "configs/splits.csv.gz"]
            + [str(p.relative_to(root)) for p in prior]
            + ["docs/round2-evaluation-v2/evaluation_sets.json"]
        },
        "rows_sha256": digest(selected),
        "ids_sha256": digest([r["id"] for r in selected]),
        "checkpoint_sha256": PRETRAINED_SHA256,
        "settings": {**SETTINGS, "half": False},
        "support": support,
        "role": "calibration_exploration",
        "versions": {p: md.version(p) for p in ("ultralytics", "torch", "numpy", "opencv-python")},
        "hardware": {"gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda},
        "maximum_image_passes": 4352,
    }
    data = (root / shared["data_root"]).resolve()
    if data.is_relative_to(output) or output.is_relative_to(data):
        raise ValueError("Output overlaps sources")
    freeze_json(output / "plan.json", plan)
    freeze_json(report / "plan.json", plan)
    pd.DataFrame(selected).drop(columns="boxes").to_csv(report / "membership.csv", index=False)
    features, timings, exports = {}, {a: [] for a in ARMS}, []
    previous_threads = cv2.getNumThreads()
    start = time.perf_counter()
    try:
        cv2.setNumThreads(8)
        for arm in ARMS:
            for kind in ("images", "labels"):
                (output / arm / kind / "val").mkdir(parents=True)
        with exploration_access(data, {"rows": selected}) as reads:
            for index, row in enumerate(selected):
                source = data / row["source_path"]
                if source.resolve() != source or file_hash(source) != row["sha256"]:
                    raise ValueError("Source path or bytes changed")
                image = read_image(source)
                if image.shape != (row["height"], row["width"], 3):
                    raise ValueError("Source geometry changed")
                features[row["id"]] = quality(image)
                if index == 0:
                    for arm in ARMS:
                        apply(image, arm)
                for repeat in range(3):
                    offset = (index + repeat) % len(ARMS)
                    for arm in ARMS[offset:] + ARMS[:offset]:
                        tick = time.perf_counter()
                        apply(image, arm)
                        timings[arm].append((time.perf_counter() - tick) * 1000)
                labels, image_hashes = [], []
                for arm in ARMS:
                    target = output / arm / "images/val" / f"{row['id']}.jpg"
                    if not cv2.imwrite(
                        str(target), apply(image, arm), [cv2.IMWRITE_JPEG_QUALITY, 95]
                    ):
                        raise OSError(f"Could not encode {target}")
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
                    labels.append(label.read_bytes())
                    image_hashes.append(file_hash(target))
                    exports.append(
                        (str(target.relative_to(output)), file_hash(target), file_hash(label))
                    )
                if len(set(labels)) != 1:
                    raise ValueError("Paired label bytes differ")
                if route(features[row["id"]]) == "skip" and len(set(image_hashes)) != 1:
                    raise ValueError("Skipped images changed between arms")
            source_reads = sorted(reads)
    finally:
        cv2.setNumThreads(previous_threads)
    freeze_json(output / "exports.json", exports)
    freeze_json(report / "features.json", features)
    routing = {i: route(f) for i, f in features.items()}
    subgroups = {
        "all": selected,
        **{
            key: [r for r in selected if routing[r["id"]] == key]
            for key in ("exposure", "detail", "both", "skip")
        },
        "dark": [r for r in selected if features[r["id"]]["brightness"] < 0.12],
        "bright": [r for r in selected if features[r["id"]]["brightness"] >= 0.12],
        "blurred": [r for r in selected if features[r["id"]]["sharpness"] < 50],
        "small_two_wheeler": [
            r
            for r in selected
            if any(
                c in (3, 4) and (x2 - x1) * (y2 - y1) < 32**2 for c, x1, y1, x2, y2 in r["boxes"]
            )
        ],
    }
    runtime = {
        a: {"mean_ms": float(np.mean(v)), "p95_ms": float(np.quantile(v, 0.95)), "calls": len(v)}
        for a, v in timings.items()
    }
    stats, metrics, native, wall = {}, {}, {}, {}
    expected = {r["id"] for r in selected}

    def check_loader(validator):
        files = validator.dataloader.dataset.im_files
        if len(files) != len(expected) or {Path(p).stem for p in files} != expected:
            raise ValueError("Native loader changed paired IDs")

    def check_mode(validator):
        if not validator.end2end or validator.args.nms is not False:
            raise ValueError("Native NMS-free contract changed")

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
        verify_coco_names(model.names)
        if model.model.yaml.get("yaml_file") != "yolo26n.yaml":
            raise ValueError("Wrong pretrained architecture")
        model.add_callback("on_val_start", check_loader)
        model.add_callback("on_val_end", check_mode)
        tick = time.perf_counter()
        result = model.val(
            validator=CalibrationValidator,
            data=str(directory / "data.yaml"),
            split="val",
            project=str(directory),
            name="eval",
            exist_ok=True,
            **plan["settings"],
        )
        wall[arm] = time.perf_counter() - tick
        stats[arm] = load_stats(directory / "eval/paired_stats.npz")
        if set(stats[arm]) != expected or any(
            not np.array_equal(stats["skip"][i]["target_cls"], stats[arm][i]["target_cls"])
            for i in expected
        ):
            raise ValueError("Native paired IDs/GT changed")
        metrics[arm] = {key: metric_record(stats[arm], rs) for key, rs in subgroups.items()}
        native[arm] = dict(
            zip(
                ("precision", "recall", "mAP50", "mAP50_95"),
                map(float, result.mean_results()),
                strict=True,
            )
        )
        if not np.isfinite(list(native[arm].values())).all() or not np.isclose(
            native[arm]["mAP50_95"], metrics[arm]["all"]["mAP50_95"]
        ):
            raise ValueError("Native and saved AP disagree")
    # Resample original source groups, rather than merged duplicate/leakage groups.
    bootstrap_rows = [{**r, "group": r["source_group"]} for r in selected]
    interval = paired_interval(
        stats["skip"], stats["routed"], {"rows": bootstrap_rows}, class_names=FIVE_CLASSES
    )
    counts = {r["id"]: {a: errors(stats[a], [r["id"]]) for a in ARMS} for r in selected}
    examples = {
        key: sorted(
            [r["id"] for r in rs],
            key=lambda i: (
                counts[i]["routed"]["TP50"] - counts[i]["skip"]["TP50"],
                counts[i]["skip"]["FP50"] - counts[i]["routed"]["FP50"],
                i,
            ),
        )[:2]
        for key, rs in subgroups.items()
    }
    freeze_json(report / "errors.json", {"counts": counts, "examples": examples})
    freeze_json(report / "uncertainty.json", interval)
    if file_hash(checkpoint) != PRETRAINED_SHA256:
        raise ValueError("Checkpoint changed during pilot")
    summary = {
        "plan_sha256": file_hash(report / "plan.json"),
        "population": "pretrained natural calibration exploration; not trained S20",
        "metrics": metrics,
        "native": native,
        "runtime": runtime,
        "routing": dict(Counter(routing.values())),
        "timing_threads": 8,
        "evaluation_wall_seconds": wall,
        "wall_seconds": time.perf_counter() - start,
        "decision": assess(metrics, runtime, interval),
        "source_reads": source_reads,
        "exports_sha256": digest(exports),
        "statistics_sha256": {a: file_hash(output / a / "eval/paired_stats.npz") for a in ARMS},
        "confirmation_accessed": False,
        "development_accessed": False,
        "final_test_accessed": False,
    }
    freeze_json(report / "summary.json", summary)
    freeze_json(
        report / "evidence.json",
        {p.name: file_hash(p) for p in sorted(report.iterdir()) if p.is_file()},
    )
    return summary
