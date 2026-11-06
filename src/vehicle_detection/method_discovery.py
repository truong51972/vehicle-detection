"""Saved-error diagnosis and one consumed-population resolution diagnostic."""

import subprocess
import time
from collections import Counter
from pathlib import Path
from shutil import copyfile

import numpy as np
import pandas as pd
import yaml

from .common import digest, file_hash, freeze_json, read_json
from .experiment_data import source_rows
from .five_class import FIVE_CLASSES
from .v2_pilot import metric_record
from .v3_pilot import route
from .v3_pilot import run as read_pilot


def read_report(root):
    report = Path(root) / "docs/issue47-discovery"
    for name, expected in read_json(report / "evidence.json").items():
        if Path(name).name != name or file_hash(report / name) != expected:
            raise ValueError("Discovery evidence changed")
    return read_json(report / "diagnosis.json")


def cohorts(rows, features):
    """Image cohorts keep every original GT; these are never size-specific AP."""
    return {
        "all": rows,
        "dark": [r for r in rows if features[r["id"]]["brightness"] < 0.12],
        "bright": [r for r in rows if features[r["id"]]["brightness"] >= 0.12],
        "low_contrast": [r for r in rows if features[r["id"]]["contrast"] < 0.20],
        "higher_noise": [r for r in rows if features[r["id"]]["noise"] > 0.01],
        "low_sharpness": [r for r in rows if features[r["id"]]["sharpness"] < 50],
        "small_two_wheeler": [
            r
            for r in rows
            if any(
                c in (3, 4) and (x2 - x1) * (y2 - y1) < 32**2 for c, x1, y1, x2, y2 in r["boxes"]
            )
        ],
        **{
            f"route_{name}": [r for r in rows if route(features[r["id"]]) == name]
            for name in ("exposure", "detail", "both", "skip")
        },
    }


def outcome(stats, rows):
    """Class FN totals follow native one-to-one TP counts; no GT-box match is inferred."""
    if not {r["id"] for r in rows} <= stats.keys():
        raise ValueError("Missing paired statistics")
    record = metric_record(stats, rows)
    for c, name in enumerate(FIVE_CLASSES):
        gt = sum(b[0] == c for r in rows for b in r["boxes"])
        if gt != sum(int((stats[r["id"]]["target_cls"] == c).sum()) for r in rows):
            raise ValueError("Original GT and native class support differ")
        tp = predictions = 0
        for row in rows:
            values = stats[row["id"]]
            keep = (values["conf"] >= 0.25) & (values["pred_cls"] == c)
            tp += int(values["tp"][keep, 0].sum())
            predictions += int(keep.sum())
        if tp > gt:
            raise ValueError("Native TP exceeds original class GT")
        # Absent-GT classes still have auditable FP counts, but never receive zero AP.
        record.setdefault("counts", {})[name] = {
            "GT": gt,
            "TP50": tp,
            "FP50": predictions - tp,
            "FN50": gt - tp,
        }
    return record


def size_support(rows, features):
    result = {}
    for c, name in enumerate(FIVE_CLASSES):
        boxes = [(r, b) for r in rows for b in r["boxes"] if b[0] == c]
        presence = [r for r in rows if any(b[0] == c for b in r["boxes"])]
        edges = np.array(
            [
                np.sqrt((b[3] - b[1]) * (b[4] - b[2])) * 640 / max(r["width"], r["height"])
                for r, b in boxes
            ]
        )
        result[name] = {
            "instances": len(boxes),
            "images": len(presence),
            "source_groups": len({r["source_group"] for r in presence}),
            "median_sqrt_area_at_640": float(np.median(edges)) if len(edges) else None,
            "below_16_pixels_at_640": int((edges < 16).sum()),
            "below_8_pixels_at_640": int((edges < 8).sum()),
            "dark_images": sum(features[r["id"]]["brightness"] < 0.12 for r in presence),
            "routes": dict(Counter(route(features[r["id"]]) for r in presence)),
        }
    return result


def assess_resolution(result):
    before, after = (result["metrics"][s] for s in ("640", "960"))
    reasons = []
    if not result["baseline_reproduced"]:
        reasons.append("Historical 640 control did not reproduce")
    if after["all"]["mAP50_95"] - before["all"]["mAP50_95"] < 0.002:
        reasons.append("Primary gain below 0.2 pp")
    classes = after["all"]["classes"]
    if any(classes[c]["source_groups"] < 5 for c in ("motorcycle", "bicycle")):
        reasons.append("Insufficient two-wheeler support")
    elif (
        np.mean(
            [
                classes[c]["AP50_95"] - before["all"]["classes"][c]["AP50_95"]
                for c in ("motorcycle", "bicycle")
            ]
        )
        < 0.005
    ):
        reasons.append("Two-wheeler mean gain below 0.5 pp")
    interval = result["paired_interval"]
    if interval["estimand"]["class_ids"] != list(range(5)) or interval["recovery_ci_low"] <= 0:
        reasons.append("Five-class primary interval includes zero or lacks support")
    for key in ("all", "dark", "bright"):
        for name, record in after[key]["classes"].items():
            if record["source_groups"] >= 5 and record["AP50_95"] - before[key]["classes"][name][
                "AP50_95"
            ] < (-0.02 if key == "all" else -0.01):
                reasons.append(f"Supported class regression: {key}/{name}")
    costs = result["timings"]
    if (
        costs["960"]["native_ms_per_image"]["inference"]
        > 3 * costs["640"]["native_ms_per_image"]["inference"]
    ):
        reasons.append("Native mean inference cost exceeds 3x")
    return {
        "actionable_resolution_signal": not reasons,
        "reasons": reasons,
        "independent_confirmation": False,
        "preprocessing_policy_selected": False,
    }


def _inputs(root):
    """Metadata and verified saved matches only; no source pixels are needed."""
    old = read_pilot(root)
    report = root / "docs/issue47"
    membership = pd.read_csv(report / "membership.csv")
    ids = set(membership.id)
    _, annotated = source_rows(root)
    rows = sorted([r for r in annotated if r["id"] in ids], key=lambda r: r["id"])
    plan = read_json(report / "plan.json")
    if digest(rows) != plan["rows_sha256"]:
        raise ValueError("Original discovery population changed")
    return old, rows, read_json(report / "features.json")


def build_diagnosis(root):
    """Explicit acquisition of a new compact analysis; never changes pilot evidence."""
    root = Path(root).resolve()
    report = root / "docs/issue47-discovery"
    if (report / "diagnosis.json").exists():
        raise FileExistsError("Diagnosis exists; use committed evidence")
    old, rows, features = _inputs(root)
    from .data import scan_bdd
    from .evaluation_stats import load_stats

    shared = yaml.safe_load((root / "configs/data.yaml").read_text())
    records, _, _ = scan_bdd(root / shared["data_root"], strict=False)
    attributes = {r["id"]: r["attributes"] for r in records}
    groups = cohorts(rows, features)
    for key in ("timeofday", "weather", "scene"):
        for value in sorted({attributes[r["id"]].get(key, "unknown") for r in rows}):
            groups[f"{key}:{value}"] = [
                r for r in rows if attributes[r["id"]].get(key, "unknown") == value
            ]
    measurements = {}
    for arm, expected in old["statistics_sha256"].items():
        path = root / f"runs/issue47/pilot/{arm}/eval/paired_stats.npz"
        if file_hash(path) != expected:
            raise ValueError("Historical matching cache changed")
        stats = load_stats(path)
        if set(stats) != {r["id"] for r in rows}:
            raise ValueError("Historical statistics IDs changed")
        measurements[arm] = {name: outcome(stats, rs) for name, rs in groups.items()}
    result = {
        "population": "Consumed PR #48 calibration exploration; descriptive discovery only",
        "ids_sha256": digest([r["id"] for r in rows]),
        "pilot_summary_sha256": file_hash(root / "docs/issue47/summary.json"),
        "size_support": size_support(rows, features),
        "cohorts": measurements,
        "individual_missed_gt_box_sizes_available": False,
        "limit": "Cohort errors do not identify individual missed GT boxes or establish cause",
    }
    freeze_json(report / "diagnosis.json", result)
    return result


def resolution(root, *, allow_inference=False):
    """Exactly two fresh native evaluations on copied, verified consumed Skip exports."""
    root = Path(root).resolve()
    report, output = root / "docs/issue47-discovery", root / "runs/issue47-discovery/resolution"
    if not allow_inference:
        read_report(root)
        return read_json(report / "resolution.json")
    if output.exists() or output.is_symlink() or (report / "resolution.json").exists():
        raise FileExistsError("Resolution attempt exists; never overwrite it")
    protected = ["docs/ISSUE47_DISCOVERY_PLAN.md", "src/vehicle_detection/method_discovery.py"]
    for name in protected:
        if (
            subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=root)
            != (root / name).read_bytes()
        ):
            raise ValueError("Commit the diagnostic plan and implementation before inference")
    old, rows, features = _inputs(root)
    if len(rows) != 1027:
        raise ValueError("Use exactly the consumed 1027-image population")
    import torch
    from ultralytics import YOLO

    from .baseline import PRETRAINED_SHA256, verify_coco_names
    from .evaluation_stats import load_stats, paired_interval
    from .stress_eval import CalibrationValidator

    checkpoint = root / "weights/yolo26n.pt"
    if file_hash(checkpoint) != PRETRAINED_SHA256:
        raise ValueError("Pinned checkpoint changed")
    acquisition = read_json(root / "docs/issue47/plan.json")
    plan = {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "code_sha256": file_hash(__file__),
        "declaration_sha256": file_hash(root / protected[0]),
        "diagnosis_sha256": file_hash(report / "diagnosis.json"),
        "ids_sha256": digest([r["id"] for r in rows]),
        "checkpoint_sha256": PRETRAINED_SHA256,
        "settings": acquisition["settings"],
        "sizes": [640, 960],
        "maximum_image_passes": 2054,
        "independent_confirmation": False,
    }
    freeze_json(output / "plan.json", plan)
    freeze_json(report / "resolution_plan.json", plan)
    dataset = output / "data"
    entries = read_json(root / "runs/issue47/pilot/exports.json")
    if digest(entries) != old["exports_sha256"]:
        raise ValueError("Historical export identities changed")
    selected = [entry for entry in entries if entry[0].startswith("skip/")]
    if len(selected) != len(rows) or {Path(e[0]).stem for e in selected} != {r["id"] for r in rows}:
        raise ValueError("Consumed export membership changed")
    for name, image_hash, label_hash in selected:
        label_name = name.replace("images/", "labels/").replace(".jpg", ".txt")
        for relative, expected in ((name, image_hash), (label_name, label_hash)):
            source = root / "runs/issue47/pilot" / relative
            target = dataset / Path(relative).relative_to("skip")
            if (
                Path(relative).is_absolute()
                or ".." in Path(relative).parts
                or file_hash(source) != expected
            ):
                raise ValueError("Historical export bytes or path changed")
            target.parent.mkdir(parents=True, exist_ok=True)
            copyfile(source, target)
            if file_hash(target) != expected:
                raise ValueError("Copied export changed")
    data_yaml = dataset / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(dataset),
                "train": "images/val",
                "val": "images/val",
                "names": dict(enumerate(FIVE_CLASSES)),
            }
        )
    )
    stats, metrics, native, timings = {}, {}, {}, {}
    expected_ids = {r["id"] for r in rows}

    def check_loader(validator):
        paths = validator.dataloader.dataset.im_files
        if len(paths) != len(rows) or {Path(p).stem for p in paths} != expected_ids:
            raise ValueError("Diagnostic loader changed IDs")

    def check_mode(validator):
        if not validator.end2end or validator.args.nms is not False:
            raise ValueError("NMS-free native contract changed")

    batch_times = []
    batch_start = 0

    def start_batch(validator):
        nonlocal batch_start
        torch.cuda.synchronize()
        batch_start = time.perf_counter()

    def end_batch(validator):
        torch.cuda.synchronize()
        batch_times.append((time.perf_counter() - batch_start) * 1000)

    for size in plan["sizes"]:
        model = YOLO(str(checkpoint))
        verify_coco_names(model.names)
        model.add_callback("on_val_start", check_loader)
        model.add_callback("on_val_end", check_mode)
        batch_times.clear()
        batch_start = 0

        model.add_callback("on_val_batch_start", start_batch)
        model.add_callback("on_val_batch_end", end_batch)
        tick = time.perf_counter()
        result = model.val(
            validator=CalibrationValidator,
            data=str(data_yaml),
            split="val",
            project=str(output),
            name=str(size),
            exist_ok=True,
            **{**plan["settings"], "imgsz": size},
        )
        wall = time.perf_counter() - tick
        path = output / str(size) / "paired_stats.npz"
        stats[size] = load_stats(path)
        if set(stats[size]) != expected_ids:
            raise ValueError("Diagnostic statistics IDs changed")
        metrics[str(size)] = {
            name: outcome(stats[size], rs) for name, rs in cohorts(rows, features).items()
        }
        native[str(size)] = dict(
            zip(
                ("precision", "recall", "mAP50", "mAP50_95"),
                map(float, result.mean_results()),
                strict=True,
            )
        )
        if not np.isfinite(list(native[str(size)].values())).all() or not np.isclose(
            native[str(size)]["mAP50_95"], metrics[str(size)]["all"]["mAP50_95"]
        ):
            raise ValueError("Native AP disagrees with saved matches")
        timings[str(size)] = {
            "native_ms_per_image": result.speed,
            "wall_seconds": wall,
            "batch_mean_ms": float(np.mean(batch_times)),
            "batch_p95_ms": float(np.quantile(batch_times, 0.95)),
            "batches": len(batch_times),
        }
    if any(
        not np.array_equal(stats[640][i]["target_cls"], stats[960][i]["target_cls"])
        for i in expected_ids
    ):
        raise ValueError("Diagnostic paired GT changed")
    bootstrap_rows = [{**r, "group": r["source_group"]} for r in rows]
    interval = paired_interval(
        stats[640], stats[960], {"rows": bootstrap_rows}, class_names=FIVE_CLASSES
    )
    if file_hash(checkpoint) != PRETRAINED_SHA256:
        raise ValueError("Checkpoint changed during diagnostic")
    result = {
        "population": "Consumed calibration exploration; resolution diagnostic only",
        "metrics": metrics,
        "native": native,
        "timings": timings,
        "paired_interval": interval,
        "baseline_reproduced": all(
            np.isclose(native["640"][k], v, atol=1e-6, rtol=0)
            for k, v in old["native"]["skip"].items()
        ),
        "statistics_sha256": {
            str(s): file_hash(output / str(s) / "paired_stats.npz") for s in plan["sizes"]
        },
        "confirmation_development_final_pixels_accessed": False,
        "policy_selected": False,
    }
    result["decision"] = assess_resolution(result)
    freeze_json(report / "resolution.json", result)
    return result
