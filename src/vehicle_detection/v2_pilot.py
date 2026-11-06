"""One preregistered local-shadow pilot; no search, confirmation or training."""

import importlib.metadata as md
import platform
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from .common import digest, file_hash, freeze_json, read_image, read_json
from .datasets import verify_label
from .experiment_data import load_experiment, shared_splits, source_rows
from .five_class import FIVE_CLASSES
from .preprocess import apply_recipe, local_shadow, quality

ARMS = ("raw", "gamma_or_skip", "local_shadow")


def require_dark_support(rows, features):
    """Stop future pilots before inference when the declared dark gate is untestable."""
    dark = [r for r in rows if features[r["id"]]["brightness"] < 0.12]
    groups = {
        name: len({r["source_group"] for r in dark if any(b[0] == c for b in r["boxes"])})
        for c, name in enumerate(FIVE_CLASSES)
    }
    if min(groups.values()) < 5:
        raise ValueError(f"Insufficient dark class support before inference: {groups}")
    return groups


def select_sample(rows, manifest, consumed):
    """Continue the historical whole-group exploration order without touching other roles."""
    if (
        manifest["manifest_id"] != digest({k: v for k, v in manifest.items() if k != "manifest_id"})
        or manifest["seed"] != 2026
    ):
        raise ValueError("Exploration manifest identity changed")
    allowed = set(manifest["sets"]["calibration_exploration"])
    calibration = {r["id"]: r for r in rows if r["role"] == "calibration"}
    confirmation = set(manifest["sets"]["calibration_confirmation"])
    if allowed & confirmation or allowed | confirmation != set(calibration):
        raise ValueError("Calibration role partition changed")
    groups = {}
    for row in calibration.values():
        if row["id"] in allowed:
            groups.setdefault(row["group"], []).append(row)
    used_ids = set(consumed.id)
    if not used_ids <= allowed or len(consumed) != len(used_ids):
        raise ValueError("Consumed exploration membership changed")
    for record in consumed.to_dict("records"):
        if any(record[k] != calibration[record["id"]][k] for k in ("group", "source_group")):
            raise ValueError("Consumed exploration groups changed")
    forbidden = {
        (key, calibration[i][key])
        for i in confirmation
        for key in ("group", "source_group", "sha256")
    }
    keys = ("group", "source_group", "sha256")
    if any((k, r[k]) in forbidden for rs in groups.values() for r in rs for k in keys):
        raise ValueError("Exploration leaks confirmation groups")
    used = set(consumed.group)
    selected = []
    for group in sorted(groups, key=lambda g: (digest([2026, "calibration_exploration", g]), g)):
        if group not in used:
            selected.extend(groups[group])
            if len(selected) >= 256:
                break
    if not 256 <= len(selected) <= 272:
        raise ValueError("Pilot exceeds the declared whole-group sample budget")
    consumed_tokens = {(k, calibration[i][k]) for i in used_ids for k in keys}
    if any((k, r[k]) in consumed_tokens for r in selected for k in keys):
        raise ValueError("Pilot overlaps consumed exploration groups")
    return sorted(selected, key=lambda r: r["id"])


def metric_record(stats, rows):
    """Native dataset AP on all GT in these images; absent classes stay absent."""
    from .evaluation_stats import class_ap

    classes = class_ap(stats, [r["id"] for r in rows], detailed=True)
    return {
        "images": len(rows),
        "mAP50_95": float(np.mean([v["AP50_95"] for v in classes.values()])) if classes else None,
        "mAP50": float(np.mean([v["AP50"] for v in classes.values()])) if classes else None,
        "classes": {
            FIVE_CLASSES[c]: {
                **values,
                "source_groups": len(
                    {r["source_group"] for r in rows if any(b[0] == c for b in r["boxes"])}
                ),
                "instances": sum(b[0] == c for r in rows for b in r["boxes"]),
            }
            for c, values in classes.items()
        },
    }


def assess(metrics, runtime, intervals):
    """Apply the written gates once; unsupported dark evidence cannot pass."""
    reasons = []
    candidate = metrics["local_shadow"]
    for control in ARMS[:2]:
        before = metrics[control]
        if candidate["all"]["mAP50_95"] - before["all"]["mAP50_95"] < 0.002:
            reasons.append(f"Primary gain below 0.2 pp vs {control}")
        dark = candidate["dark"]
        if set(dark["classes"]) != set(FIVE_CLASSES) or any(
            v["source_groups"] < 5 for v in dark["classes"].values()
        ):
            reasons.append(f"Insufficient five-class dark support vs {control}")
        elif dark["mAP50_95"] - before["dark"]["mAP50_95"] < 0.005:
            reasons.append(f"Dark gain below 0.5 pp vs {control}")
        for subgroup, record in candidate.items():
            for cls, values in record["classes"].items():
                if values["source_groups"] >= 5:
                    limit = 0.02 if subgroup == "all" else 0.01
                    if values["AP50_95"] - before[subgroup]["classes"][cls]["AP50_95"] < -limit:
                        reasons.append(f"Class regression vs {control}: {subgroup}/{cls}")
        interval = intervals[control]
        if interval["estimand"]["class_ids"] != list(range(5)):
            reasons.append(f"Insufficient primary bootstrap support vs {control}")
        if interval["recovery_ci_low"] <= 0:
            reasons.append(f"Primary paired interval includes zero vs {control}")
    if runtime["local_shadow"]["mean_ms"] > 10 or runtime["local_shadow"]["p95_ms"] > 25:
        reasons.append("Preprocessing cost exceeds mean 10 ms / p95 25 ms")
    return {"advance": not reasons, "reasons": reasons, "v2_trained": False}


def run(root, *, allow_inference=False):
    """Explicit opt-in runs exactly three arms once; default reads compact evidence only."""
    root = Path(root).resolve()
    report = root / "docs/issue45"
    if not allow_inference:
        evidence = read_json(report / "evidence.json")
        for name, expected in evidence.items():
            if Path(name).name != name or file_hash(report / name) != expected:
                raise ValueError("Pilot evidence changed")
        return read_json(report / "summary.json")

    output = root / "runs/issue45/pilot"
    if output.exists() or output.is_symlink():
        raise FileExistsError("Pilot output exists; inspect it rather than overwriting evidence")

    import torch
    from ultralytics import YOLO

    from .baseline import PRETRAINED_SHA256, verify_coco_names
    from .evaluation_stats import load_stats, paired_interval
    from .research_round3 import exploration_access
    from .research_round3_report import errors
    from .stress_eval import SETTINGS, CalibrationValidator

    start = time.perf_counter()
    shared, rows = shared_splits(root)
    manifest = read_json(root / "docs/round2-evaluation-v2/evaluation_sets.json")
    if manifest["protocol_id"] != shared["source_protocol_id"]:
        raise ValueError("Exploration belongs to another protocol")
    consumed_path = root / "docs/round2-research-v3/membership.csv.gz"
    prior_evidence = read_json(root / "docs/round2-research-v3/evidence.json")
    if (
        file_hash(consumed_path)
        != prior_evidence["artifacts"]["docs/round2-research-v3/membership.csv.gz"]
    ):
        raise ValueError("Historical exploration membership changed")
    consumed = pd.read_csv(consumed_path)
    selected = select_sample(rows, manifest, consumed)
    _, annotated = source_rows(root)
    lookup = {r["id"]: r for r in annotated}
    selected = [lookup[r["id"]] for r in selected]
    if any(
        len({r["source_group"] for r in selected if any(b[0] == c for b in r["boxes"])}) < 5
        for c in range(5)
    ):
        raise ValueError("Insufficient primary five-class support; no inference authorized")
    checkpoint = root / "weights/yolo26n.pt"
    if file_hash(checkpoint) != PRETRAINED_SHA256:
        raise ValueError("Pinned pretrained checkpoint changed")
    configs = [load_experiment(root, v) for v in ("v0", "v1")]
    operators = {
        name: (lambda image, c=c: apply_recipe(image, c["variant"], c["recipes"])[0])
        for name, c in zip(ARMS[:2], configs, strict=True)
    }
    operators["local_shadow"] = local_shadow
    plan = {
        "protocol_id": shared["source_protocol_id"],
        "splits_sha256": shared["splits_sha256"],
        "manifest_sha256": file_hash(root / "docs/round2-evaluation-v2/evaluation_sets.json"),
        "consumed_sha256": file_hash(consumed_path),
        "declaration_sha256": file_hash(root / "docs/ISSUE45_PILOT.md"),
        "rows_sha256": digest(selected),
        "ids_sha256": digest([r["id"] for r in selected]),
        "checkpoint_sha256": PRETRAINED_SHA256,
        "controls": configs,
        "code": {
            name: file_hash(Path(__file__).with_name(name))
            for name in ("v2_pilot.py", "preprocess.py", "baseline.py", "evaluation_stats.py")
        },
        "settings": {**SETTINGS, "half": False},
        "versions": {p: md.version(p) for p in ("ultralytics", "torch", "opencv-python", "numpy")},
        "hardware": {
            "platform": platform.platform(),
            "gpu": torch.cuda.get_device_name(0),
            "cuda_runtime": torch.version.cuda,
        },
        "role": "calibration_exploration",
        "maximum_image_passes": 816,
    }
    freeze_json(output / "plan.json", plan)
    report.mkdir(parents=True, exist_ok=True)
    freeze_json(report / "plan.json", plan)
    pd.DataFrame(selected).drop(columns="boxes").to_csv(report / "membership.csv", index=False)
    data = (root / shared["data_root"]).resolve()
    if output.is_relative_to(data) or data.is_relative_to(output):
        raise ValueError("Pilot output overlaps original sources")
    timings = {name: [] for name in ARMS}
    features, exported = {}, []
    prior_threads = cv2.getNumThreads()
    try:
        cv2.setNumThreads(8)
        for name in ARMS:
            for kind in ("images", "labels"):
                (output / name / kind / "val").mkdir(parents=True)
        with exploration_access(data, {"rows": selected}) as reads:
            for index, row in enumerate(selected):
                source = data / row["source_path"]
                if source.resolve() != source or not source.is_relative_to(data):
                    raise ValueError("Unsafe source path")
                if file_hash(source) != row["sha256"]:
                    raise ValueError("Original source changed")
                image = read_image(source)
                if image.shape != (row["height"], row["width"], 3):
                    raise ValueError("Original geometry changed")
                features[row["id"]] = quality(image)
                if index == 0:
                    for operator in operators.values():
                        operator(image)
                for repeat in range(3):
                    offset = (index + repeat) % 3
                    for name in ARMS[offset:] + ARMS[:offset]:
                        tick = time.perf_counter()
                        operators[name](image)
                        timings[name].append((time.perf_counter() - tick) * 1000)
                labels = []
                for name, operator in operators.items():
                    target = output / name / "images/val" / f"{row['id']}.jpg"
                    if not cv2.imwrite(
                        str(target), operator(image), [cv2.IMWRITE_JPEG_QUALITY, 95]
                    ):
                        raise OSError(f"Could not encode {target}")
                    if read_image(target).shape != image.shape:
                        raise ValueError("Exported geometry changed")
                    label = output / name / "labels/val" / f"{row['id']}.txt"
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
                    exported.append(
                        (str(target.relative_to(output)), file_hash(target), file_hash(label))
                    )
                if len(set(labels)) != 1:
                    raise ValueError("Paired label bytes changed")
            source_reads = sorted(reads)
    finally:
        cv2.setNumThreads(prior_threads)
    freeze_json(output / "exports.json", exported)
    freeze_json(report / "features.json", features)
    require_dark_support(selected, features)
    runtime = {
        name: {"mean_ms": float(np.mean(v)), "p95_ms": float(np.quantile(v, 0.95)), "calls": len(v)}
        for name, v in timings.items()
    }
    subgroups = {
        "all": selected,
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
    stats, metrics, native, wall = {}, {}, {}, {}
    expected = {r["id"] for r in selected}

    def check_loader(validator):
        files = validator.dataloader.dataset.im_files
        if len(files) != len(expected) or {Path(p).stem for p in files} != expected:
            raise ValueError("Native loader changed paired IDs")

    def check_mode(validator):
        if not validator.end2end or validator.args.nms is not False:
            raise ValueError("Native NMS-free contract changed")

    for name in ARMS:
        directory = output / name
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
        wall[name] = time.perf_counter() - tick
        stats[name] = load_stats(directory / "eval/paired_stats.npz")
        if set(stats[name]) != expected:
            raise ValueError("Native statistics changed paired IDs")
        if any(
            not np.array_equal(stats["raw"][i]["target_cls"], stats[name][i]["target_cls"])
            for i in expected
        ):
            raise ValueError("Native paired ground truth changed")
        metrics[name] = {key: metric_record(stats[name], rs) for key, rs in subgroups.items()}
        native[name] = dict(
            zip(
                ("precision", "recall", "mAP50", "mAP50_95"),
                map(float, result.mean_results()),
                strict=True,
            )
        )
        if not np.isfinite(list(native[name].values())).all():
            raise ValueError("Nonfinite native metrics")
        if not np.isclose(native[name]["mAP50_95"], metrics[name]["all"]["mAP50_95"]):
            raise ValueError("Native and saved-match AP disagree")
    intervals = {
        name: paired_interval(
            stats[name], stats["local_shadow"], {"rows": selected}, class_names=FIVE_CLASSES
        )
        for name in ARMS[:2]
    }
    if file_hash(checkpoint) != PRETRAINED_SHA256:
        raise ValueError("Checkpoint changed during pilot")
    error_counts = {
        r["id"]: {name: errors(stats[name], [r["id"]]) for name in ARMS} for r in selected
    }
    examples = {}
    for key, rs in subgroups.items():
        examples[key] = sorted(
            [r["id"] for r in rs],
            key=lambda i: (
                error_counts[i]["local_shadow"]["TP50"] - error_counts[i]["raw"]["TP50"],
                error_counts[i]["raw"]["FP50"] - error_counts[i]["local_shadow"]["FP50"],
                i,
            ),
        )[:2]
    freeze_json(report / "errors.json", {"counts": error_counts, "examples": examples})
    _plot(report, output, selected, metrics, runtime, examples, error_counts)
    summary = {
        "plan_sha256": file_hash(report / "plan.json"),
        "metrics": metrics,
        "native": native,
        "runtime": runtime,
        "timing_threads": 8,
        "evaluation_wall_seconds": wall,
        "wall_seconds": time.perf_counter() - start,
        "decision": assess(metrics, runtime, intervals),
        "source_reads": source_reads,
        "exports_sha256": digest(exported),
        "statistics_sha256": {
            name: file_hash(output / name / "eval/paired_stats.npz") for name in ARMS
        },
        "final_test_accessed": False,
        "confirmation_accessed": False,
    }
    freeze_json(report / "uncertainty.json", intervals)
    freeze_json(report / "summary.json", summary)
    freeze_json(
        report / "evidence.json",
        {p.name: file_hash(p) for p in sorted(report.iterdir()) if p.is_file()},
    )
    return summary


def _plot(report, output, rows, metrics, runtime, examples, counts):
    """Scientific comparison and original GT frames; counts are not image-level AP."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    x = np.arange(5)
    for index, name in enumerate(ARMS):
        axes[0].bar(
            x + (index - 1) * 0.25,
            [metrics[name]["all"]["classes"][c]["AP50_95"] * 100 for c in FIVE_CLASSES],
            width=0.25,
            label=name,
        )
    axes[0].set(
        xticks=x,
        xticklabels=FIVE_CLASSES,
        ylabel="Native AP50:95 (%)",
        title="Natural calibration exploration; pretrained, paired",
    )
    axes[0].legend()
    axes[1].bar(ARMS, [runtime[n]["mean_ms"] for n in ARMS])
    axes[1].scatter(ARMS, [runtime[n]["p95_ms"] for n in ARMS], color="black", label="p95")
    axes[1].set(ylabel="CPU preprocessing (ms)", title="Matched 8-thread timing; JPEG/I/O excluded")
    axes[1].legend()
    fig.savefig(report / "pilot.png", dpi=140)
    plt.close(fig)
    lookup = {r["id"]: r for r in rows}
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for axis, key in zip(
        axes.flat, ("dark", "bright", "blurred", "small_two_wheeler"), strict=True
    ):
        if not examples[key]:
            axis.set_title(f"{key}: no sampled images")
        else:
            sample_id = examples[key][0]
            image = read_image(output / "raw/images/val" / f"{sample_id}.jpg")
            axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            for c, x1, y1, x2, y2 in lookup[sample_id]["boxes"]:
                axis.add_patch(
                    Rectangle(
                        (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="orange", linewidth=0.7
                    )
                )
                if c in (3, 4):
                    axis.text(x1, y1, FIVE_CLASSES[c], color="orange", fontsize=7)
            pair = counts[sample_id]
            axis.set_title(
                f"{key}: {sample_id}\nTP/FP Raw {pair['raw']['TP50']}/{pair['raw']['FP50']}"
                f" → local {pair['local_shadow']['TP50']}/{pair['local_shadow']['FP50']}"
            )
        axis.axis("off")
    fig.suptitle("Illustrative conf .25 / IoU .50 changes; orange = GT; no per-image AP")
    fig.savefig(report / "errors.png", dpi=120)
    plt.close(fig)
