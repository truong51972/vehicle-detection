"""Fixed three-arm S20 development diagnostics; no training, search or final-test API."""

import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .common import digest, file_hash, freeze_json, read_image, read_json
from .experiment_data import load_experiment, source_rows
from .five_class import FIVE_CLASSES
from .preprocess import apply_recipe, quality
from .v2_pilot import metric_record

ARMS = ("v0", "v1", "v2")


def selected_epoch(history, checkpoint_metrics):
    """Authenticate selection against saved best-checkpoint metrics, not stripped epoch=-1."""
    keys = [k for k in checkpoint_metrics if k in history.columns]
    if "metrics/mAP50-95(B)" not in keys:
        raise ValueError("Missing checkpoint selection metrics")
    matching = history[
        np.isclose(
            history[keys].to_numpy(), [checkpoint_metrics[k] for k in keys], atol=5e-6, rtol=0
        ).all(axis=1)
    ]
    if len(matching) != 1:
        raise ValueError("Selected checkpoint does not identify one native history row")
    return int(matching.iloc[0].epoch)


def run(root, *, allow_inference=False):
    """Replay compact evidence by default; opt-in permits the two predeclared control replays."""
    root = Path(root).resolve()
    report = root / "docs/issue45-s20"
    if not allow_inference:
        for name, expected in read_json(report / "evidence.json").items():
            if Path(name).name != name or file_hash(report / name) != expected:
                raise ValueError("S20 evidence changed")
        return read_json(report / "summary.json")
    output = root / "runs/issue45-s20/diagnostics"
    if output.exists() or output.is_symlink():
        raise FileExistsError("Diagnostic attempt exists; inspect it before any further inference")

    from ultralytics import YOLO

    from .evaluation import ExperimentValidator, evaluate_experiment
    from .evaluation_stats import load_stats
    from .research_round3 import exploration_access
    from .research_round3_report import errors

    start = time.perf_counter()
    shared, rows = source_rows(root)
    dev = [r for r in rows if r["role"] == "development_validation"]
    if len(dev) != 13975 or shared["training_subset"] != "S20":
        raise ValueError("Expected the frozen full S20 development population")
    pilot_ids = set(pd.read_csv(root / "docs/issue45/membership.csv").id)
    timing_rows = [r for r in rows if r["id"] in pilot_ids]
    if len(timing_rows) != 256 or any(r["role"] != "calibration" for r in timing_rows):
        raise ValueError("Timing may only reuse the consumed 256-image pilot")
    expected = {r["id"] for r in dev}
    output.mkdir(parents=True)
    report.mkdir(parents=True, exist_ok=True)
    data = (root / shared["data_root"]).resolve()
    features, timings = {}, {arm: [] for arm in ARMS}
    configs = {arm: load_experiment(root, arm) for arm in ARMS}
    original_threads = cv2.getNumThreads()
    try:
        cv2.setNumThreads(8)
        with exploration_access(data, {"rows": dev + timing_rows}) as reads:
            for row in dev + timing_rows:
                source = data / row["source_path"]
                if source.resolve() != source or not source.is_relative_to(data):
                    raise ValueError("Unsafe original source path")
                if file_hash(source) != row["sha256"]:
                    raise ValueError("Original source changed")
                image = read_image(source)
                if image.shape != (row["height"], row["width"], 3):
                    raise ValueError("Original geometry changed")
                if row["role"] == "development_validation":
                    features[row["id"]] = quality(image)
                else:
                    index = len(timings["v0"]) // 3
                    if index == 0:
                        for c in configs.values():
                            apply_recipe(image, c["variant"], c["recipes"])
                    for repeat in range(3):
                        offset = (index + repeat) % 3
                        for arm in ARMS[offset:] + ARMS[:offset]:
                            c = configs[arm]
                            tick = time.perf_counter()
                            apply_recipe(image, c["variant"], c["recipes"])
                            timings[arm].append((time.perf_counter() - tick) * 1000)
            source_reads = sorted(reads)
    finally:
        cv2.setNumThreads(original_threads)
    subgroups = {
        "all": dev,
        "dark": [r for r in dev if features[r["id"]]["brightness"] < 0.12],
        "bright": [r for r in dev if features[r["id"]]["brightness"] >= 0.12],
        "blurred": [r for r in dev if features[r["id"]]["sharpness"] < 50],
        "small_two_wheeler": [
            r
            for r in dev
            if any(
                c in (3, 4) and (x2 - x1) * (y2 - y1) < 32**2 for c, x1, y1, x2, y2 in r["boxes"]
            )
        ],
    }
    # Support is saved before diagnostic GPU inference; no post-outcome resampling.
    freeze_json(
        report / "support.json",
        {
            key: {
                "images": len(rs),
                "class_source_groups": {
                    name: len({r["source_group"] for r in rs if any(b[0] == c for b in r["boxes"])})
                    for c, name in enumerate(FIVE_CLASSES)
                },
            }
            for key, rs in subgroups.items()
        },
    )
    freeze_json(output / "features.json", features)
    freeze_json(output / "rows.json", dev)
    records, stats, diagnostic_seconds, artifacts = {}, {}, {}, {}

    def check_loader(validator):
        files = validator.dataloader.dataset.im_files
        if len(files) != len(expected) or {Path(p).stem for p in files} != expected:
            raise ValueError("Diagnostic loader changed paired development IDs")

    for arm in ARMS:
        native = evaluate_experiment(root, arm, settings={"workers": 0})
        directory = root / f"runs/{arm}"
        checkpoint = directory / "train/weights/best.pt"
        model = YOLO(str(checkpoint))
        history = pd.read_csv(directory / "train/results.csv")
        chosen = selected_epoch(history, model.ckpt["train_metrics"])
        if arm == "v2":
            stats_path = directory / "eval/paired_stats.npz"
        else:
            identity = {**native["inputs"], "report_sha256": file_hash(__file__)}
            freeze_json(output / arm / "identity.json", identity)
            model.add_callback("on_val_start", check_loader)
            tick = time.perf_counter()
            metrics = model.val(
                validator=ExperimentValidator,
                data=str(directory / "data/data.yaml"),
                project=str(output),
                name=arm,
                exist_ok=True,
                split="val",
                **native["inputs"]["settings"],
            )
            diagnostic_seconds[arm] = time.perf_counter() - tick
            if not np.allclose(
                metrics.mean_results(),
                [native["overall"][k] for k in ("precision", "recall", "mAP50", "mAP50_95")],
                atol=1e-6,
                rtol=0,
            ):
                raise ValueError("Diagnostic replay disagrees with original native metrics")
            stats_path = output / arm / "paired_stats.npz"
        stats[arm] = load_stats(stats_path)
        if set(stats[arm]) != expected or any(
            not np.array_equal(stats["v0"][i]["target_cls"], stats[arm][i]["target_cls"])
            for i in expected
        ):
            raise ValueError("Paired diagnostic IDs/GT changed")
        aggregate = {key: metric_record(stats[arm], rs) for key, rs in subgroups.items()}
        if not np.isclose(
            aggregate["all"]["mAP50_95"], native["overall"]["mAP50_95"], atol=1e-6, rtol=0
        ):
            raise ValueError("Saved native matching/AP disagrees")
        records[arm] = {
            "native": native,
            "subgroups": aggregate,
            "selected_epoch": chosen,
            "training_csv_seconds": float(history.time.iloc[-1]),
            "runtime": {
                "mean_ms": float(np.mean(timings[arm])),
                "p95_ms": float(np.quantile(timings[arm], 0.95)),
                "calls": len(timings[arm]),
            },
            "config_sha256": file_hash(root / f"configs/{arm}.yaml"),
            "data_sha256": file_hash(directory / "data/data.yaml"),
            "train_data_sha256": file_hash(directory / "train.yaml"),
            "best_sha256": file_hash(checkpoint),
            "last_sha256": file_hash(directory / "train/weights/last.pt"),
            "statistics_sha256": file_hash(stats_path),
        }
        for source, target in [
            (directory / "train/results.csv", f"{arm}_results.csv"),
            (directory / "train/args.yaml", f"{arm}_args.yaml"),
            (directory / "eval/metrics.json", f"{arm}_metrics.json"),
        ]:
            shutil.copyfile(source, report / target)
        artifacts[arm] = {
            "statistics": str(stats_path.relative_to(root)),
            "sha256": file_hash(stats_path),
        }
    counts = {r["id"]: {arm: errors(stats[arm], [r["id"]]) for arm in ARMS} for r in dev}
    examples = {
        key: sorted(
            [r["id"] for r in rs],
            key=lambda i: (
                counts[i]["v2"]["TP50"] - counts[i]["v0"]["TP50"],
                counts[i]["v0"]["FP50"] - counts[i]["v2"]["FP50"],
                i,
            ),
        )[:2]
        for key, rs in subgroups.items()
    }
    error_summary = {
        key: {arm: errors(stats[arm], [r["id"] for r in rs]) for arm in ARMS}
        for key, rs in subgroups.items()
    }
    freeze_json(output / "errors.json", counts)
    _plot(root, report, dev, records, examples, counts)
    seconds = read_json(root / "runs/issue45-s20/seconds.json")
    original_seconds = read_json(root / "runs/issue43/native_seconds.json")
    summary = {
        "role": "development_validation",
        "training_subset": "S20",
        "seed": 2026,
        "records": records,
        "errors": error_summary,
        "examples": {
            key: [{"id": i, "counts": counts[i]} for i in ids] for key, ids in examples.items()
        },
        "diagnostic_seconds": diagnostic_seconds,
        "v2_notebook_seconds": seconds,
        "v0_v1_notebook_seconds": original_seconds,
        "execution": read_json(root / "runs/issue45-s20/execution.json"),
        "parity": read_json(root / "runs/issue45-s20/parity.json"),
        "development_ids_sha256": digest(sorted(expected)),
        "source_reads_sha256": digest(source_reads),
        "source_reads": len(source_reads),
        "diagnostic_artifacts": artifacts,
        "report_sha256": file_hash(__file__),
        "wall_seconds": time.perf_counter() - start,
        "final_test_accessed": False,
        "confirmation_accessed": False,
    }
    freeze_json(report / "summary.json", summary)
    freeze_json(
        report / "evidence.json",
        {p.name: file_hash(p) for p in sorted(report.iterdir()) if p.is_file()},
    )
    return summary


def _plot(root, report, rows, records, examples, counts):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    x = np.arange(5)
    for index, arm in enumerate(ARMS):
        axes[0].bar(
            x + (index - 1) * 0.25,
            [records[arm]["subgroups"]["all"]["classes"][c]["AP50_95"] * 100 for c in FIVE_CLASSES],
            width=0.25,
            label=arm,
        )
        history = pd.read_csv(report / f"{arm}_results.csv")
        axes[1].plot(history.epoch, history["metrics/mAP50-95(B)"] * 100, label=arm)
    axes[0].set(xticks=x, xticklabels=FIVE_CLASSES, ylabel="Policy-matched development AP50:95 (%)")
    axes[1].set(xlabel="Epoch", ylabel="Shared Raw selection mAP50:95 (%)")
    for axis in axes:
        axis.legend()
    fig.suptitle("S20, development, single seed — exploratory; final holdout sealed")
    fig.savefig(report / "comparison.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    lookup = {r["id"]: r for r in rows}
    for axis, key in zip(
        axes.flat, ("dark", "bright", "blurred", "small_two_wheeler"), strict=True
    ):
        i = examples[key][0]
        image = read_image(root / "runs/v0/data/images/val" / f"{i}.jpg")
        axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        for c, x1, y1, x2, y2 in lookup[i]["boxes"]:
            axis.add_patch(
                Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="orange", linewidth=0.7)
            )
            if c in (3, 4):
                axis.text(x1, y1, FIVE_CLASSES[c], color="orange", fontsize=7)
        a, b = counts[i]["v0"], counts[i]["v2"]
        axis.set_title(f"{key}: {i}\nTP/FP v0 {a['TP50']}/{a['FP50']} → v2 {b['TP50']}/{b['FP50']}")
        axis.axis("off")
    fig.suptitle("Illustrative conf .25 / IoU .50 counts; orange = original GT; no per-image AP")
    fig.savefig(report / "errors.png", dpi=120)
    plt.close(fig)
