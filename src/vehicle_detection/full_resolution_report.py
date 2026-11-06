"""Finish Issue #51's interrupted CPU reporting from immutable saved matches only."""

import os
import resource
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .common import digest, file_hash, freeze_json, read_json
from .full_resolution_training import MATRIX, RESAMPLES, STUDY, historical_inventory


def reporting_inputs(root):
    output = root / f"runs/{STUDY}"
    report = root / f"docs/{STUDY}"
    statistics = read_json(output / "interrupted-reporting/statistics.json")
    if set(statistics) != {f"{a}/{b}" for a, b in MATRIX}:
        raise ValueError("Expected all four saved matrix cells")
    for name, expected in read_json(output / "prepared.json")["sha256"].items():
        path = output / name
        if not path.is_relative_to(output) or path.resolve() != path or file_hash(path) != expected:
            raise ValueError("Prepared metadata/export identity changed")
    names = ["rows.json", "features.json", "prepared.json", "plan.json"]
    inputs = {f"runs/{STUDY}/{name}": file_hash(output / name) for name in names}
    for key, record in statistics.items():
        if key not in {f"{a}/{b}" for a, b in MATRIX}:
            raise ValueError("Unexpected saved matrix cell")
        path = root / record["path"]
        if not path.is_relative_to(output) or path.resolve() != path:
            raise ValueError("Unexpected matching-array path")
        if file_hash(path) != record["sha256"]:
            raise ValueError("Saved matching arrays changed")
        inputs[record["path"]] = record["sha256"]
    for a, b in MATRIX:
        for suffix in ("metrics", "measurement"):
            name = f"eval{a}_{b}_{suffix}.json"
            inputs[f"docs/{STUDY}/{name}"] = file_hash(report / name)
    for size in (640, 960):
        receipt = report / f"train{size}_checkpoint.json"
        inputs[str(receipt.relative_to(root))] = file_hash(receipt)
        for name, expected in read_json(receipt)["checkpoint_hashes"].items():
            path = output / f"train{size}/weights/{name}.pt"
            if file_hash(path) != expected:
                raise ValueError("Completed training checkpoint changed")
    for name in ("full_resolution_report.py", "full_resolution_training.py", "evaluation_stats.py"):
        path = root / "src/vehicle_detection" / name
        inputs[str(path.relative_to(root))] = file_hash(path)
    return inputs, statistics


def cohort_interval(root, name, inputs):
    # This continuation has no model/loader calls; workers additionally hide CUDA.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from .evaluation_stats import load_stats, paired_interval
    from .five_class import FIVE_CLASSES
    from .method_discovery import cohorts

    output = root / f"runs/{STUDY}"
    directory = output / "reporting-recovery"
    receipt = directory / f"{name}.json"
    identity = {"inputs": inputs, "cohort": name, "resamples": RESAMPLES, "seed": 2026}
    if receipt.exists():
        saved = read_json(receipt)
        if saved["identity"] != identity or digest(saved["interval"]) != saved["interval_sha256"]:
            raise ValueError("Completed cohort receipt changed")
        return name, saved["interval"]
    started = time.perf_counter()
    rows = read_json(output / "rows.json")["val"]
    selected = cohorts(rows, read_json(output / "features.json"))[name]
    selected = [{**r, "group": r["source_group"]} for r in selected]
    ids = {r["id"] for r in selected}
    before, after = [load_stats(output / f"eval{s}_960/paired_stats.npz") for s in (640, 960)]
    interval = paired_interval(
        {k: v for k, v in before.items() if k in ids},
        {k: v for k, v in after.items() if k in ids},
        {"rows": selected},
        resamples=RESAMPLES,
        seed=2026,
        class_names=FIVE_CLASSES,
    )
    if interval["resamples"] != 1000 or interval["valid_draws"] + interval["invalid_draws"] != 1000:
        raise ValueError("Fixed bootstrap budget changed")
    freeze_json(
        receipt,
        {
            "identity": identity,
            "interval": interval,
            "interval_sha256": digest(interval),
            "wall_seconds": time.perf_counter() - started,
            "cpu_seconds": resource.getrusage(resource.RUSAGE_SELF).ru_utime
            + resource.getrusage(resource.RUSAGE_SELF).ru_stime,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    )
    return name, interval


def finish(root):
    """Continue only CPU analysis; completed cohort receipts survive process interruption."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from .evaluation_stats import load_stats
    from .full_resolution_training import assess, plot_results
    from .resolution_training import examples

    root = Path(root).resolve()
    output, report = root / f"runs/{STUDY}", root / f"docs/{STUDY}"
    if (report / "result.json").exists() or (report / "evidence.json").exists():
        raise FileExistsError("Completed report exists; never overwrite it")
    inputs, statistics = reporting_inputs(root)
    started = time.perf_counter()
    intervals = {}
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures = [
            pool.submit(cohort_interval, root, name, inputs) for name in ("all", "dark", "bright")
        ]
        for future in as_completed(futures):
            name, interval = future.result()
            intervals[name] = interval
            print(f"{name}: {interval['valid_draws']}/{RESAMPLES} valid draws saved", flush=True)
    if reporting_inputs(root)[0] != inputs:
        raise ValueError("Reporting inputs changed during analysis")
    metrics, timings = {}, {}
    for a, b in MATRIX:
        key = f"{a}/{b}"
        metrics[key] = read_json(report / f"eval{a}_{b}_metrics.json")
        timings[key] = read_json(report / f"eval{a}_{b}_measurement.json")
    before, after = [load_stats(root / statistics[k]["path"]) for k in ("640/960", "960/960")]
    a, b, c, d = [
        metrics[k]["all"]["mAP50_95"] for k in ("640/640", "640/960", "960/640", "960/960")
    ]
    if abs(intervals["all"]["recovery"] - (d - b)) > 1e-8:
        raise ValueError("Primary bootstrap and native matrix disagree")
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
        "development_images": len(before),
        "confirmation_final_accessed": False,
    }
    result["decision"] = assess(result)
    original = read_json(output / "preservation.json")
    current = historical_inventory(root)
    if (
        any(current["tracked"].get(n) != h for n, h in original["tracked"].items())
        or current["local"] != original["local"]
    ):
        raise ValueError("Historical acquisition files changed")
    freeze_json(
        report / "preservation.json",
        {
            "tracked_hashes": len(original["tracked"]),
            "local_size_mtimes": len(original["local"]),
            "new_reporting_files_excluded_from_historical_membership": True,
        },
    )
    freeze_json(
        report / "analysis_recovery.json",
        {
            "inputs": inputs,
            "resamples_per_cohort": RESAMPLES,
            "cpu_processes": 3,
            "wall_seconds": time.perf_counter() - started,
            "new_training_or_image_inference_passes": 0,
        },
    )
    freeze_json(report / "result.json", result)
    plot_results(report, result)
    freeze_json(
        report / "evidence.json",
        {p.name: file_hash(p) for p in sorted(report.iterdir()) if p.is_file()},
    )
    print("CPU reporting completed", flush=True)


if __name__ == "__main__":
    finish(Path.cwd())
