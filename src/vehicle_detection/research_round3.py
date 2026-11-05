"""Fixed, bounded exploration replication; never consumes confirmation or releases a policy."""

import argparse
import builtins
import importlib.metadata as md
import io
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd

from .common import digest, file_hash, freeze_json, read_json
from .data import convert_labels, scan_bdd
from .five_class import FIVE_CLASSES, derive_labels, remap_boxes
from .preprocessing_experiments import (
    calibration_sample,
    export_candidate,
    validate_sample,
)
from .quality_followup import benchmark_runtime
from .stress import converter_records

REPORT = Path("docs/round2-research-v3")
OUTPUT = Path("runs/round2-research-v3")
SENTINELS = (
    "clean",
    "synthetic_shot_read_s1",
    "synthetic_shot_read_s3",
    "synthetic_underexposure_s3",
    "synthetic_motion_s3",
)


def configs_from_evidence(quality, previous):
    """One fixed noise range from old observations, no grid or detection-score tuning."""
    table = pd.DataFrame(quality["records"])
    if quality["checksum"] != digest(quality["records"]):
        raise ValueError("Historical quality checksum changed")
    low = float(table[table.scenario == "clean"].noise_highpass.quantile(0.95))
    high = float(
        np.mean(
            [
                table[table.scenario == f"synthetic_shot_read_s{s}"].noise_highpass.median()
                for s in (2, 3)
            ]
        )
    )
    bilateral = {"bilateral": {}}
    return {
        "skip": {"steps": []},
        "frozen_v2": {"v2": True},
        "pr38_locked": previous["selected"]["config"],
        "bilateral": {"steps": [bilateral]},
        "noise_range": {
            "rules": [
                {"metric": "noise_highpass", "above": high, "steps": []},
                {"metric": "noise_highpass", "above": low, "steps": [bilateral]},
            ]
        },
    }


@contextmanager
def exploration_access(data_root, pilot):
    """Reject source writes, all non-pilot source reads and the reserved confirmation plan."""
    data_root = Path(data_root).resolve()
    allowed = {data_root / r["source_path"] for r in pilot["rows"]}
    original_open, original_io, original_imread = builtins.open, io.open, cv2.imread
    reads = set()
    reserved = Path("runs/round2-quality-v2/confirmation_plan.json").resolve()

    def check(path, mode="r"):
        if not isinstance(path, (str, bytes, Path)):
            return
        path = Path(path).absolute()
        if path == reserved:
            raise ValueError("Reserved confirmation plan stays untouched")
        if path.is_relative_to(data_root):
            if any(c in mode for c in "wax+") or path.resolve() not in allowed:
                raise ValueError("Only frozen exploration source reads are allowed")
            reads.add(str(path.relative_to(data_root)))

    def opening(original):
        def guarded(path, mode="r", *args, **kwargs):
            check(path, mode)
            return original(path, mode, *args, **kwargs)

        return guarded

    def imread(path, *args, **kwargs):
        check(path)
        return original_imread(path, *args, **kwargs)

    with (
        patch("builtins.open", opening(original_open)),
        patch("io.open", opening(original_io)),
        patch("cv2.imread", imread),
    ):
        yield reads


def metric_record(stats, rows):
    from .evaluation_stats import class_ap

    ids = [r["id"] for r in rows]
    classes = class_ap(stats, ids, detailed=True)
    if set(classes) != set(range(5)):
        raise ValueError("Fixed five-class AP requires every class")
    return {
        "mAP50_95": float(np.mean([v["AP50_95"] for v in classes.values()])),
        "mAP50": float(np.mean([v["AP50"] for v in classes.values()])),
        "classes": [{"class": label, **classes[c]} for c, label in enumerate(FIVE_CLASSES)],
    }


def assess(records, scenarios):
    """Predeclared advancement gate, applied only to the new exploration slice."""
    table = pd.DataFrame(records)
    scores = table.groupby(["candidate", "family"]).mAP50_95.mean().groupby("candidate").mean()
    indexed = table.set_index(["candidate", "scenario"])
    decisions = []
    for name in ("pr38_locked", "bilateral", "noise_range"):
        reasons = []
        for control in ("skip", "frozen_v2"):
            if scores[name] - scores[control] < 0.002:
                reasons.append(f"Gain below 0.2 AP percentage points vs {control}")
            for scenario in scenarios:
                a, b = indexed.loc[(name, scenario["id"])], indexed.loc[(control, scenario["id"])]
                limit = 0.005 if scenario["id"] == "clean" else 0.01
                if a.mAP50_95 - b.mAP50_95 < -limit:
                    reasons.append(f"Scenario regression: {control}/{scenario['id']}")
                if any(
                    x["AP50_95"] - y["AP50_95"] < -0.02
                    for x, y in zip(a.classes, b.classes, strict=True)
                ):
                    reasons.append(f"Class regression: {control}/{scenario['id']}")
        decisions.append(
            {
                "candidate": name,
                "score": float(scores[name]),
                "reasons": reasons,
                "advance": not reasons,
            }
        )
    return {"scores": {k: float(v) for k, v in scores.items()}, "candidates": decisions}


def run(*, allow_inference=False, measure_runtime=False):
    from .evaluation_stats import load_stats
    from .stress_eval import SETTINGS, evaluate_calibration

    protocol = read_json("runs/issue5-v1/protocol.json")
    manifest = read_json("docs/round2-evaluation-v2/evaluation_sets.json")
    v2 = read_json("docs/issue6-v2/policy.json")
    pilot = calibration_sample(protocol, manifest, "calibration_exploration", target_images=512)
    old = calibration_sample(protocol, manifest, "calibration_exploration", target_images=256)
    old_groups = {r["group"] for r in old["rows"]}
    new = [r for r in pilot["rows"] if r["group"] not in old_groups]
    validate_sample(protocol, manifest, pilot)
    if len(pilot["rows"]) != 512 or len(new) != 256:
        raise ValueError("This experiment is bounded to the frozen 512/256 prefix")
    quality_path = Path("runs/round2-quality-v1/quality.json")
    previous_path = Path("docs/round2-quality-v2/search.json")
    for root, path in (
        ("docs/round2-quality-v1", quality_path),
        ("docs/round2-quality-v2", previous_path),
    ):
        expected = read_json(Path(root) / "evidence.json")["artifacts"][str(path)]
        if file_hash(path) != expected:
            raise ValueError("Historical input changed")
    configs = configs_from_evidence(read_json(quality_path), read_json(previous_path))
    support = {
        name: {
            label: len(
                {r["source_group"] for r in rows if c in {b[0] for b in remap_boxes(r["boxes"])}}
            )
            for c, label in enumerate(FIVE_CLASSES)
        }
        for name, rows in (("all", pilot["rows"]), ("new", new))
    }
    if min(support["new"].values()) < 5:
        raise ValueError("New exploration slice has insufficient class-group support")
    source_names = (
        "research_round3.py",
        "preprocessing_experiments.py",
        "quality_followup.py",
        "quality_search.py",
        "evaluation_data.py",
        "evaluation_stats.py",
        "stress_eval.py",
        "stress.py",
        "preprocess.py",
        "baseline.py",
    )
    plan = {
        "version": "issue30-autonomous-exploration-round3-v1",
        "pilot_id": pilot["pilot_id"],
        "old_pilot_id": old["pilot_id"],
        "protocol_id": protocol["manifest_id"],
        "manifest_id": manifest["manifest_id"],
        "rows_sha256": digest(pilot["rows"]),
        "new_ids_sha256": digest([r["id"] for r in new]),
        "configs": configs,
        "sentinels": list(SENTINELS),
        "settings": SETTINGS,
        "checkpoint_sha256": file_hash("weights/yolo26n.pt"),
        "v2_policy_id": v2["policy_id"],
        "hypotheses_sha256": file_hash(REPORT / "hypotheses.md"),
        "prior_quality_sha256": file_hash(quality_path),
        "prior_search_sha256": file_hash(previous_path),
        "code": {n: file_hash(Path(__file__).with_name(n)) for n in source_names},
        "versions": {n: md.version(n) for n in ("numpy", "opencv-python", "ultralytics", "torch")},
        "class_source_groups": support,
        "budget": {
            "initial_evaluations": 25,
            "maximum_evaluations": 50,
            "maximum_image_passes": 25600,
            "runtime_calls": 76800,
            "bootstrap_draws": 100,
        },
        "advancement": (
            "new exploration gain >=0.002 vs both controls; clean >=-0.005; "
            "other scenario >=-0.01; every class/scenario >=-0.02"
        ),
        "cost": {"mean_per_scenario_ms": 10, "p95_per_scenario_ms": 25, "threads": 8, "repeats": 3},
        "boundaries": (
            "Exploration half only. Reserved confirmation plan not read. "
            "No confirmation/development/final pixels or predictions. No release."
        ),
    }
    plan["plan_id"] = digest(plan)
    freeze_json(REPORT / "plan.json", plan)
    freeze_json(OUTPUT / "sample.json", pilot)
    converted = OUTPUT / "conversion/five"
    if not (converted / "conversion.json").exists():
        if not allow_inference:
            raise FileNotFoundError("Prepare conversion with --infer")
        source_records, _, _ = scan_bdd("data/bdd100k", strict=False)
        six = OUTPUT / "conversion/six"
        convert_labels(converter_records(pilot, protocol, source_records, "data/bdd100k"), six)
        derive_labels(six, converted, {"tune": pilot["rows"]})
    records, all_stats, sliced = [], {}, []
    scenarios = [s for s in manifest["scenarios"] if s["id"] in SENTINELS]
    cv2.setNumThreads(8)
    with exploration_access("data/bdd100k", pilot) as reads:

        def measure(selected):
            for scenario in selected:
                key = scenario["id"]
                for name, config in configs.items():
                    directory = OUTPUT / "metrics" / key / name
                    dataset = OUTPUT / "inputs" / key / name
                    if not allow_inference and not (directory / "native_result.json").exists():
                        raise FileNotFoundError("Missing native cache; explicitly allow inference")
                    export_candidate(
                        protocol,
                        manifest,
                        pilot,
                        "data/bdd100k",
                        converted,
                        dataset,
                        scenario,
                        config,
                        v2,
                    )
                    result = evaluate_calibration(
                        protocol,
                        pilot,
                        dataset,
                        "weights/yolo26n.pt",
                        directory,
                        allow_inference=allow_inference,
                    )
                    stats = load_stats(directory / "paired_stats.npz")
                    all_stats[key, name] = stats
                    records.append(
                        {
                            "candidate": name,
                            "scenario": key,
                            "family": scenario["family"],
                            **result["overall"],
                            "classes": result["classes"],
                            "native_result_sha256": file_hash(directory / "native_result.json"),
                            "statistics_sha256": file_hash(directory / "paired_stats.npz"),
                        }
                    )
                    for label, rows in (("all", pilot["rows"]), ("new", new), ("old", old["rows"])):
                        sliced.append(
                            {
                                "slice": label,
                                "candidate": name,
                                "scenario": key,
                                "family": scenario["family"],
                                **metric_record(stats, rows),
                            }
                        )
                    freeze_json(directory / "review_record.json", records[-1])
                    print(f"Verified {key}/{name}", flush=True)

        measure(scenarios)
        initial = assess([r for r in sliced if r["slice"] == "new"], scenarios)
        freeze_json(REPORT / "pilot_decision.json", initial)
        if any(r["advance"] for r in initial["candidates"]):
            extra = [s for s in manifest["scenarios"] if s["id"] not in SENTINELS]
            measure(extra)
            scenarios += extra
        freeze_json(REPORT / "measurements.json", records)
        freeze_json(REPORT / "slices.json", sliced)
        runtime_path = REPORT / "runtime.json"
        if not runtime_path.exists():
            if not measure_runtime:
                raise FileNotFoundError("Missing direct runtime; explicitly allow measurement")
            runtime = benchmark_runtime(
                protocol, manifest, pilot, "data/bdd100k", configs, v2, repeats=3, threads=8
            )
            freeze_json(runtime_path, runtime)
        freeze_json(
            REPORT / "access.json",
            {
                "source_paths_read": sorted(reads),
                "reserved_confirmation_plan_read": False,
                "source_writes": False,
                "confirmation_development_final_pixels": False,
            },
        )
    report = {
        "plan_id": plan["plan_id"],
        "scenarios": scenarios,
        "decision": assess([r for r in sliced if r["slice"] == "new"], scenarios),
        "evaluations": len(records),
        "image_passes": len(records) * 512,
        "v3_released": False,
        "status": "exploration_only_retain_frozen_v2",
    }
    freeze_json(REPORT / "summary.json", report)
    return protocol, manifest, pilot, new, configs, all_stats, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infer", action="store_true")
    parser.add_argument("--measure-runtime", action="store_true")
    args = parser.parse_args()
    run(allow_inference=args.infer, measure_runtime=args.measure_runtime)
