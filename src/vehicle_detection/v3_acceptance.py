"""Descriptive acceptance audit of frozen features/matches; never acquires pixels or models."""

import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .common import file_hash, freeze_json, read_json
from .five_class import FIVE_CLASSES
from .v3_adaptive import STUDY, read_report, route

REPORT = "docs/issue47-v3-960-acceptance/saved_audit.json"


def failed_gates(features):
    """Describe the frozen image gate; overlapping failures are not object-level causes."""
    failures = {
        "contrast_ge_0.20": features["contrast"] >= 0.20,
        "brightness_le_0.04": features["brightness"] <= 0.04,
        "brightness_ge_0.80": features["brightness"] >= 0.80,
        "noise_gt_0.01": features["noise"] > 0.01,
        "black_fraction_ge_0.30": features["black_fraction"] >= 0.30,
        "white_fraction_ge_0.10": features["white_fraction"] >= 0.10,
    }
    rejected = [name for name, failed in failures.items() if failed]
    if bool(rejected) == route(features):
        raise ValueError("Diagnostic gate definitions disagree with the frozen route")
    return rejected


def route_audit(features, stats):
    if set(features) != set(stats):
        raise ValueError("Saved features and native matching IDs differ")
    rejected = {i: failed_gates(f) for i, f in features.items()}
    result = {}
    for cls in ("motorcycle", "bicycle"):
        counts = {
            i: int((s["target_cls"] == FIVE_CLASSES.index(cls)).sum()) for i, s in stats.items()
        }
        failures, combinations = Counter(), Counter()
        for sample_id, count in counts.items():
            if count and rejected[sample_id]:
                combinations[" + ".join(rejected[sample_id])] += count
                for gate in rejected[sample_id]:
                    failures[gate] += count
        result[cls] = {
            "GT": sum(counts.values()),
            "apply_GT": sum(n for i, n in counts.items() if not rejected[i]),
            "skip_GT": sum(n for i, n in counts.items() if rejected[i]),
            "apply_images": sum(n > 0 and not rejected[i] for i, n in counts.items()),
            "skip_images": sum(n > 0 and bool(rejected[i]) for i, n in counts.items()),
            "overlapping_failed_gate_GT": dict(failures),
            "exclusive_failed_gate_combinations_GT": dict(combinations),
        }
    return result


def review(root):
    """Reconcile saved native AP/counts with the committed result, without inference/resampling."""
    from .evaluation_stats import class_ap, load_stats
    from .research_round3_report import errors

    root = Path(root).resolve()
    report = root / f"docs/{STUDY}"
    result = read_report(root)
    if (
        file_hash(report / "evidence.json")
        != read_json(report / "validation.json")["evidence_index_sha256"]
    ):
        raise ValueError("Original acquisition index and validation receipt disagree")
    rows = pd.read_csv(report / "membership.csv.gz", keep_default_na=False).to_dict("records")
    features = json.loads(gzip.decompress((report / "features.json.gz").read_bytes()))
    expected = {r["id"] for r in rows}
    if set(features) != expected:
        raise ValueError("Frozen membership and features differ")
    statistics = {}
    for arm, record in result["statistics"].items():
        path = root / record["path"]
        if file_hash(path) != record["sha256"]:
            raise ValueError("Saved native matching file changed")
        stats = load_stats(path)
        if set(stats) != expected:
            raise ValueError("Saved native matching IDs changed")
        statistics[arm] = stats
    raw = statistics["raw"]
    if any(
        not np.array_equal(raw[i]["target_cls"], stats[i]["target_cls"])
        for stats in statistics.values()
        for i in expected
    ):
        raise ValueError("Paired native target classes changed")
    cohorts = {
        "all": rows,
        "dark": [r for r in rows if features[r["id"]]["brightness"] < 0.12],
        "low_sharpness": [r for r in rows if features[r["id"]]["sharpness"] < 50],
    }
    supports = {}
    for name, selected in cohorts.items():
        ids = [r["id"] for r in selected]
        supports[name] = {"images": len(ids), "classes": {}}
        for c, cls in enumerate(FIVE_CLASSES):
            counts = {i: int((raw[i]["target_cls"] == c).sum()) for i in ids}
            supports[name]["classes"][cls] = {
                "GT": sum(counts.values()),
                "source_groups": len({r["source_group"] for r in selected if counts[r["id"]]}),
            }
        for arm, stats in statistics.items():
            observed = class_ap(stats, ids, detailed=True)
            saved = result["metrics"][arm][name]
            if len(ids) != saved["images"] or {FIVE_CLASSES[c] for c in observed} != set(
                saved["classes"]
            ):
                raise ValueError("Cohort population or represented classes changed")
            for c, cls in enumerate(FIVE_CLASSES):
                gt = supports[name]["classes"][cls]["GT"]
                found = errors(stats, ids, cls=c)
                if {"GT": gt, **found, "FN50": gt - found["TP50"]} != saved["counts"][cls]:
                    raise ValueError("Saved native TP/FP/FN differ from frozen result")
                if c in observed and (
                    gt != saved["classes"][cls]["instances"]
                    or supports[name]["classes"][cls]["source_groups"]
                    != saved["classes"][cls]["source_groups"]
                ):
                    raise ValueError("Saved cohort GT/group support differs from frozen result")
                if c in observed and any(
                    not np.isclose(
                        observed[c][metric], saved["classes"][cls][metric], rtol=0, atol=1e-12
                    )
                    for metric in ("AP50", "AP50_95")
                ):
                    raise ValueError("Saved native class AP differs from frozen result")
    return {
        "status": "Descriptive acceptance audit; not independent validation or a new experiment",
        "inputs": {
            n: file_hash(report / n)
            for n in ("evidence.json", "validation.json", "membership.csv.gz", "features.json.gz")
        },
        "statistics": result["statistics"],
        "analysis_sha256": file_hash(__file__),
        "rare_class_routing": route_audit(features, raw),
        "cohort_support": supports,
        "verified": (
            "All/dark/low-sharpness class AP50/AP50:95 and TP/FP/FN match all three frozen arms"
        ),
        "new_source_pixel_reads": 0,
        "new_inference_passes": 0,
        "new_bootstrap_draws": 0,
    }


if __name__ == "__main__":
    root = Path.cwd()
    result = review(root)
    freeze_json(root / REPORT, result)
    print(json.dumps(result["rare_class_routing"], indent=2))
