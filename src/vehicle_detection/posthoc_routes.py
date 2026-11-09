"""Descriptive Apply/Skip aggregation of frozen post-hoc native matches; no inference."""

from pathlib import Path

import numpy as np

from .common import digest, file_hash, freeze_json, read_json
from .five_class import FIVE_CLASSES
from .full_resolution_training import require_new
from .posthoc_training import STUDY
from .posthoc_training import read_report as original_report

REPORT = "docs/issue47-posthoc-routes"


def read_report(root):
    report = Path(root) / REPORT
    evidence = read_json(report / "evidence.json")
    if "result.json" not in evidence:
        raise ValueError("Missing route result hash")
    for name, expected in evidence.items():
        if Path(name).name != name or file_hash(report / name) != expected:
            raise ValueError("Route evidence changed")
    return read_json(report / "result.json")


def partition(rows, exports):
    routes = {}
    for split, sample_id, _, _, _, applied in exports:
        if split == "val":
            if sample_id in routes or type(applied) is not bool:
                raise ValueError("Duplicate or invalid frozen route")
            routes[sample_id] = applied
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(routes):
        raise ValueError("Route/development IDs differ")
    return {
        name: [r for r in rows if routes[r["id"]] == applied]
        for name, applied in (("apply", True), ("skip", False))
    }


def cohort_record(stats, rows):
    from .method_discovery import outcome

    record = outcome(stats, rows)
    missing = set(FIVE_CLASSES) - record["classes"].keys()
    for cls, name in enumerate(FIVE_CLASSES):
        present = [r for r in rows if any(b[0] == cls for b in r["boxes"])]
        groups = len({r["source_group"] for r in present})
        values = record["classes"].setdefault(
            name, {"AP50": None, "AP50_95": None, "instances": 0, "source_groups": 0}
        )
        values.update(
            images=len(present),
            support="missing"
            if not present
            else "weak (<5 source groups)"
            if groups < 5
            else "≥5 source groups",
        )
    if missing:
        record.update(mAP50_95=None, mAP50=None)
    record["missing_classes"] = sorted(missing)
    record["source_groups"] = len({r["source_group"] for r in rows})
    record["leakage_groups"] = len({r["group"] for r in rows})
    return record


def differences(before, after):
    def change(a, b):
        return None if a is None or b is None else b - a

    return {
        **{m: change(before[m], after[m]) for m in ("mAP50_95", "mAP50")},
        "classes": {
            name: {
                m: change(before["classes"][name][m], after["classes"][name][m])
                for m in ("AP50_95", "AP50")
            }
            for name in FIVE_CLASSES
        },
    }


def build(root):
    """Read only hash-bound receipts/matching arrays; write a separate compact report."""
    from .evaluation_stats import load_stats
    from .method_discovery import outcome

    root = Path(root).resolve()
    report, output = root / REPORT, root / f"runs/{STUDY}"
    require_new(report)
    original = original_report(root)
    if file_hash(output / "prepared.json") != original["data"]["prepared_sha256"]:
        raise ValueError("Prepared receipt changed")
    receipt = read_json(output / "prepared.json")["sha256"]
    for name in ("rows.json", "exports.json"):
        if file_hash(output / name) != receipt[name]:
            raise ValueError("Frozen route/GT receipt changed")
    rows = read_json(output / "rows.json")["val"]
    cohorts = partition(rows, read_json(output / "exports.json"))
    if {k: len(v) for k, v in cohorts.items()} != original["data"]["routing"]["val"]:
        raise ValueError("Frozen route counts differ")
    ids = {r["id"] for r in rows}
    metrics, targets = {}, None
    for cell, saved in original["statistics"].items():
        path = root / saved["path"]
        if file_hash(path) != saved["sha256"]:
            raise ValueError("Original native matches changed")
        stats = load_stats(path)
        if set(stats) != ids:
            raise ValueError("Native matching IDs differ")
        if targets is None:
            targets = {i: s["target_cls"] for i, s in stats.items()}
        if any(not np.array_equal(stats[i]["target_cls"], targets[i]) for i in ids):
            raise ValueError("Paired original GT changed")
        if outcome(stats, rows) != original["metrics"][cell]["all"]:
            raise ValueError("Full-development native aggregation does not reproduce")
        metrics[cell] = {name: cohort_record(stats, rs) for name, rs in cohorts.items()}
    result = {
        "scope": "Descriptive saved-match Apply/Skip analysis; not independent confirmation",
        "original_evidence_index_sha256": file_hash(root / f"docs/{STUDY}/evidence.json"),
        "prepared_receipt_sha256": original["data"]["prepared_sha256"],
        "inputs": {
            **{
                str((output / n).relative_to(root)): receipt[n]
                for n in ("rows.json", "exports.json")
            },
            "src/vehicle_detection/posthoc_routes.py": file_hash(Path(__file__)),
        },
        "statistics": original["statistics"],
        "cohorts": {
            name: {"images": len(rs), "ids_sha256": digest(sorted(r["id"] for r in rs))}
            for name, rs in cohorts.items()
        },
        "metrics": metrics,
        "contrasts": {
            name: {
                label: differences(metrics[before][name], metrics["adaptive/adaptive"][name])
                for label, before in (("D-B", "raw/adaptive"), ("D-A", "raw/raw"))
            }
            for name in cohorts
        },
        "support_rule": (
            "Flag <5 original source groups as weak; this is not statistical adequacy. "
            "Missing-class AP and five-class macro AP remain null, never zero-imputed."
        ),
        "selection_limit": (
            "Epoch12 maximizes the matched Raw-development actual640/pad0 selector, "
            "not Adaptive/960 optimality. No checkpoint is reselected."
        ),
        "new_training_runs": 0,
        "new_inference_image_passes": 0,
        "new_source_pixel_reads": 0,
        "confirmation_accessed": False,
        "final_test_accessed": False,
    }
    freeze_json(report / "result.json", result)
    freeze_json(report / "evidence.json", {"result.json": file_hash(report / "result.json")})
    return result
