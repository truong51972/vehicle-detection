"""Exploratory development comparisons from saved native matches; no inference/training."""

import importlib.metadata as md
from pathlib import Path

import numpy as np

from .common import digest, file_hash, freeze_json, read_json, reuse_evaluation, write_json
from .evaluation_stats import bootstrap_support, class_ap, load_stats, paired_interval
from .protocol import CONDITIONS, validate_protocol


def development_report(protocol, policy, checkpoints, directory, output, plan, *, resamples=100):
    """Report all natural slices and predeclared contrasts; retain support/exclusion ledgers."""
    if type(resamples) is not int or resamples < 0 or resamples == 1:
        raise ValueError("Use zero bootstrap draws for points only, or at least two draws")
    validate_protocol(protocol)
    directory, output = Path(directory), Path(output)
    names = [f"{m}-{t}" for m in ("pretrained", "raw", "adaptive") for t in ("raw", "adaptive")]
    recorded, results, settings = {}, {}, []
    for name in names:
        path = directory / name / "native_result.json"
        saved = read_json(path)
        identity = saved["identity"]
        model, _ = name.split("-")
        if (
            identity["role"] != "development_validation"
            or identity["protocol_id"] != protocol["manifest_id"]
            or identity["policy_id"] != policy["policy_id"]
            or identity["checkpoint_sha256"] != file_hash(checkpoints[model])
        ):
            raise ValueError("Recorded evaluation role/protocol/policy/checkpoint changed")
        settings.append(identity["settings"])
        results[name] = reuse_evaluation(path, identity)
        recorded[name] = file_hash(path)
    if any(s != settings[0] for s in settings):
        raise ValueError("Evaluation matrix settings differ")
    identity = {
        "version": "development-analysis-v1",
        "evaluations": recorded,
        "plan_sha256": file_hash(plan),
        "code_sha256": file_hash(__file__),
        "statistics_code_sha256": file_hash(Path(__file__).with_name("evaluation_stats.py")),
        "resamples": resamples,
        "seed": 2026,
        "settings": settings[0],
        "minimum_source_groups": 5,
        "versions": {p: md.version(p) for p in ("numpy", "ultralytics")},
    }
    cached = reuse_evaluation(output / "summary.json", identity)
    if cached is not None:
        return cached
    freeze_json(output / "identity.json", identity)
    stats = {n: load_stats(directory / n / "paired_stats.npz") for n in names}
    rows = protocol["splits"]["development_validation"]
    expected = {r["id"]: sorted(int(b[0]) for b in r["boxes"]) for r in rows}
    for values in stats.values():
        if set(values) != set(expected) or any(
            sorted(map(int, values[i]["target_cls"])) != target for i, target in expected.items()
        ):
            raise ValueError("Saved matching statistics differ from frozen development GT/IDs")
    errors = []
    attributes = {r["id"]: r["attributes"] for r in rows}
    for name, values in stats.items():
        ranked = []
        for sample_id, record in values.items():
            shown = record["conf"] >= 0.25
            matched = int(record["tp"][shown, 0].sum())
            ranked.append(
                {
                    "evaluation": name,
                    "id": sample_id,
                    "false_positives_at_025": int(shown.sum()) - matched,
                    "false_negatives_at_025": len(record["target_cls"]) - matched,
                    **attributes[sample_id],
                }
            )
        errors.extend(
            sorted(
                ranked,
                key=lambda r: (-r["false_negatives_at_025"], -r["false_positives_at_025"], r["id"]),
            )[:12]
        )
    slices = [("overall", "all", rows)]
    slices += [
        (axis, condition, [r for r in rows if r["attributes"][axis] == condition])
        for axis in CONDITIONS
        for condition in sorted({r["attributes"][axis] for r in rows})
    ]
    supports, points = [], {}
    for axis, condition, selected in slices:
        selected_ids = {r["id"] for r in selected}
        for name in names:
            ids = [i for i in stats[name] if i in selected_ids]
            values = class_ap(stats[name], ids, detailed=True)
            points[name, axis, condition] = values
            supports.append(
                {
                    "evaluation": name,
                    "axis": axis,
                    "condition": condition,
                    "images": len(ids),
                    "groups": len({r["group"] for r in selected}),
                    "mAP50_95": float(np.mean([v["AP50_95"] for v in values.values()]))
                    if values
                    else None,
                    "mAP50": float(np.mean([v["AP50"] for v in values.values()]))
                    if values
                    else None,
                    "classes": {str(c): v for c, v in values.items()},
                }
            )
            if (
                axis == "overall"
                and abs(supports[-1]["mAP50_95"] - results[name]["overall"]["mAP50_95"]) > 1e-8
            ):
                raise ValueError("Native matching-statistic AP differs from saved metric")
    overall = [(f"{m}-raw", f"{m}-adaptive") for m in ("pretrained", "raw", "adaptive")]
    overall += [
        ("pretrained-raw", "raw-raw"),
        ("pretrained-raw", "adaptive-raw"),
        ("raw-raw", "adaptive-raw"),
        ("raw-adaptive", "adaptive-adaptive"),
    ]
    comparisons, artifacts = [], {}
    for axis, condition, selected in slices:
        contrasts = (
            overall
            if axis == "overall"
            else [("raw-raw", "adaptive-raw"), ("raw-adaptive", "adaptive-adaptive")]
        )
        pilot = {"rows": selected}
        for before, after in contrasts:
            row = {
                "before": before,
                "after": after,
                "axis": axis,
                "condition": condition,
                "images": len(selected),
                "groups": len({r["group"] for r in selected}),
            }
            selected_ids = {r["id"] for r in selected}
            subset = {i: value for i, value in stats[before].items() if i in selected_ids}
            paired = {i: stats[after][i] for i in subset}
            try:
                policy_support = bootstrap_support(subset, pilot)
            except ValueError as exc:
                if "No classes meet" not in str(exc):
                    raise
                comparisons.append({**row, "status": "unsupported", "reason": str(exc)})
                continue
            if resamples == 0:
                target = policy_support["class_ids"]
                a = float(np.mean([points[before, axis, condition][c]["AP50_95"] for c in target]))
                b = float(np.mean([points[after, axis, condition][c]["AP50_95"] for c in target]))
                comparisons.append(
                    {
                        **row,
                        "status": "not_resampled",
                        "estimand": policy_support,
                        "before_ap": a,
                        "after_ap": b,
                        "recovery": b - a,
                        "reason": "Point estimates only; paired uncertainty pending",
                    }
                )
                continue
            filename = digest([before, after, axis, condition]) + ".json"
            path = output / filename
            if path.exists():
                record = read_json(path)
                if record["identity"] != identity or record["checksum"] != digest(record["result"]):
                    raise ValueError("Changed cached interval")
                interval = record["result"]
            else:
                print("Bootstrap", before, after, axis, condition, flush=True)
                interval = paired_interval(subset, paired, pilot, resamples=resamples)
                write_json(
                    path, {"identity": identity, "result": interval, "checksum": digest(interval)}
                )
            if interval["estimand"] != policy_support:
                raise ValueError("Changed fixed support target")
            artifacts[filename] = file_hash(path)
            comparisons.append(
                {
                    **row,
                    "status": "exploratory",
                    **{k: v for k, v in interval.items() if k != "draws"},
                    "ledger": filename,
                }
            )
    result = {
        "matrix": [
            {
                "evaluation": n,
                **results[n]["overall"],
                "classes": results[n]["classes"],
                "speed_ms": results[n]["speed_ms"],
            }
            for n in names
        ],
        "conditions": supports,
        "errors": errors,
        "comparisons": comparisons,
        "artifacts": artifacts,
        "provenance": identity,
        "final_test_accessed": False,
        "limitation": (
            "Single training seed; conditional exploratory group intervals; "
            "no multiplicity adjustment"
        ),
    }
    write_json(
        output / "summary.json",
        {"identity": identity, "result": result, "checksum": digest(result)},
    )
    return result
