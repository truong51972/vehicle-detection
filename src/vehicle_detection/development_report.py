"""Development comparisons from saved native YOLO matching statistics; never infer."""

from importlib.metadata import version
from pathlib import Path

import numpy as np

from .common import digest, file_hash, read_json, reuse_evaluation, write_json
from .evaluation_stats import bootstrap_support, class_ap, load_stats, paired_interval
from .protocol import CONDITIONS, validate_protocol


def development_report(protocol, policy, checkpoints, directory, output, plan, *, resamples=100):
    """Native 3×2 matrix, descriptive condition slices and two paired group contrasts."""
    if type(resamples) is not int or resamples < 0 or resamples == 1:
        raise ValueError("Use 0 for points only or at least 2 group bootstrap draws")
    validate_protocol(protocol)
    directory, output = Path(directory), Path(output)
    names = [
        f"{model}-{input_name}"
        for model in ("pretrained", "raw", "adaptive")
        for input_name in ("raw", "adaptive")
    ]
    hashes = {model: file_hash(path) for model, path in checkpoints.items()}
    evaluations, saved, configs = {}, {}, []
    for name in names:
        model, _ = name.split("-")
        path = directory / name / "native_result.json"
        document = read_json(path)
        bound = document["identity"]
        if (
            bound["role"] != "development_validation"
            or bound["protocol_id"] != protocol["manifest_id"]
            or bound["policy_id"] != policy["policy_id"]
            or bound["checkpoint_sha256"] != hashes[model]
        ):
            raise ValueError("Recorded evaluation role/protocol/policy/checkpoint changed")
        configs.append(bound["settings"])
        evaluations[name] = reuse_evaluation(path, bound)
        saved[name] = file_hash(path)

    if any(config != configs[0] for config in configs):
        raise ValueError("Evaluation matrix settings differ")
    config = configs[0]
    identity = {
        "version": "development-analysis-v3",
        "evaluations": saved,
        "plan_sha256": file_hash(plan),
        "resamples": resamples,
        "seed": 2026,
        "settings": config,
        "minimum_source_groups": 5,
        "versions": {p: version(p) for p in ("numpy", "ultralytics")},
    }
    summary = output / "summary.json"
    if summary.is_file():
        old = read_json(summary)["identity"]
        if old.get("version") in ("development-analysis-v1", "development-analysis-v2"):
            stable = (
                "evaluations",
                "plan_sha256",
                "resamples",
                "seed",
                "settings",
                "minimum_source_groups",
                "versions",
            )
            if any(old.get(k) != identity[k] for k in stable):
                raise ValueError("Historical analysis inputs/settings changed")
            return reuse_evaluation(summary, old)
    cached = reuse_evaluation(summary, identity)
    if cached is not None:
        return cached

    stats = {name: load_stats(directory / name / "paired_stats.npz") for name in names}
    rows = protocol["splits"]["development_validation"]
    expected = {r["id"]: sorted(int(b[0]) for b in r["boxes"]) for r in rows}
    for records in stats.values():
        if set(records) != set(expected) or any(
            sorted(map(int, records[i]["target_cls"])) != classes for i, classes in expected.items()
        ):
            raise ValueError("Saved matching statistics differ from frozen development GT")

    slices = [("overall", "all", rows)]
    slices.extend(
        (axis, value, [r for r in rows if r["attributes"][axis] == value])
        for axis in CONDITIONS
        for value in sorted({r["attributes"][axis] for r in rows})
    )
    conditions = []
    for axis, value, selected in slices:
        ids = [r["id"] for r in selected]
        for name in names:
            per_class = class_ap(stats[name], ids, detailed=True)
            conditions.append(
                {
                    "evaluation": name,
                    "axis": axis,
                    "condition": value,
                    "images": len(ids),
                    "groups": len({r["group"] for r in selected}),
                    "mAP50": float(np.mean([p["AP50"] for p in per_class.values()]))
                    if per_class
                    else None,
                    "mAP50_95": float(np.mean([p["AP50_95"] for p in per_class.values()]))
                    if per_class
                    else None,
                    "classes": {str(k): v for k, v in per_class.items()},
                }
            )

    comparisons = []
    pilot = {"rows": rows}
    for treatment in ("raw", "adaptive"):
        before, after = f"raw-{treatment}", f"adaptive-{treatment}"
        support = bootstrap_support(stats[before], pilot)
        if resamples:
            interval = paired_interval(stats[before], stats[after], pilot, resamples=resamples)
            status = "exploratory"
        else:
            ids = list(stats[before])
            a, b = class_ap(stats[before], ids), class_ap(stats[after], ids)
            target = support["class_ids"]
            mean_a, mean_b = (float(np.mean([values[c] for c in target])) for values in (a, b))
            interval = {
                "before_ap": mean_a,
                "after_ap": mean_b,
                "recovery": mean_b - mean_a,
                "estimand": support,
                "reason": "Point estimates only; group uncertainty not computed",
            }
            status = "not_resampled"
        comparisons.append(
            {
                "before": before,
                "after": after,
                "axis": "overall",
                "condition": "all",
                "images": len(rows),
                "groups": len({r["group"] for r in rows}),
                "status": status,
                **interval,
            }
        )

    errors = []
    for name, records in stats.items():
        ranked = []
        for sample_id, values in records.items():
            shown = values["conf"] >= 0.25
            matched = int(values["tp"][shown, 0].sum())
            ranked.append(
                {
                    "evaluation": name,
                    "id": sample_id,
                    "false_positives_at_025": int(shown.sum()) - matched,
                    "false_negatives_at_025": len(values["target_cls"]) - matched,
                }
            )
        errors.extend(
            sorted(
                ranked,
                key=lambda row: (
                    -row["false_negatives_at_025"],
                    -row["false_positives_at_025"],
                    row["id"],
                ),
            )[:12]
        )

    result = {
        "matrix": [
            {
                "evaluation": name,
                **value["overall"],
                "classes": value["classes"],
                "speed_ms": value["speed_ms"],
            }
            for name, value in evaluations.items()
        ],
        "conditions": conditions,
        "comparisons": comparisons,
        "errors": errors,
        "artifacts": {},
        "final_test_accessed": False,
        "limitation": (
            "Single-seed, exploratory two-contrast paired group analysis; "
            "conditional support is descriptive; no multiplicity adjustment"
        ),
    }
    write_json(summary, {"identity": identity, "result": result, "checksum": digest(result)})
    return result
