"""Development comparisons from saved native YOLO matching statistics; never infer."""

import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from .common import digest, file_hash, read_json, reuse_evaluation, write_json
from .data import CLASSES
from .datasets import verify_role
from .evaluation_stats import bootstrap_support, class_ap, load_stats, paired_interval
from .five_class import FIVE_CLASSES, remap_boxes
from .offline import verify_alignment
from .protocol import CONDITIONS, validate_protocol


def route_report(protocol, policy, inputs, directory, output):
    """Descriptive matched Gamma/Skip AP from frozen export decisions and native matches."""
    validate_protocol(protocol)
    directory, output = Path(directory), Path(output)
    rows = protocol["splits"]["development_validation"]
    datasets = {
        treatment: verify_role(
            inputs[treatment],
            rows,
            "tune",
            variant,
            policy["recipes"] if treatment == "adaptive" else None,
            deep=False,
        )
        for treatment, variant in (("raw", "raw_reencoded"), ("adaptive", "adaptive"))
    }
    verify_alignment([inputs["raw"], inputs["adaptive"]], deep=False)
    names = datasets["raw"]["export"]["conversion"].get("classes", CLASSES)
    if names == FIVE_CLASSES:
        rows = [{**r, "boxes": remap_boxes(r["boxes"])} for r in rows]
    meta = {
        t: pd.read_csv(Path(inputs[t]) / "metadata.csv").set_index("id", verify_integrity=True)
        for t in ("raw", "adaptive")
    }
    routes = {}
    for row in rows:
        sample_id = row["id"]
        raw, adaptive = (meta[t].loc[sample_id] for t in ("raw", "adaptive"))
        if raw.source_sha256 != adaptive.source_sha256 or json.loads(raw.decisions) != []:
            raise ValueError("Route source pairing/Raw decisions changed")
        decisions = json.loads(adaptive.decisions)
        if decisions not in ([], [{"name": "gamma", "params": policy["recipes"]["gamma"]}]):
            raise ValueError("Expected frozen Gamma-or-Skip export decisions")
        routes[sample_id] = "gamma" if decisions else "skip"
        if not decisions and raw.output_sha256 != adaptive.output_sha256:
            raise ValueError("Skip export bytes differ under the JPEG95 control")

    saved, bounds = {}, {}
    for model in ("pretrained", "raw", "adaptive"):
        for treatment in ("raw", "adaptive"):
            name = f"{model}-{treatment}"
            path = directory / name / "native_result.json"
            bound = read_json(path)["identity"]
            if (
                bound["role"] != "development_validation"
                or bound["protocol_id"] != protocol["manifest_id"]
                or bound["policy_id"] != policy["policy_id"]
                or bound["dataset"] != datasets[treatment]
                or bound.get("class_names", CLASSES) != names
            ):
                raise ValueError("Route evaluation dataset/taxonomy/protocol/policy changed")
            reuse_evaluation(path, bound)
            bounds[name], saved[name] = bound, file_hash(path)
        if (
            bounds[f"{model}-raw"]["checkpoint_sha256"]
            != bounds[f"{model}-adaptive"]["checkpoint_sha256"]
        ):
            raise ValueError("Route comparison must use the same checkpoint")
    configs = [b["settings"] for b in bounds.values()]
    if any(c != configs[0] for c in configs):
        raise ValueError("Route evaluation settings differ")
    identity = {
        "version": "development-routes-v1",
        "evaluations": saved,
        "metadata": {t: file_hash(Path(inputs[t]) / "metadata.csv") for t in meta},
        "protocol_id": protocol["manifest_id"],
        "policy_id": policy["policy_id"],
        "versions": {p: version(p) for p in ("numpy", "ultralytics")},
    }
    summary = output / "summary.json"
    cached = reuse_evaluation(summary, identity)
    if cached is not None:
        return cached

    slices, populations, crosstab = [], [], []
    for route in ("gamma", "skip"):
        selected = [r for r in rows if routes[r["id"]] == route]
        populations.append(
            {
                "route": route,
                "images": len(selected),
                "percent": 100 * len(selected) / len(rows),
                "groups": len({r["group"] for r in selected}),
            }
        )
        slices.append((route, "all", selected))
        for time in sorted({r["attributes"]["timeofday"] for r in rows}):
            contextual = [r for r in selected if r["attributes"]["timeofday"] == time]
            crosstab.append(
                {
                    "route": route,
                    "timeofday": time,
                    "images": len(contextual),
                    "groups": len({r["group"] for r in contextual}),
                }
            )
            slices.append((route, time, contextual))
    comparisons, supports, controls = [], [], []
    expected = {r["id"]: sorted(int(b[0]) for b in r["boxes"]) for r in rows}
    for model in ("pretrained", "raw", "adaptive"):
        before, after = (
            load_stats(directory / f"{model}-{t}" / "paired_stats.npz") for t in ("raw", "adaptive")
        )
        for records in (before, after):
            if set(records) != set(expected) or any(
                sorted(map(int, records[i]["target_cls"])) != cs for i, cs in expected.items()
            ):
                raise ValueError("Route matching statistics differ from frozen development GT")
        skip_ids = [i for i in expected if routes[i] == "skip"]
        mismatches = [
            i
            for i in skip_ids
            if any(not np.array_equal(before[i][key], after[i][key]) for key in before[i])
        ]
        controls.append(
            {
                "model": model,
                "images": len(skip_ids),
                "export_hash_mismatches": 0,
                "native_stat_mismatches": len(mismatches),
                "example_ids": mismatches[:10],
            }
        )
        for route, time, selected in slices:
            ids = [r["id"] for r in selected]
            a, b = (class_ap(records, ids, detailed=True) for records in (before, after))
            target = sorted({int(box[0]) for r in selected for box in r["boxes"]})
            if set(a) != set(target) or set(b) != set(target):
                raise ValueError("Route AP class denominator differs from paired GT")
            entry = {
                "model": model,
                "route": route,
                "timeofday": time,
                "images": len(ids),
                "groups": len({r["group"] for r in selected}),
                "class_ids": target,
                "status": "descriptive"
                if len({r["group"] for r in selected}) >= 5 and target
                else "insufficient",
            }
            for metric, field in (("AP50", "mAP50"), ("AP50_95", "mAP50_95")):
                raw, adaptive = (
                    float(np.mean([p[c][metric] for c in target])) if target else None
                    for p in (a, b)
                )
                entry.update(
                    {
                        f"raw_{field}": raw,
                        f"adaptive_{field}": adaptive,
                        f"delta_{field}": adaptive - raw if raw is not None else None,
                    }
                )
            comparisons.append(entry)
            for cls, label in enumerate(names):
                counts = {r["id"]: sum(int(box[0]) == cls for box in r["boxes"]) for r in selected}
                groups = len({r["group"] for r in selected if counts[r["id"]]})
                support = {
                    "model": model,
                    "route": route,
                    "timeofday": time,
                    "class": label,
                    "instances": sum(counts.values()),
                    "images": sum(n > 0 for n in counts.values()),
                    "groups": groups,
                    "status": "descriptive" if groups >= 5 else "insufficient",
                }
                for field in ("AP50", "AP50_95", "precision", "recall"):
                    raw, adaptive = a.get(cls, {}).get(field), b.get(cls, {}).get(field)
                    support.update({f"raw_{field}": raw, f"adaptive_{field}": adaptive})
                supports.append(support)
    result = {
        "populations": populations,
        "timeofday": crosstab,
        "comparisons": comparisons,
        "classes": supports,
        "skip_control": controls,
        "artifacts": {},
        "limitation": (
            "Descriptive matched input deltas; no route-subset CI or robust/causal claim. "
            "Night is context, not the frozen brightness-triggered Gamma route. "
            "Precision/recall use native subset max-F1 operating points above conf0.001; "
            "failure counts elsewhere use conf0.25/IoU0.50. Training-branch effects are separate."
        ),
        "final_test_accessed": False,
    }
    write_json(summary, {"identity": identity, "result": result, "checksum": digest(result)})
    return result


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
    evaluations, saved, configs, taxonomies = {}, {}, [], []
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
        taxonomies.append(bound.get("class_names", CLASSES))
        evaluations[name] = reuse_evaluation(path, bound)
        saved[name] = file_hash(path)

    if any(config != configs[0] for config in configs):
        raise ValueError("Evaluation matrix settings differ")
    class_names = taxonomies[0]
    if class_names not in (CLASSES, FIVE_CLASSES) or any(t != class_names for t in taxonomies):
        raise ValueError("Evaluation matrix taxonomies differ")
    config = configs[0]
    identity = {
        "version": "development-analysis-v4",
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
        if old.get("version") in (
            "development-analysis-v1",
            "development-analysis-v2",
            "development-analysis-v3",
        ):
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
    if class_names == FIVE_CLASSES:
        rows = [{**r, "boxes": remap_boxes(r["boxes"])} for r in rows]
    expected = {r["id"]: sorted(int(b[0]) for b in r["boxes"]) for r in rows}
    for records in stats.values():
        if set(records) != set(expected) or any(
            sorted(map(int, records[i]["target_cls"])) != classes for i, classes in expected.items()
        ):
            raise ValueError("Saved matching statistics differ from frozen development GT")

    slices = [("overall", "all", rows)]
    object_size_support = []
    slices.extend(
        (axis, value, [r for r in rows if r["attributes"][axis] == value])
        for axis in CONDITIONS
        for value in sorted({r["attributes"][axis] for r in rows})
    )
    # shortcut: size slices select whole images; use native area-aware AP for object-area claims.
    for size, low, high in (
        ("small", 0, 32**2),
        ("medium", 32**2, 96**2),
        ("large", 96**2, float("inf")),
    ):
        selected = [
            r
            for r in rows
            if any(low <= (x2 - x1) * (y2 - y1) < high for _, x1, y1, x2, y2 in r["boxes"])
        ]
        for cls, label in enumerate(class_names):
            counts = {
                r["id"]: sum(
                    c == cls and low <= (x2 - x1) * (y2 - y1) < high
                    for c, x1, y1, x2, y2 in r["boxes"]
                )
                for r in selected
            }
            groups = len({r["group"] for r in selected if counts[r["id"]]})
            object_size_support.append(
                {
                    "size": size,
                    "class": label,
                    "instances": sum(counts.values()),
                    "images": sum(n > 0 for n in counts.values()),
                    "source_groups": groups,
                    "status": "descriptive" if groups >= 5 else "insufficient",
                }
            )
        slices.append(("object_size_image_presence", size, selected))
    conditions = []
    for axis, value, selected in slices:
        ids = [r["id"] for r in selected]
        for name in names:
            per_class = class_ap(stats[name], ids, detailed=True)
            supported = {}
            for cls, label in enumerate(class_names):
                counts = {r["id"]: sum(b[0] == cls for b in r["boxes"]) for r in selected}
                groups = len({r["group"] for r in selected if counts[r["id"]]})
                supported[str(cls)] = {
                    "class": label,
                    "images": sum(n > 0 for n in counts.values()),
                    "instances": sum(counts.values()),
                    "source_groups": groups,
                    "status": "descriptive" if groups >= 5 else "insufficient",
                    "AP50": None,
                    "AP50_95": None,
                    "precision": None,
                    "recall": None,
                    **per_class.get(cls, {}),
                }
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
                    "classes": supported,
                }
            )

    comparisons = []
    pilot = {"rows": rows}
    for treatment in ("raw", "adaptive"):
        before, after = f"raw-{treatment}", f"adaptive-{treatment}"
        try:
            support = bootstrap_support(stats[before], pilot, class_names=class_names)
        except ValueError as exc:
            if "No classes meet" not in str(exc):
                raise
            comparisons.append(
                {
                    "before": before,
                    "after": after,
                    "axis": "overall",
                    "condition": "all",
                    "images": len(rows),
                    "groups": len({r["group"] for r in rows}),
                    "status": "unsupported",
                    "reason": str(exc),
                }
            )
            continue
        if resamples:
            try:
                interval = paired_interval(
                    stats[before], stats[after], pilot, resamples=resamples, class_names=class_names
                )
                status = "exploratory"
            except ValueError as exc:
                if "Fewer than two valid fixed-class draws" not in str(exc):
                    raise
                interval = {"estimand": support, "reason": str(exc)}
                status = "unsupported"
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
        "class_names": class_names,
        "object_size_support": object_size_support,
        "size_slice_semantics": (
            "Overlapping image-presence slices at original-pixel areas <32², 32²–96², ≥96²; "
            "all GT/predictions retained. Not object-area-restricted AP. No size-specific CI."
        ),
        "artifacts": {},
        "final_test_accessed": False,
        "limitation": (
            "Single-seed, exploratory two-contrast paired group analysis; "
            "conditional support is descriptive; no multiplicity adjustment"
        ),
    }
    write_json(summary, {"identity": identity, "result": result, "checksum": digest(result)})
    return result
