"""Exploration-only quality routing from preserved native matches; never release a policy."""

import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .common import digest, file_hash, read_json
from .evaluation_data import scenario_image
from .five_class import FIVE_CLASSES, remap_boxes, support_audit
from .preprocessing_experiments import (
    confirmation_decision,
    detection_report,
    rank_candidates,
    validate_candidate,
    validate_manifest,
    validate_sample,
)
from .protocol import summarize
from .stress import diagnostics, validate_pilot

FEATURES = ("brightness", "contrast", "noise_highpass", "sharpness")
QUANTILES = (0.25, 0.5, 0.75)


def diverse_beam(trials):
    """Retain four leaders and add four diverse rules without dropping a useful leader."""
    chosen, remaining = list(trials[:4]), list(trials[4:])
    while remaining and len(chosen) < 8:
        keys = [
            (
                r["metric"],
                "above" if "above" in r else "below",
                r["action"],
                r.get("above", r.get("below")),
            )
            for t in chosen
            for r in t["rules"]
        ]

        def diversity(trial, keys=keys):
            r = trial["rules"][0]
            key = (
                r["metric"],
                "above" if "above" in r else "below",
                r["action"],
                r.get("above", r.get("below")),
            )
            fresh = [not any(k[j] == value for k in keys) for j, value in enumerate(key)]
            return (sum(fresh), fresh[0], fresh[2])

        best = max(remaining, key=diversity)
        chosen.append(best)
        remaining.remove(best)
    return [t["rules"][0] for t in chosen]


def route_assignments(features, rules):
    """First-match actions from measured image features, with conditional Skip."""
    chosen = pd.Series("skip", index=features.index)
    remaining = pd.Series(True, index=features.index)
    for rule in rules:
        values = features[rule["metric"]]
        match = values < rule["below"] if "below" in rule else values > rule["above"]
        chosen.loc[match & remaining] = rule["action"]
        remaining.loc[match] = False
    return chosen


def collect_quality(protocol, manifest, pilot, data_root):
    """Measure only exploration pixels before treatment/JPEG; annotations describe support."""
    validate_sample(protocol, manifest, pilot)
    if pilot["sample_role"] != "calibration_exploration":
        raise ValueError("Quality search may inspect exploration pixels only")
    records = []
    for scenario in manifest["scenarios"]:
        for row in pilot["rows"]:
            pixels = scenario_image(row, data_root, scenario, seed=manifest["seed"])
            tick = time.perf_counter()
            scores = diagnostics(pixels)
            records.append(
                {
                    "id": row["id"],
                    "group": row["group"],
                    "scenario": scenario["id"],
                    "family": scenario["family"],
                    "severity": scenario["severity"],
                    **row["attributes"],
                    **scores,
                    "routing_ms": (time.perf_counter() - tick) * 1000,
                }
            )
    return pd.DataFrame(records)


def _quality_index(table, pilot, scenarios):
    expected = {(s["id"], r["id"]) for s in scenarios for r in pilot["rows"]}
    keys = list(zip(table.scenario, table.id, strict=True))
    groups = {r["id"]: r["group"] for r in pilot["rows"]}
    values = table[[*FEATURES, "noise", "black_fraction", "white_fraction", "routing_ms"]]
    if (
        len(keys) != len(expected)
        or set(keys) != expected
        or any(groups.get(r.id) != r.group for r in table.itertuples())
        or not np.isfinite(values.to_numpy()).all()
        or (values < 0).any().any()
    ):
        raise ValueError("Quality rows must be finite, paired exploration IDs/groups/scenarios")
    return table.set_index(["scenario", "id"])


def quality_summary(table, pilot, scenarios):
    """Retain ranges, overlapping proxies, condition/group support and representative IDs."""
    _quality_index(table, pilot, scenarios)
    distributions, representatives = [], []
    for scenario in scenarios:
        rows = table[table.scenario == scenario["id"]]
        for feature in (*FEATURES, "noise", "black_fraction", "white_fraction"):
            values = rows[feature]
            distributions.append(
                {
                    "scenario": scenario["id"],
                    "family": scenario["family"],
                    "severity": scenario["severity"],
                    "feature": feature,
                    "quantiles": {
                        str(q): float(values.quantile(q)) for q in (0, 0.1, *QUANTILES, 0.9, 1)
                    },
                    "zero_fraction": float((values == 0).mean()),
                }
            )
            for q in (0.1, 0.5, 0.9):
                nearest = (values - values.quantile(q)).abs().idxmin()
                representatives.append(
                    {
                        "scenario": scenario["id"],
                        "feature": feature,
                        "quantile": q,
                        "id": rows.loc[nearest, "id"],
                        "value": float(rows.loc[nearest, feature]),
                    }
                )
    support = []
    for cls, name in enumerate(FIVE_CLASSES):
        rows = [r for r in pilot["rows"] if cls in {b[0] for b in remap_boxes(r["boxes"])}]
        support.append(
            {"class": name, "images": len(rows), "groups": len({r["group"] for r in rows})}
        )
    return {
        "distributions": distributions,
        "representatives": representatives,
        "class_support": support,
        "annotation_support": support_audit(pilot["rows"]),
        "source_groups": len({r["group"] for r in pilot["rows"]}),
        "limits": [
            "Noise/high-pass and Laplacian sharpness respond to texture and each other.",
            "Clipping can collapse the median-residual noise proxy to zero.",
            "Overlapping clean/perturbed ranges cannot identify a corruption reliably.",
            "Quality is measured before the one final JPEG95 encode, as at routing time.",
            "Five source groups per class is an engineering floor, not adequate uncertainty.",
        ],
    }


def load_exploration_actions(protocol, manifest, pilot, report, run_root):
    """Read only hash-bound PR #37 exploration statistics and per-image operator costs."""
    validate_sample(protocol, manifest, pilot)
    if pilot["sample_role"] != "calibration_exploration":
        raise ValueError("Search cannot load confirmation/development outcomes")
    report, run_root = Path(report), Path(run_root)
    plan = read_json(report / "plan.json")
    if (
        plan["plan_id"] != digest({k: v for k, v in plan.items() if k != "plan_id"})
        or plan["pilots"]["exploration"] != pilot["pilot_id"]
        or plan["manifest_id"] != manifest["manifest_id"]
    ):
        raise ValueError("Preserved exploration plan differs")
    bound = read_json(report / "evidence_manifest.json")["artifacts"]
    if file_hash(report / "plan.json") != bound.get(f"docs/{report.name}/plan.json"):
        raise ValueError("Preserved exploration plan changed")
    measurements = {}
    for path in sorted(report.glob("exploration-*.json")):
        if file_hash(path) != bound.get(f"docs/{report.name}/{path.name}"):
            raise ValueError("Preserved exploration report changed")
        for record in read_json(path):
            key = (record["scenario"], record["candidate"])
            if key in measurements and measurements[key] != record:
                raise ValueError("Conflicting preserved exploration measurements")
            measurements[key] = record
    configs = {
        name: config
        for name, config in plan["candidates"].items()
        if name == "frozen_v2" or "steps" in config
    }
    actions = {
        scenario["id"]: {
            name: load_action(
                protocol,
                manifest,
                pilot,
                scenario,
                name,
                config,
                measurements[(scenario["id"], name)],
                plan,
                run_root / "exploration",
            )
            for name, config in configs.items()
        }
        for scenario in manifest["scenarios"]
    }
    return configs, actions


def load_action(protocol, manifest, pilot, scenario, name, config, record, plan, root):
    """Verify one measured native action and its paired cost inventory."""
    from .stress_eval import load_recorded_calibration

    root = Path(root)
    expected_gt = {r["id"]: sorted(b[0] for b in remap_boxes(r["boxes"])) for r in pilot["rows"]}
    directory = root / "metrics" / scenario["id"] / name
    cache = read_json(directory / "native_result.json")
    identity = cache["identity"]
    export = identity["export"]
    if (
        export["candidate"] != config
        or export["scenario"] != scenario
        or export["manifest_id"] != manifest["manifest_id"]
        or export["v2_policy_id"] != plan["v2_policy_id"]
    ):
        raise ValueError("Cached action differs from the preserved experiment")
    _, stats = load_recorded_calibration(
        directory,
        {**record, "files_identity": identity["files_identity"]},
        pilot=pilot,
        protocol_id=protocol["manifest_id"],
        checkpoint=plan["checkpoint_sha256"],
        settings=plan["settings"],
        code=plan["implementation"]["stress_eval.py"],
    )
    if any(sorted(map(int, stats[i]["target_cls"])) != gt for i, gt in expected_gt.items()):
        raise ValueError("Cached action differs from the five-class annotations")
    input_root = root / "inputs" / scenario["id"] / name
    metadata = pd.read_csv(input_root / "metadata.csv")
    verified = read_json(input_root / "verified.json")
    if (
        verified["files_identity"] != identity["files_identity"]
        or verified["metadata_sha256"] != file_hash(input_root / "metadata.csv")
        or verified["export_sha256"] != file_hash(input_root / "export.json")
        or read_json(input_root / "export.json") != export
    ):
        raise ValueError("Action input inventory changed")
    if len(metadata) != len(expected_gt) or set(metadata.id) != set(expected_gt):
        raise ValueError("Action costs differ from exploration IDs")
    return {
        "stats": stats,
        "costs": metadata.set_index("id").restore_ms.to_dict(),
    }


def search_rules(table, pilot, scenarios, configs, actions, *, max_mean_ms, max_p95_ms):
    """Search quantile rules and a diverse eight-rule beam using native dataset AP."""
    from .evaluation_stats import class_ap

    if any(
        type(v) not in (int, float) or not np.isfinite(v) or v <= 0
        for v in (max_mean_ms, max_p95_ms)
    ):
        raise ValueError("Cost limits must be finite positive milliseconds")
    if pilot.get("sample_role") != "calibration_exploration":
        raise ValueError("Rule optimization requires exploration only")
    indexed = _quality_index(table, pilot, scenarios)
    for config in configs.values():
        validate_candidate(config)
        if config != {"v2": True} and "steps" not in config:
            raise ValueError("Cached search actions must be fixed operator sequences")
    if configs.get("skip") != {"steps": []} or configs.get("frozen_v2") != {"v2": True}:
        raise ValueError("Search requires unchanged Skip/v2 controls")
    ids = [r["id"] for r in pilot["rows"]]
    minimum_groups = min(
        len({r["group"] for r in pilot["rows"] if cls in {b[0] for b in remap_boxes(r["boxes"])}})
        for cls in range(len(FIVE_CLASSES))
    )
    if minimum_groups < 5:
        raise ValueError("Search requires five source groups per target class")
    if set(actions) != {s["id"] for s in scenarios}:
        raise ValueError("Action scenarios differ from quality measurements")
    for variants in actions.values():
        if set(variants) != set(configs):
            raise ValueError("Action configurations differ across scenarios")
        for action in variants.values():
            if set(action["stats"]) != set(ids) or set(action["costs"]) != set(ids):
                raise ValueError("Action statistics/costs must preserve paired IDs")
            if any(not np.isfinite(v) or v < 0 for v in action["costs"].values()):
                raise ValueError("Action costs must be finite nonnegative milliseconds")
            for row in pilot["rows"]:
                expected = sorted(b[0] for b in remap_boxes(row["boxes"]))
                if sorted(map(int, action["stats"][row["id"]]["target_cls"])) != expected:
                    raise ValueError("Action target classes differ from paired annotations")
    thresholds = {f: sorted(set(float(table[f].quantile(q)) for q in QUANTILES)) for f in FEATURES}
    trials, seen = [], set()

    def measure(name, rules=None):
        records, signature = [], []
        for scenario in scenarios:
            key = scenario["id"]
            chosen = dict.fromkeys(ids, name if rules is None else "skip")
            if rules is not None:
                chosen = route_assignments(indexed.loc[key].loc[ids], rules).to_dict()
            signature.extend(chosen.values())
            costs = [
                actions[key][action]["costs"][i]
                + (float(indexed.loc[(key, i), "routing_ms"]) if rules is not None else 0)
                for i, action in chosen.items()
            ]
            mean_ms, p95_ms = float(np.mean(costs)), float(np.quantile(costs, 0.95))
            if rules is not None and (mean_ms > max_mean_ms or p95_ms > max_p95_ms):
                return None
            stats = {i: actions[key][action]["stats"][i] for i, action in chosen.items()}
            per_class = class_ap(stats, ids, detailed=True)
            if set(per_class) != set(range(len(FIVE_CLASSES))):
                raise ValueError("Native statistics must support the fixed five classes")
            records.append(
                {
                    "candidate": name,
                    "scenario": key,
                    "family": scenario["family"],
                    "mAP50_95": float(np.mean([v["AP50_95"] for v in per_class.values()])),
                    "mAP50": float(np.mean([v["AP50"] for v in per_class.values()])),
                    "classes": [
                        {"class": label, **per_class[c]} for c, label in enumerate(FIVE_CLASSES)
                    ],
                    "full_class_support": True,
                    "minimum_class_groups": minimum_groups,
                    "mean_ms": mean_ms,
                    "p95_ms": p95_ms,
                    "routes": pd.Series(chosen).value_counts().to_dict(),
                }
            )
        signature = tuple(signature)
        if rules is not None and signature in seen:
            return None
        seen.add(signature)
        return records

    controls = measure("skip") + measure("frozen_v2")

    def evaluate(rules):
        name = digest(rules)[:16]
        records = measure(name, rules)
        if records is None:
            return
        config = {
            "rules": [
                {k: v for k, v in r.items() if k != "action"}
                | {"steps": configs[r["action"]]["steps"]}
                for r in rules
            ]
        }
        # Reuse the frozen accuracy/class-regression gates only; this is exploration.
        gates = confirmation_decision(
            pd.DataFrame([*controls, *records]), scenarios, {"candidate": name}, {name: config}, {}
        )
        ranked = next(r for r in gates["confirmation_ranking"] if r["candidate"] == name)
        trials.append(
            {
                **ranked,
                "rules": rules,
                "config": config,
                "reasons": gates["reasons"],
                "metrics": records,
            }
        )

    for feature, values in thresholds.items():
        for threshold in values:
            for direction in ("below", "above"):
                for action, config in sorted(configs.items()):
                    if action != "skip" and "steps" in config:
                        evaluate([{"metric": feature, direction: threshold, "action": action}])
    trials.sort(key=lambda r: (-r["score"], r["mean_ms"], r["candidate"]))
    beam = diverse_beam(trials)
    for first in beam:
        for second in beam:
            if first != second:
                evaluate([first, second])
    baseline = max(r["score"] for r in rank_candidates(pd.DataFrame(controls), scenarios))
    trials.sort(key=lambda r: (-r["score"], r["mean_ms"], len(r["rules"]), r["candidate"]))
    seed = next((t for t in trials if not t["reasons"] and t["score"] > baseline), None)
    if seed:
        for position, rule in enumerate(seed["rules"]):
            direction = "above" if "above" in rule else "below"
            values = table[rule["metric"]]
            q = min(QUANTILES, key=lambda q: abs(float(values.quantile(q)) - rule[direction]))
            for nearby in (q - 0.05, q + 0.05):
                refined = [dict(r) for r in seed["rules"]]
                refined[position][direction] = float(values.quantile(nearby))
                evaluate(refined)
    trials.sort(key=lambda r: (-r["score"], r["mean_ms"], len(r["rules"]), r["candidate"]))
    eligible = [t for t in trials if not t["reasons"] and t["score"] > baseline]
    selected = eligible[0] if eligible else None
    if selected:
        # Retain subgroup AP and representative errors only for the locked route.
        for record in selected["metrics"]:
            key = record["scenario"]
            assignments = route_assignments(indexed.loc[key].loc[ids], selected["rules"])
            routed = {i: actions[key][action]["stats"][i] for i, action in assignments.items()}
            record.update(detection_report(pilot, routed, actions[key]["skip"]["stats"]))
    result = {
        "status": "pending_fresh_confirmation" if selected else "no_exploration_improvement",
        "selected": selected,
        "controls": controls,
        "ranking": [{k: v for k, v in t.items() if k != "metrics"} for t in trials],
        "thresholds": thresholds,
        "quantiles": list(QUANTILES),
        "beam": beam,
        "local_refinement": "one pass, +/-0.05 quantile per rule of the eligible leader",
        "cost_limits_ms": {"mean_per_scenario": max_mean_ms, "p95_per_scenario": max_p95_ms},
        "quality_sha256": digest(table.to_dict("records")),
        "pilot_id": pilot["pilot_id"],
        "limits": [
            "Only actions supplied with measured native matches are searched.",
            "24 single rules per action, 56 bounded ordered pairs, at most four local refinements.",
            "Cost is cached action time plus measured routing; direct runtime must be confirmed.",
            "No confirmation/development outcomes enter optimization; no v3 is released.",
        ],
    }
    result["search_id"] = digest(result)
    return result


def fresh_confirmation(protocol, manifest, used_pilots, *, target_images=256):
    """Plan unused whole confirmation groups using metadata only, without opening pixels."""
    validate_manifest(manifest, protocol)
    if type(target_images) is not int or target_images < 1 or not used_pilots:
        raise ValueError("Require a positive sample target and previously used pilots")
    excluded = set()
    for pilot in used_pilots:
        validate_pilot(pilot, protocol)
        excluded.update(r["group"] for r in pilot["rows"])
        excluded.update(r["source_group"] for r in pilot["rows"])
    allowed = set(manifest["sets"]["calibration_confirmation"])
    groups = defaultdict(list)
    for row in protocol["splits"]["calibration"]:
        if row["id"] in allowed and not {row["group"], row["source_group"]} & excluded:
            groups[row["group"]].append(row)
    rows = []
    for group in sorted(groups, key=lambda g: (digest([manifest["seed"], "quality-search", g]), g)):
        rows.extend(groups[group])
        if len(rows) >= target_images:
            break
    pilot = {
        "version": "issue30-unused-confirmation-v1",
        "protocol_id": protocol["manifest_id"],
        "evaluation_manifest_id": manifest["manifest_id"],
        "role": "calibration",
        "sample_role": "calibration_confirmation",
        "target_images": target_images,
        "excluded_groups_sha256": digest(sorted(excluded)),
        "used_pilot_ids": sorted(p["pilot_id"] for p in used_pilots),
        "rows": sorted(rows, key=lambda r: r["id"]),
        "summary": summarize(rows),
    }
    pilot["pilot_id"] = digest(pilot)
    if rows:
        validate_pilot(pilot, protocol)
    return {
        "pilot": pilot,
        "available_images": sum(len(v) for v in groups.values()),
        "available_groups": len(groups),
        "target_met": len(rows) >= target_images,
        "support": support_audit(rows),
        "class_groups": {
            label: len({r["group"] for r in rows if cls in {b[0] for b in remap_boxes(r["boxes"])}})
            for cls, label in enumerate(FIVE_CLASSES)
        },
        "status": "metadata_only_not_evaluated",
        "limit": "Unused within this round; historical v2 calibration is not an untouched holdout.",
    }
