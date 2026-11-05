"""Measured exploration follow-ups for PR #38; no confirmation or policy release."""

import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .adaptive import validate_policy
from .common import digest, file_hash, freeze_json, read_json
from .evaluation_data import scenario_image
from .five_class import FIVE_CLASSES
from .preprocess import validate_image
from .preprocessing_experiments import (
    apply_candidate,
    export_candidate,
    validate_candidate,
    validate_sample,
)
from .quality_search import load_action, route_assignments


def additional_settings():
    """Refine Gamma's midpoint and half-strength unsharp with/without smoothing."""
    mild = {"sharpen": {"amount": 0.075, "sigma": 1.0}}
    return {
        "gamma_0775": {"steps": [{"gamma": {"gamma": 0.775}}]},
        "sharpen_0075": {"steps": [mild]},
        "sharpen_0075_then_bilateral": {"steps": [mild, {"bilateral": {}}]},
    }


def measure_operator_values(
    protocol,
    manifest,
    pilot,
    data_root,
    converted,
    output,
    v2,
    checkpoint,
    *,
    allow_inference=False,
):
    """Measure three predeclared actions using the existing native exporter and evaluator."""
    validate_sample(protocol, manifest, pilot)
    if pilot["sample_role"] != "calibration_exploration":
        raise ValueError("Additional settings may use exploration only")
    from .stress_eval import SETTINGS, evaluate_calibration

    output = Path(output)
    configs = additional_settings()
    plan = {
        "version": "issue30-additional-settings-v1",
        "pilot_id": pilot["pilot_id"],
        "manifest_id": manifest["manifest_id"],
        "configs": configs,
        "checkpoint_sha256": file_hash(checkpoint),
        "settings": SETTINGS,
        "opencv_threads": 8,
        "rationale": [
            "Gamma 0.775 is the midpoint of measured 0.70/0.85; test a milder correction.",
            "Halve measured unsharp 0.15 to 0.075; test reduced edge amplification.",
            "Retain the measured unsharp-then-bilateral order to isolate strength, not order.",
            "PR #37 denoise strength 5 failed confirmation and is not promoted.",
        ],
    }
    plan["plan_id"] = digest(plan)
    freeze_json(output / "plan.json", plan)
    cv2.setNumThreads(8)
    records = []
    for scenario in manifest["scenarios"]:
        for name, config in configs.items():
            directory = output / "metrics" / scenario["id"] / name
            if not allow_inference and not (directory / "native_result.json").is_file():
                return None
            meta = export_candidate(
                protocol,
                manifest,
                pilot,
                data_root,
                converted,
                output / "inputs" / scenario["id"] / name,
                scenario,
                config,
                v2,
            )
            result = evaluate_calibration(
                protocol,
                pilot,
                output / "inputs" / scenario["id"] / name,
                checkpoint,
                directory,
                allow_inference=allow_inference,
            )
            records.append(
                {
                    "candidate": name,
                    "scenario": scenario["id"],
                    "family": scenario["family"],
                    **result["overall"],
                    "classes": result["classes"],
                    "mean_ms": float(meta.restore_ms.mean()),
                    "p95_ms": float(meta.restore_ms.quantile(0.95)),
                    "native_result_sha256": file_hash(directory / "native_result.json"),
                    "statistics_sha256": file_hash(directory / "paired_stats.npz"),
                }
            )
            print(f"Additional settings: {scenario['id']}/{name} verified", flush=True)
    freeze_json(output / "measurements.json", records)
    return records


def load_operator_values(protocol, manifest, pilot, output, prior_plan):
    """Load only completed, verified measurements; an unmeasured action cannot enter search."""
    validate_sample(protocol, manifest, pilot)
    if pilot["sample_role"] != "calibration_exploration":
        raise ValueError("Additional settings may use exploration only")
    output = Path(output)
    plan = read_json(output / "plan.json")
    if (
        plan["plan_id"] != digest({k: v for k, v in plan.items() if k != "plan_id"})
        or plan["pilot_id"] != pilot["pilot_id"]
        or plan["manifest_id"] != manifest["manifest_id"]
        or plan["configs"] != additional_settings()
        or plan["checkpoint_sha256"] != prior_plan["checkpoint_sha256"]
        or plan["settings"] != prior_plan["settings"]
    ):
        raise ValueError("Additional operator plan changed")
    records = read_json(output / "measurements.json")
    expected = {(s["id"], n) for s in manifest["scenarios"] for n in plan["configs"]}
    if (
        len(records) != len(expected)
        or {(r["scenario"], r["candidate"]) for r in records} != expected
    ):
        raise ValueError("Every additional action requires every measured scenario")
    measurements = {(r["scenario"], r["candidate"]): r for r in records}
    return plan["configs"], {
        s["id"]: {
            name: load_action(
                protocol,
                manifest,
                pilot,
                s,
                name,
                config,
                measurements[(s["id"], name)],
                prior_plan,
                output,
            )
            for name, config in plan["configs"].items()
        }
        for s in manifest["scenarios"]
    }


def apply_quality_candidate(image, candidate, v2):
    """Match the frozen router's pixels, computing required features only, first-match first."""
    validate_image(image)
    validate_candidate(candidate)
    if "rules" not in candidate:
        return apply_candidate(image, candidate, v2)
    scores, small_gray = {}, None
    steps = []
    for rule in candidate["rules"]:
        metric = rule["metric"]
        if metric not in scores:
            if metric == "noise_highpass":
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
                kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
                response = cv2.filter2D(gray, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
                value = float(np.abs(response[1:-1, 1:-1]).mean() / 6 * np.sqrt(np.pi / 2))
            else:
                if small_gray is None:
                    scale = min(1, 512 / max(image.shape[:2]))
                    small = cv2.resize(
                        image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
                    )
                    small_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                if metric == "brightness":
                    value = float(small_gray.mean() / 255)
                elif metric == "contrast":
                    value = float(small_gray.std() / 255)
                else:
                    value = float(cv2.Laplacian(small_gray, cv2.CV_64F).var())
            scores[metric] = value
        value = scores[metric]
        if ("below" in rule and value < rule["below"]) or (
            "above" in rule and value > rule["above"]
        ):
            steps = rule["steps"]
            break
    return apply_candidate(image, {"steps": steps}, v2)


def benchmark_runtime(
    protocol,
    manifest,
    pilot,
    data_root,
    configs,
    v2,
    *,
    repeats=3,
    threads=8,
):
    """Same-process matched preprocessing timings, after warmup; exclude reads/corruption/JPEG."""
    validate_sample(protocol, manifest, pilot)
    if pilot["sample_role"] != "calibration_exploration":
        raise ValueError("Runtime measurement requires exploration only")
    if type(repeats) is not int or repeats < 1 or type(threads) is not int or threads < 1:
        raise ValueError("Runtime repeats/threads must be positive integers")
    if not {"skip", "frozen_v2"} <= configs.keys():
        raise ValueError("Runtime requires both controls")
    validate_policy(v2, protocol)
    for config in configs.values():
        validate_candidate(config)
    original_threads = cv2.getNumThreads()
    records = []
    try:
        cv2.setNumThreads(threads)
        names = list(configs)
        for scenario in manifest["scenarios"]:
            timings = {name: [] for name in names}
            for image_index, row in enumerate(pilot["rows"]):
                pixels = scenario_image(row, data_root, scenario, seed=manifest["seed"])
                if image_index == 0:
                    for name in names:
                        expected = apply_candidate(pixels, configs[name], v2)[0]
                        actual = apply_quality_candidate(pixels, configs[name], v2)[0]
                        if not np.array_equal(expected, actual):
                            raise ValueError("Selective routing changed the measured action pixels")
                for repeat in range(repeats):
                    offset = (image_index + repeat) % len(names)
                    for name in names[offset:] + names[:offset]:
                        tick = time.perf_counter()
                        apply_quality_candidate(pixels, configs[name], v2)
                        timings[name].append((time.perf_counter() - tick) * 1000)
            records.extend(
                {
                    "candidate": name,
                    "scenario": scenario["id"],
                    "family": scenario["family"],
                    "mean_ms": float(np.mean(values)),
                    "p95_ms": float(np.quantile(values, 0.95)),
                    "calls": len(values),
                }
                for name, values in timings.items()
            )
            print(f"Matched {threads}-thread runtime: {scenario['id']} complete", flush=True)
    finally:
        cv2.setNumThreads(original_threads)
    return {
        "threads": threads,
        "repeats": repeats,
        "configs": configs,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "records": records,
        "scope": "preprocessing only; exploratory, not confirmation",
    }


def pareto_comparison(search, additional=()):
    """Expose accuracy/cost tradeoffs rather than hiding a cheaper near-tie."""
    controls = pd.DataFrame(search["controls"])
    points = list(search["ranking"])
    points.extend(p for p in additional if p["candidate"] not in {r["candidate"] for r in points})
    for name, rows in controls.groupby("candidate"):
        points.append(
            {
                "candidate": name,
                "score": float(rows.groupby("family").mAP50_95.mean().mean()),
                "mean_ms": float(rows.mean_ms.mean()),
                "reasons": [],
            }
        )
    return [
        {
            "candidate": p["candidate"],
            "score": p["score"],
            "mean_ms": p["mean_ms"],
            "passes_exploration_regressions": not p["reasons"],
            "pareto": not any(
                q["score"] >= p["score"]
                and q["mean_ms"] <= p["mean_ms"]
                and (q["score"] > p["score"] or q["mean_ms"] < p["mean_ms"])
                for q in points
                if not q["reasons"]
            ),
        }
        for p in points
    ]


def clean_error_report(table, pilot, selected, actions):
    """Describe native IoU50 TP/FP at conf>=0.25, alongside the dataset AP report."""
    from .evaluation_stats import class_ap

    if pilot["sample_role"] != "calibration_exploration":
        raise ValueError("Clean routing inspection requires exploration only")
    features = table[table.scenario == "clean"].set_index("id")
    if len(features) != len(pilot["rows"]) or set(features.index) != {
        r["id"] for r in pilot["rows"]
    }:
        raise ValueError("Clean inspection must preserve paired IDs")
    assignments = route_assignments(features, selected["rules"])
    routed = {i: actions["clean"][name]["stats"][i] for i, name in assignments.items()}
    comparisons, examples = [], []
    for control in ("skip", "frozen_v2"):
        counts = []
        for sample_id in features.index:
            pair = []
            for stats in (actions["clean"][control]["stats"][sample_id], routed[sample_id]):
                keep = stats["conf"] >= 0.25
                pair.append(
                    {
                        "TP50": int(np.sum(stats["tp"][:, 0] & keep)),
                        "FP50": int(np.sum(~stats["tp"][:, 0] & keep)),
                    }
                )
            counts.append(
                {
                    "id": sample_id,
                    "action": assignments[sample_id],
                    "delta_TP50": pair[1]["TP50"] - pair[0]["TP50"],
                    "delta_FP50": pair[1]["FP50"] - pair[0]["FP50"],
                    "before": pair[0],
                    "after": pair[1],
                }
            )
        before = actions["clean"][control]["stats"]
        before_ap, after_ap = (
            class_ap(s, list(features.index), detailed=True) for s in (before, routed)
        )
        classes = []
        for cls, label in enumerate(FIVE_CLASSES):
            pair = []
            for values in (before, routed):
                tp = fp = 0
                for sample_id in features.index:
                    stats = values[sample_id]
                    keep = (stats["conf"] >= 0.25) & (stats["pred_cls"] == cls)
                    tp += int(np.sum(stats["tp"][:, 0] & keep))
                    fp += int(np.sum(~stats["tp"][:, 0] & keep))
                pair.append({"TP50": tp, "FP50": fp})
            classes.append(
                {
                    "class": label,
                    "before": pair[0],
                    "after": pair[1],
                    "delta_AP50_95": after_ap[cls]["AP50_95"] - before_ap[cls]["AP50_95"],
                    "delta_AP50": after_ap[cls]["AP50"] - before_ap[cls]["AP50"],
                }
            )
        comparisons.append(
            {
                "control": control,
                "before": {k: sum(r["before"][k] for r in counts) for k in ("TP50", "FP50")},
                "after": {k: sum(r["after"][k] for r in counts) for k in ("TP50", "FP50")},
                "TP_loss_images": sum(r["delta_TP50"] < 0 for r in counts),
                "FP_rise_images": sum(r["delta_FP50"] > 0 for r in counts),
                "classes": classes,
            }
        )
        examples.extend(
            {"control": control, **r}
            for r in sorted(counts, key=lambda r: (r["delta_TP50"], -r["delta_FP50"], r["id"]))
            if r["delta_TP50"] < 0 or r["delta_FP50"] > 0
        )
    clean = next(r for r in selected["metrics"] if r["scenario"] == "clean")
    return {
        "routes": assignments.value_counts().to_dict(),
        "comparisons": comparisons,
        "failure_examples": examples[:8],
        "classes": clean["classes"],
        "subgroups": clean["subgroups"],
        "limit": "IoU50/conf0.25 counts are illustrative, not a replacement for native AP",
    }


def plot_review_overview(table, pilot, manifest, data_root, search, pareto, clean, destination):
    """One small committed figure: overlapping features, thresholds, cost and real error cases."""
    import matplotlib.pyplot as plt

    selected = search["selected"]
    figure, axes = plt.subplots(3, 2, figsize=(13, 13))
    rules = selected["rules"] if selected else []
    features = list(dict.fromkeys([r["metric"] for r in rules] + ["contrast", "noise_highpass"]))[
        :2
    ]
    for ax, feature in zip(axes[0], features, strict=True):
        for family, rows in table.groupby("family"):
            ax.hist(rows[feature], bins=35, density=True, histtype="step", label=family)
        for rule in rules:
            if rule["metric"] == feature:
                ax.axvline(rule.get("above", rule.get("below")), color="black", linestyle="--")
        ax.set_title(f"Exploration {feature}; dashed = chosen threshold")
        ax.legend(fontsize=7)
    points = pd.DataFrame(pareto)
    ax = axes[1, 0]
    ax.scatter(points.mean_ms, points.score * 100, s=12, alpha=0.35, label="screening estimates")
    front = points[points.pareto & points.passes_exploration_regressions].sort_values("mean_ms")
    ax.plot(front.mean_ms, front.score * 100, "o-", markersize=4, label="eligible Pareto front")
    for name in ("skip", "frozen_v2", selected["candidate"] if selected else "skip"):
        row = points[points.candidate == name].iloc[0]
        ax.annotate(name[:12], (row.mean_ms, row.score * 100), fontsize=7)
    ax.set(xlabel="Estimated preprocessing ms", ylabel="Family-balanced mAP50:95 (%)")
    ax.legend(fontsize=7)
    ax = axes[1, 1]
    ax.barh(list(clean["routes"]), list(clean["routes"].values()))
    ax.set(xlabel="Clean exploration images", title="Clean routing, including Skip")
    ax.tick_params(axis="y", labelsize=7)
    rows = {r["id"]: r for r in pilot["rows"]}
    examples = []
    for example in clean["failure_examples"]:
        if example["id"] not in {e["id"] for e in examples}:
            examples.append(example)
        if len(examples) == 2:
            break
    for ax, example in zip(axes[2], examples, strict=False):
        pixels = scenario_image(
            rows[example["id"]], data_root, manifest["scenarios"][0], seed=manifest["seed"]
        )
        ax.imshow(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB))
        ax.set_title(
            f"{example['id']} vs {example['control']}\n"
            f"TP50 delta {example['delta_TP50']:+d}; FP50 delta {example['delta_FP50']:+d}",
            fontsize=9,
        )
        ax.axis("off")
    for ax in axes[2][len(examples) :]:
        ax.text(0.5, 0.5, "No additional TP-loss/FP-rise examples", ha="center")
        ax.axis("off")
    figure.suptitle("Exploration only — no confirmed v3; counts at conf ≥0.25, IoU50")
    figure.tight_layout()
    figure.savefig(destination, dpi=120)
    plt.close(figure)


def plot_runtime_comparison(runtime, destination):
    """Show direct matched mean/p95 against the declared per-scenario budgets."""
    import matplotlib.pyplot as plt

    records = pd.DataFrame(runtime["records"])
    worst = records.groupby("candidate")[["mean_ms", "p95_ms"]].max()
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, limit in zip(axes, ("mean_ms", "p95_ms"), (10, 25), strict=True):
        ax.barh([name[:16] for name in worst.index], worst[metric])
        ax.axvline(limit, color="red", linestyle="--", label=f"budget {limit} ms")
        ax.set(
            xlabel=f"Worst-scenario {metric}", title=f"Direct {runtime['threads']}-thread runtime"
        )
        ax.legend()
    figure.suptitle("Same process, warmup, rotated candidate order; reads/corruption/JPEG excluded")
    figure.tight_layout()
    figure.savefig(destination, dpi=120)
    plt.close(figure)
