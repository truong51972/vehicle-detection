"""Verify and interpret the fixed round-3 caches without inference or source-role changes."""

import gzip
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from .common import digest, file_hash, freeze_json, read_image, read_json
from .evaluation_stats import paired_interval
from .five_class import FIVE_CLASSES, remap_boxes, support_audit
from .preprocessing_experiments import detection_report
from .research_round3 import OUTPUT, REPORT, exploration_access, metric_record
from .stress_eval import load_recorded_calibration


def family_interval(intervals, scenarios):
    """Use the same source-group draw across scenarios before equal-family averaging."""
    draws = []
    for index in range(100):
        families = {}
        missing = []
        for scenario in scenarios:
            row = intervals[scenario["id"]]["draws"][index]
            if not row["valid"]:
                missing.append(scenario["id"])
            else:
                families.setdefault(scenario["family"], []).append(row["recovery"])
        value = None if missing else float(np.mean([np.mean(v) for v in families.values()]))
        draws.append({"draw": index, "recovery": value, "missing_scenarios": missing})
    values = [d["recovery"] for d in draws if d["recovery"] is not None]
    if len(values) < 2:
        raise ValueError("Too few complete fixed-class source-group draws")
    point = {}
    for scenario in scenarios:
        point.setdefault(scenario["family"], []).append(intervals[scenario["id"]]["recovery"])
    return {
        "recovery": float(np.mean([np.mean(v) for v in point.values()])),
        "ci95": np.quantile(values, [0.025, 0.975]).tolist(),
        "valid_draws": len(values),
        "invalid_draws": 100 - len(values),
        "draws": draws,
        "limit": "Exploration, no selection/multiplicity adjustment; not confirmation",
    }


def errors(stats, ids, cls=None):
    tp = fp = 0
    for sample_id in ids:
        record = stats[sample_id]
        keep = record["conf"] >= 0.25
        if cls is not None:
            keep &= record["pred_cls"] == cls
        found = int(record["tp"][keep, 0].sum())
        tp += found
        fp += int(keep.sum()) - found
    return {"TP50": tp, "FP50": fp}


def verify_artifacts():
    if (REPORT / "evidence.json").exists():
        for path, expected in read_json(REPORT / "evidence.json")["artifacts"].items():
            if file_hash(path) != expected:
                raise ValueError(f"Round-3 evidence changed: {path}")


def build_report():
    verify_artifacts()
    plan, summary = (read_json(REPORT / n) for n in ("plan.json", "summary.json"))
    if plan["plan_id"] != digest({k: v for k, v in plan.items() if k != "plan_id"}):
        raise ValueError("Preregistered plan changed")
    for name, expected in plan["code"].items():
        if file_hash(Path(__file__).with_name(name)) != expected:
            raise ValueError("Measurement implementation changed")
    pilot = read_json(OUTPUT / "sample.json")
    if pilot["pilot_id"] != plan["pilot_id"] or digest(pilot["rows"]) != plan["rows_sha256"]:
        raise ValueError("Sample changed")
    historical = read_json("docs/round2-preprocessing-v2/samples.json")["samples"]["exploration"]
    old_groups = {r["group"] for r in historical["rows"]}
    new_rows = [r for r in pilot["rows"] if r["group"] not in old_groups]
    if digest([r["id"] for r in new_rows]) != plan["new_ids_sha256"]:
        raise ValueError("New exploration membership changed")
    # Bootstrap original source groups; merged leakage groups remain intact in sampling.
    source_pilot = {"rows": [{**r, "group": r["source_group"]} for r in new_rows]}
    ids = [r["id"] for r in new_rows]
    membership = (
        pd.DataFrame(
            [
                {
                    "id": r["id"],
                    "group": r["group"],
                    "source_group": r["source_group"],
                    "source_sha256": r["sha256"],
                    "slice": "old" if r["group"] in old_groups else "new",
                }
                for r in pilot["rows"]
            ]
        )
        .to_csv(index=False)
        .encode()
    )
    compressed_membership = gzip.compress(membership, mtime=0)
    membership_path = REPORT / "membership.csv.gz"
    if membership_path.exists() and membership_path.read_bytes() != compressed_membership:
        raise ValueError("Frozen membership changed")
    membership_path.write_bytes(compressed_membership)
    stats, routing = {}, []
    for record in read_json(REPORT / "measurements.json"):
        scenario, name = record["scenario"], record["candidate"]
        directory = OUTPUT / "metrics" / scenario / name
        identity = read_json(directory / "native_result.json")["identity"]
        if identity["export"]["candidate"] != plan["configs"][name]:
            raise ValueError("Cached treatment changed")
        result, full = load_recorded_calibration(
            directory,
            {**record, "files_identity": identity["files_identity"]},
            pilot=pilot,
            protocol_id=plan["protocol_id"],
            checkpoint=plan["checkpoint_sha256"],
            settings=plan["settings"],
            code=plan["code"]["stress_eval.py"],
        )
        if result["overall"]["mAP50_95"] != record["mAP50_95"]:
            raise ValueError("Recorded native AP changed")
        aggregate = metric_record(full, pilot["rows"])
        if any(
            abs(aggregate[key] - result["overall"][key]) > 1e-8 for key in ("mAP50_95", "mAP50")
        ):
            raise ValueError("Native fixed-class reaggregation differs")
        stats[scenario, name] = {i: full[i] for i in ids}
        metadata_path = OUTPUT / "inputs" / scenario / name / "metadata.csv"
        metadata = pd.read_csv(metadata_path)
        if metadata.id.duplicated().any() or set(metadata.id) != set(full):
            raise ValueError("Routing inventory differs from native evaluation IDs")
        for label, chosen in (("all", list(full)), ("new", ids)):
            rows = metadata[metadata.id.isin(chosen)]
            routing.append(
                {
                    "candidate": name,
                    "scenario": scenario,
                    "slice": label,
                    "actions": rows.decisions.value_counts().to_dict(),
                    "processed_fraction": float((rows.decisions != "[]").mean()),
                    "metadata_sha256": file_hash(metadata_path),
                }
            )
    reports, intervals, family = [], {}, []
    scenarios = summary["scenarios"]
    for name in ("pr38_locked", "bilateral", "noise_range"):
        for control in ("skip", "frozen_v2"):
            pair = {}
            for scenario in scenarios:
                key = scenario["id"]
                before, after = stats[key, control], stats[key, name]
                native = detection_report(source_pilot, after, before)
                changes = []
                for i in ids:
                    a, b = errors(before, [i]), errors(after, [i])
                    changes.append(
                        {
                            "id": i,
                            "before": a,
                            "after": b,
                            "delta_TP50": b["TP50"] - a["TP50"],
                            "delta_FP50": b["FP50"] - a["FP50"],
                        }
                    )
                reports.append(
                    {
                        "candidate": name,
                        "control": control,
                        "scenario": key,
                        "before": errors(before, ids),
                        "after": errors(after, ids),
                        "images_losing_TP": sum(r["delta_TP50"] < 0 for r in changes),
                        "images_gaining_FP": sum(r["delta_FP50"] > 0 for r in changes),
                        "class_errors": [
                            {
                                "class": c,
                                "before": errors(before, ids, j),
                                "after": errors(after, ids, j),
                            }
                            for j, c in enumerate(FIVE_CLASSES)
                        ],
                        "worst_TP": sorted(changes, key=lambda r: (r["delta_TP50"], r["id"]))[:4],
                        "worst_FP": sorted(changes, key=lambda r: (-r["delta_FP50"], r["id"]))[:4],
                        **native,
                    }
                )
                interval = paired_interval(
                    before, after, source_pilot, resamples=100, seed=2026, class_names=FIVE_CLASSES
                )
                if interval["estimand"]["class_ids"] != list(range(5)):
                    raise ValueError("Bootstrap cannot silently drop a target class")
                pair[key] = interval
                print(f"Uncertainty verified: {name}/{control}/{key}", flush=True)
            intervals[name + "/" + control] = pair
            family.append(
                {"candidate": name, "control": control, **family_interval(pair, scenarios)}
            )
    freeze_json(REPORT / "routing.json", routing)
    freeze_json(REPORT / "failures.json", reports)
    freeze_json(REPORT / "uncertainty.json", family)
    compressed = gzip.compress(
        json.dumps(intervals, sort_keys=True, allow_nan=False).encode(), mtime=0
    )
    path = REPORT / "uncertainty_draws.json.gz"
    if path.exists() and path.read_bytes() != compressed:
        raise ValueError("Immutable bootstrap evidence changed")
    path.write_bytes(compressed)
    freeze_json(
        REPORT / "support.json",
        {
            "new": support_audit(new_rows),
            "all": support_audit(pilot["rows"]),
            "source_groups": plan["class_source_groups"],
            "new_images": len(ids),
            "new_groups": len({r["source_group"] for r in new_rows}),
        },
    )
    runtime = read_json(REPORT / "runtime.json")
    if runtime["configs"] != plan["configs"] or runtime["threads"] != 8 or runtime["repeats"] != 3:
        raise ValueError("Runtime settings differ from the fixed comparison")
    table = pd.DataFrame(runtime["records"])
    if (
        len(table) != 50
        or not (table.calls == 1536).all()
        or table.duplicated(["candidate", "scenario"]).any()
    ):
        raise ValueError("Incomplete matched runtime matrix")
    cost = []
    for name, rows in table.groupby("candidate"):
        cost.append(
            {
                "candidate": name,
                "mean_ms": float(rows.mean_ms.mean()),
                "worst_scenario_mean_ms": float(rows.mean_ms.max()),
                "worst_scenario_p95_ms": float(rows.p95_ms.max()),
                "passes": bool((rows.mean_ms <= 10).all() and (rows.p95_ms <= 25).all()),
            }
        )
    freeze_json(REPORT / "cost.json", cost)
    slices = pd.DataFrame(read_json(REPORT / "slices.json"))
    scores = (
        slices.groupby(["slice", "candidate", "family"])
        .mAP50_95.mean()
        .groupby(["slice", "candidate"])
        .mean()
    )
    pareto = []
    for c in cost:
        score = float(scores.loc[("new", c["candidate"])])
        pareto.append({**c, "score": score})
    accuracy_reasons = {r["candidate"]: r["reasons"] for r in summary["decision"]["candidates"]}
    for p in pareto:
        p["accuracy_reasons"] = accuracy_reasons.get(p["candidate"], [])
        p["passes_accuracy_gates"] = not p["accuracy_reasons"]
        p["pareto"] = not any(
            q["score"] >= p["score"]
            and q["mean_ms"] <= p["mean_ms"]
            and (q["score"] > p["score"] or q["mean_ms"] < p["mean_ms"])
            for q in pareto
        )
    freeze_json(REPORT / "pareto.json", pareto)
    plot_results(slices, pareto, family, scenarios)
    plot_failures(pilot, reports)
    artifacts = [
        p
        for p in REPORT.iterdir()
        if p.name not in ("evidence.json", "validation.json") and p.is_file()
    ]
    artifacts += [OUTPUT / "sample.json", Path(__file__)]
    freeze_json(
        REPORT / "evidence.json",
        {
            "plan_id": plan["plan_id"],
            "artifacts": {str(p): file_hash(p) for p in sorted(artifacts)},
            "status": summary["status"],
            "v3_released": False,
        },
    )


def plot_results(slices, cost, uncertainty, scenarios):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    names = [r["candidate"] for r in cost]
    scores = (
        slices.groupby(["slice", "candidate", "family"])
        .mAP50_95.mean()
        .groupby(["slice", "candidate"])
        .mean()
    )
    x = np.arange(len(names))
    for offset, label in ((-0.2, "old"), (0.2, "new")):
        axes[0, 0].bar(
            x + offset, [100 * scores.loc[(label, n)] for n in names], 0.4, label=label + " 256"
        )
    axes[0, 0].set_xticks(x, names, rotation=20)
    axes[0, 0].set_ylabel("Five-class AP50:95 (%)")
    axes[0, 0].set_title(f"Fixed {len(scenarios)}-scenario score (equal families)")
    axes[0, 0].legend()
    for p in cost:
        axes[0, 1].scatter(p["mean_ms"], 100 * p["score"], marker="D" if p["pareto"] else "o")
        axes[0, 1].annotate(
            p["candidate"],
            (p["mean_ms"], 100 * p["score"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    axes[0, 1].set_xlabel("Direct mean preprocessing ms (all 10 scenarios)")
    axes[0, 1].set_ylabel("New 256 AP50:95 (%)")
    axes[0, 1].set_title("Exploration accuracy / matched runtime; diamonds = Pareto")
    for index, p in enumerate(uncertainty):
        axes[1, 0].plot(np.array(p["ci95"]) * 100, [index, index], color="tab:blue")
        axes[1, 0].scatter(p["recovery"] * 100, index, color="tab:blue")
    axes[1, 0].set_yticks(
        range(len(uncertainty)),
        [p["candidate"] + " vs " + p["control"] for p in uncertainty],
        fontsize=8,
    )
    axes[1, 0].axvline(0, color="black", linewidth=0.7)
    axes[1, 0].set_xlabel("AP change (percentage points); exploratory 95% interval")
    axes[1, 0].set_title("100 paired source-group draws; new 256 only")
    before = slices[(slices["slice"] == "new") & (slices.candidate == "skip")].set_index("scenario")
    for name in ("pr38_locked", "bilateral", "noise_range"):
        rows = slices[(slices["slice"] == "new") & (slices.candidate == name)].set_index("scenario")
        axes[1, 1].plot(
            range(len(scenarios)),
            [
                100 * (rows.loc[s["id"], "mAP50_95"] - before.loc[s["id"], "mAP50_95"])
                for s in scenarios
            ],
            marker="o",
            label=name,
        )
    axes[1, 1].set_xticks(
        range(len(scenarios)),
        [s["id"].replace("synthetic_", "") for s in scenarios],
        rotation=25,
        fontsize=8,
    )
    axes[1, 1].axhline(0, color="black", linewidth=0.7)
    axes[1, 1].set_ylabel("AP change vs Raw (percentage points)")
    axes[1, 1].set_title("New exploration scenario regressions")
    axes[1, 1].legend(fontsize=8)
    fig.suptitle("Round 3: fixed hypotheses, exploration only, no v3 release")
    fig.tight_layout()
    fig.savefig(REPORT / "results.png", dpi=140)
    plt.close(fig)


def plot_failures(pilot, reports):
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    examples = []
    for name in ("pr38_locked", "noise_range"):
        record = next(
            r
            for r in reports
            if r["candidate"] == name and r["control"] == "skip" and r["scenario"] == "clean"
        )
        examples += [
            (name, "lowest TP change", record["worst_TP"][0]),
            (name, "highest FP change", record["worst_FP"][0]),
        ]
    with exploration_access("data/bdd100k", pilot):
        for ax, (name, reason, error) in zip(axes.flat, examples, strict=True):
            row = next(r for r in pilot["rows"] if r["id"] == error["id"])
            ax.imshow(read_image(Path("data/bdd100k") / row["source_path"])[:, :, ::-1])
            for b in remap_boxes(row["boxes"]):
                ax.add_patch(
                    Rectangle(
                        (b[1], b[2]),
                        b[3] - b[1],
                        b[4] - b[2],
                        fill=False,
                        edgecolor="orange",
                        linewidth=0.6,
                    )
                )
            ax.set_title(
                f"{name}, {reason}, {error['id']}\n"
                f"Raw {error['before']} → treatment {error['after']}",
                fontsize=9,
            )
            ax.axis("off")
    fig.suptitle("New exploration clean error examples; orange = GT, counts at conf .25 / IoU .50")
    fig.tight_layout()
    fig.savefig(REPORT / "failures.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    build_report()
