"""Deterministic development-evidence figures; no inference, image reads or global style changes."""

from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from .comparison import (
    POLICIES,
    VERSIONS,
    class_deltas,
    compatible_runs,
    matched_studies,
)
from .data import FIVE_CLASSES

METRICS = ("mAP50", "mAP50_95", "precision", "recall")
STATUS_COLORS = {
    "cached": "#b8dfcc",
    "evaluated": "#85c6df",
    "missing": "#eeeeee",
    "blocked": "#f4d49b",
    "stale": "#e8a1a1",
    "incomplete": "#d0b6dd",
    "unindexed": "#a9bfdc",
}


def _note(fig, report, keys=(), extra=""):
    sources = sorted(
        {
            report["contexts"][v].get("run_id", report["contexts"][v]["run"].name)
            for key in keys
            for v in key.split("/")
            if v in report["contexts"]
        }
    )
    note = "Development validation · model/input labels · " + extra
    if sources:
        note += "\nSource runs: " + ", ".join(sources)
    fig.text(0.02, 0.01, fill(note, 165), fontsize=8, va="bottom")
    fig.tight_layout(rect=(0, 0.10, 1, 0.95))
    return fig


def _unavailable(ax, reason):
    ax.text(0.5, 0.5, fill(reason, 65), ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()


def plot_coverage(report):
    fig, (ax, reasons) = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={"width_ratios": [1, 1.4]})
    statuses = list(STATUS_COLORS)
    grid = np.full((len(VERSIONS), len(VERSIONS)), np.nan)
    for row in report["rows"]:
        y, x = VERSIONS.index(row["model"]), VERSIONS.index(row["input"])
        grid[y, x] = statuses.index(row["status"])
        key = f"{row['model']}/{row['input']}"
        result = report["results"].get(key) or report.get("unindexed", {}).get(key)
        value = f"\n{result['overall']['mAP50_95'] * 100:.3f}%" if result else ""
        ax.text(x, y, row["status"] + value, ha="center", va="center", fontsize=9)
    ax.imshow(
        grid, cmap=ListedColormap(list(STATUS_COLORS.values())), vmin=-0.5, vmax=len(statuses) - 0.5
    )
    ax.set(
        xticks=range(5),
        yticks=range(5),
        xticklabels=VERSIONS,
        yticklabels=VERSIONS,
        xlabel="Input policy",
        ylabel="Trained model",
        title="13 permitted cells · mAP50:95 (%)",
    )
    for y, x in zip(*np.where(np.isnan(grid)), strict=True):
        ax.text(x, y, "—", ha="center", va="center", color="gray")
    pending = [
        f"{r['model']}/{r['input']}: {r['status']} — {r['detail']}"
        for r in report["rows"]
        if r["status"] not in {"cached", "evaluated"}
    ]
    reasons.set_axis_off()
    reasons.text(
        0,
        1,
        "Unavailable cells\n\n" + "\n\n".join(fill(s, 78) for s in pending)
        if pending
        else "All permitted cells available",
        va="top",
        fontsize=9,
    )
    fig.suptitle("Coverage / preflight · archived matched Raw is a separate study")
    return _note(
        fig,
        report,
        report["results"],
        "Blank cells are outside the design; no policy-to-policy evaluation.",
    )


def plot_overall(report):
    studies = matched_studies(report)
    fig, axes = plt.subplots(
        max(1, len(studies)), 4, figsize=(19, max(4, len(studies) * 4.8)), squeeze=False
    )
    if not studies:
        for ax in axes.flat:
            _unavailable(
                ax,
                "No verified same-profile baseline; see coverage and provenance for "
                "absolute saved values.",
            )
    for row, study in enumerate(studies):
        keys = study["keys"]
        baseline = report["results"][study["baseline"]]["overall"]
        labels = [k.replace("matched_raw/matched_raw", "matched Raw/Raw") for k in keys]
        for ax, metric in zip(axes[row], METRICS, strict=True):
            values = [report["results"][key]["overall"][metric] * 100 for key in keys]
            ax.barh(labels, values, color=["#657987"] + ["#4e91a8"] * (len(keys) - 1))
            for y, value in enumerate(values):
                delta = value - baseline[metric] * 100
                ax.text(value + 0.3, y, f"{value:.2f}% ({delta:+.3f} pp)", va="center", fontsize=8)
            ax.set(
                xlim=(0, 100),
                xlabel=f"{metric} (%) · delta vs {study['baseline']} (pp)",
                title=f"{study['profile']} · {metric}",
            )
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Overall accuracy · separate matched profiles, no pooled leaderboard")
    return _note(
        fig,
        report,
        [k for s in studies for k in s["keys"]],
        "Baseline bars are gray. Percentages and percentage-point changes are distinct.",
    )


def plot_unindexed(report):
    """Absolute pending results, never baseline deltas or verified rankings."""
    pending = dict(report.get("unindexed", {}))
    pending.update(
        {
            f"{r['model']}/{r['input']}": report["results"][f"{r['model']}/{r['input']}"]
            for r in report["rows"]
            if r["status"] == "unindexed" and f"{r['model']}/{r['input']}" in report["results"]
        }
    )
    fig, axes = plt.subplots(1, 4, figsize=(18, max(3.5, len(pending) * 0.4)))
    for ax, metric in zip(axes, METRICS, strict=True):
        if not pending:
            _unavailable(ax, "No results pending attestation")
            continue
        ax.barh(list(pending), [r["overall"][metric] * 100 for r in pending.values()])
        ax.set(xlabel=f"{metric} (%)", title=metric, xlim=(0, 100))
    fig.suptitle("Unindexed / pending attestation · descriptive only · no matched treatment effect")
    return _note(
        fig,
        report,
        pending,
        "Identity/hash review required before factorial effects, CIs or matched rankings.",
    )


def plot_saved_cohorts(report, frame):
    keys = list(frame.pair.unique()) if not frame.empty else []
    fig, axes = plt.subplots(
        max(1, (len(keys) + 1) // 2),
        2,
        figsize=(14, max(4, (len(keys) + 1) // 2 * 3.5)),
        squeeze=False,
    )
    for ax in axes.flat:
        ax.set_axis_off()
    if not keys:
        _unavailable(axes[0, 0], "Saved descriptive cohort summaries unavailable")
    for ax, key in zip(axes.flat, keys, strict=False):
        ax.set_axis_on()
        group = frame.loc[frame.pair == key]
        ax.bar(range(len(group)), group.mAP50_95_pct, color="#4e91a8")
        for x, row in enumerate(group.itertuples()):
            value = row.mAP50_95_pct
            support = f"n={row.images:g}" if np.isfinite(row.images) else "n=N/A"
            gt = f"GT={row.gt_instances:g}" if np.isfinite(row.gt_instances) else "GT unrecorded"
            label = f"{value:.2f}%" if np.isfinite(value) else "N/A"
            ax.text(
                x,
                value + 1 if np.isfinite(value) else 1,
                f"{label}\n{support}\n{gt}\nclasses={row.represented_classes}",
                ha="center",
                fontsize=8,
            )
        maximum = group.mAP50_95_pct.max()
        ax.set(
            title=f"{key} · {group.profile.iloc[0]}",
            ylabel="Saved mAP50:95 (%)",
            xticks=range(len(group)),
            xticklabels=group.cohort,
            xlim=(-0.6, len(group) - 0.4),
            ylim=(0, max(40, maximum + 20) if np.isfinite(maximum) else 40),
        )
    fig.suptitle("Saved within-run cohorts · descriptive, non-paired, not causal")
    return _note(
        fig,
        report,
        keys,
        "No paired membership/GT claim. Missing or unsupported cohorts are N/A; "
        "GT-instance counts may be unrecorded.",
    )


def plot_factorial(report):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))
    for ax, policy in zip(axes, POLICIES, strict=True):
        keys = ["v0/v0", f"v0/{policy}", f"{policy}/v0", f"{policy}/{policy}"]
        contexts = report["contexts"]
        complete = all(k in report["results"] for k in keys)
        comparable = (
            "v0" in contexts
            and policy in contexts
            and compatible_runs(contexts["v0"], contexts[policy])
        )
        ax.set_title(policy)
        if not complete or not comparable:
            _unavailable(
                ax,
                "Incomplete or non-comparable A/B/C/D. No factorial contrast. "
                "The archived v4 matched Raw is not historical A.",
            )
            continue
        values = [report["results"][k]["overall"]["mAP50_95"] * 100 for k in keys]
        ax.plot([0, 1], values[:2], "o-", label="Raw-trained: A → B")
        ax.plot([0, 1], values[2:], "s-", label=f"{policy}-trained: C → D")
        ax.set(xticks=[0, 1], xticklabels=["Raw input", f"{policy} input"], ylabel="mAP50:95 (%)")
        for x, y, label in zip([0, 1, 0, 1], values, "ABCD", strict=True):
            ax.annotate(
                f"{label} {y:.3f}", (x, y), xytext=(3, 6), textcoords="offset points", fontsize=8
            )
        ax.margins(x=0.15, y=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("2×2 interactions · A Raw/Raw, B Raw/p, C p/Raw, D p/p")
    return _note(
        fig,
        report,
        report["results"],
        "Only complete, same-protocol arms; lines indicate point estimates, not significance.",
    )


def plot_effects(report, effects):
    fig, ax = plt.subplots(figsize=(13, max(4, len(effects) * 0.32)))
    if effects.empty:
        _unavailable(ax, "No complete comparable 2×2; CI unavailable.")
    else:
        labels = []
        for y, row in enumerate(effects.itertuples()):
            label = f"{row.policy}: {row.contrast}"
            lo, hi = getattr(row, "ci_low_pp", np.nan), getattr(row, "ci_high_pp", np.nan)
            if np.isfinite(lo) and np.isfinite(hi):
                ax.plot([lo, hi], [y, y], color="#4e91a8", linewidth=2)
                label += " · paired 95% CI"
            else:
                label += " · CI unavailable"
            ax.scatter(row.delta_pp, y, color="#b35a49", zorder=3)
            labels.append(label)
        ax.set(yticks=range(len(labels)), yticklabels=labels, xlabel="Δ mAP50:95 (pp)")
        ax.axvline(0, color="gray", linewidth=1)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle(
        "Six factorial contrasts · source-group uncertainty only when enabled and available"
    )
    return _note(
        fig,
        report,
        report["results"],
        "95% percentile intervals are conditional on fixed checkpoints; no "
        "multiple-comparison adjustment.",
    )


def plot_classes(report):
    studies = matched_studies(report)
    fig, axes = plt.subplots(
        max(1, len(studies)), 2, figsize=(15, max(4, len(studies) * 4.4)), squeeze=False
    )
    if not studies:
        for ax in axes.flat:
            _unavailable(ax, "Matched class deltas unavailable.")
    for row, study in enumerate(studies):
        for ax, metric in zip(axes[row], ("mAP50", "mAP50-95"), strict=True):
            frame = class_deltas(report, study, metric)
            values = frame.pivot(index="class", columns="pair", values="delta_pp").reindex(
                FIVE_CLASSES
            )
            # Keep deterministic arm order rather than pandas' lexical order.
            values = values.reindex(columns=study["keys"][1:])
            finite = values.to_numpy()[np.isfinite(values.to_numpy())]
            limit = max(float(np.max(np.abs(finite))) if len(finite) else 0, 0.01)
            color = ax.imshow(values, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
            for y in range(len(values)):
                for x in range(len(values.columns)):
                    value = values.iloc[y, x]
                    ax.text(
                        x,
                        y,
                        f"{value:+.2f}" if np.isfinite(value) else "N/A",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )
            ax.set(
                xticks=range(len(values.columns)),
                xticklabels=values.columns,
                yticks=range(5),
                yticklabels=FIVE_CLASSES,
                title=f"{study['profile']} · {metric} vs {study['baseline']}",
            )
            ax.tick_params(axis="x", rotation=45)
            for label in ax.get_yticklabels()[3:]:
                label.set_fontweight("bold")
            fig.colorbar(color, ax=ax, label="Δ AP (pp)", shrink=0.75)
    fig.suptitle("Per-class changes · zero-centered scales · two-wheelers highlighted")
    return _note(
        fig,
        report,
        [k for s in studies for k in s["keys"]],
        "N/A means absent/zero GT support; never imputed as AP=0.",
    )


def plot_cohorts(report, cohorts):
    fig, axes = plt.subplots(1, 4, figsize=(19, 5))
    for ax, policy in zip(axes, POLICIES, strict=True):
        ax.set_title(policy)
        frame = cohorts.get(policy)
        if frame is None or frame.empty:
            _unavailable(
                ax,
                "Fixed policy cohort IDs / paired statistics unavailable. Need "
                "validated metadata.csv with raw brightness, routing, source groups "
                "and development IDs.",
            )
            continue
        for (model, inputs), group in frame.groupby(["model", "input"], sort=True):
            supported = group.loc[group.images > 0]
            ax.plot(supported.cohort, supported.mAP50_95_pct, "o-", label=f"{model}/{inputs}")
        support = frame.drop_duplicates("cohort")
        ax.set(xlabel="Policy-defined fixed cohorts", ylabel="mAP50:95 (%)")
        ax.text(
            0.01,
            0.01,
            "\n".join(
                f"{r.cohort}: n={r.images}, GT={r.gt_instances}" for r in support.itertuples()
            ),
            transform=ax.transAxes,
            fontsize=7,
        )
        ax.legend(fontsize=7)
    fig.suptitle("Dark / Bright and Apply / Skip · identical image membership across arms")
    return _note(
        fig,
        report,
        report["results"],
        "Missing cohorts have no AP estimate; subgroup evidence requires fixed IDs, "
        "not matching counts alone.",
    )


def plot_convergence(report, histories):
    losses = sorted(
        {
            c
            for history in histories.values()
            for c in history
            if c.startswith(("train/", "val/")) and c.endswith("_loss")
        }
    )
    columns = ("metrics/mAP50-95(B)", "metrics/mAP50(B)", *losses)
    profiles = sorted({ctx["profile"] for ctx in report["contexts"].values()})
    fig, axes = plt.subplots(
        max(1, len(profiles)),
        len(columns),
        figsize=(23, max(4, len(profiles) * 3.7)),
        squeeze=False,
    )
    if not profiles:
        for ax in axes.flat:
            _unavailable(ax, "Training histories unavailable")
    for row, profile in enumerate(profiles):
        versions = [
            v
            for v, ctx in report["contexts"].items()
            if ctx["profile"] == profile and v in histories
        ]
        for ax, column in zip(axes[row], columns, strict=True):
            found = False
            scale = 100 if column.startswith("metrics/") else 1
            for v in versions:
                history = histories[v]
                if column not in history:
                    continue
                found = True
                (line,) = ax.plot(
                    history.epoch, history[column] * scale, label=f"{v} ({len(history)}/50)"
                )
                if column == "metrics/mAP50-95(B)" and history[column].notna().any():
                    i = history[column].idxmax()
                    ax.scatter(
                        history.loc[i, "epoch"],
                        history.loc[i, column] * scale,
                        color=line.get_color(),
                        s=24,
                    )
            ax.set(
                title=f"{profile}\n{column}",
                xlabel="Epoch",
                ylabel="Calibration AP (%)" if scale == 100 else "Loss",
            )
            if found:
                ax.legend(fontsize=7)
                ax.grid(alpha=0.2)
            else:
                _unavailable(ax, "History/metric unavailable")
    fig.suptitle("Training / calibration convergence · batch profiles shown separately")
    return _note(
        fig,
        report,
        [f"{v}/{v}" for v in histories],
        "Dots: peak rounded CSV mAP50:95, inferred selected epoch; exact checkpoint "
        "epoch unavailable. Curves use calibration, not development; losses shown "
        "where recorded.",
    )


def plot_cost(report, histories, cpu_cost=None):
    studies = matched_studies(report)
    fig, axes = plt.subplots(
        max(1, len(studies)), 3, figsize=(18, max(4, len(studies) * 4.8)), squeeze=False
    )
    if not studies:
        for ax in axes.flat:
            _unavailable(ax, "No comparable cost study; runtime / device provenance required.")
    for row, study in enumerate(studies):
        keys = [k for k in study["keys"] if k.split("/")[0] == k.split("/")[1]]
        ax = axes[row, 0]
        for key in keys:
            v = key.split("/")[0]
            history = histories.get(v)
            if history is not None and "time" in history:
                ax.scatter(
                    history.time.iloc[-1] / 60,
                    report["results"][key]["overall"]["mAP50_95"] * 100,
                    label=key,
                )
        ax.set(
            xlabel="Native train history (minutes)",
            ylabel="Development mAP50:95 (%)",
            title=study["profile"],
        )
        if ax.collections:
            ax.legend(fontsize=8)
        else:
            _unavailable(ax, "Training duration unavailable")
        ax = axes[row, 1]
        y = np.arange(len(keys))
        for i, component in enumerate(("preprocess", "inference", "postprocess")):
            values = [
                report["results"][key].get("speed_ms", {}).get(component, np.nan) for key in keys
            ]
            ax.barh(y + (i - 1) * 0.24, values, height=0.24, label=component)
        ax.set(
            yticks=y,
            yticklabels=keys,
            xlabel="Native ms/image (components shown separately)",
            title=study["profile"],
        )
        ax.legend(fontsize=8)
        ax = axes[row, 2]
        policies = {k.split("/")[0] for k in keys}
        cost = (
            cpu_cost.loc[cpu_cost.policy.isin(policies)]
            if cpu_cost is not None and not cpu_cost.empty
            else None
        )
        if cost is None or cost.empty:
            _unavailable(
                ax,
                (
                    "v4 policy cost unavailable. Enable MEASURE_V4_PREPROCESS_COST. "
                    if "v4" in policies
                    else "CPU policy mean/p50/p95 unavailable. "
                )
                + "Optional benchmark needs validated Raw development JPEG proxies. "
                "Offline preparation is not "
                "online latency.",
            )
        else:
            cost.set_index("policy")[["mean_ms", "p50_ms", "p95_ms"]].plot.barh(ax=ax)
            ax.set(
                xlabel="CPU policy ms/image · separate benchmark",
                title="Raw JPEG95 proxy; one OpenCV thread",
            )
        runtime = report["contexts"][study["baseline"].split("/")[0]]["runtime"]
        axes[row, 0].text(0, -0.35, fill(runtime, 60), transform=axes[row, 0].transAxes, fontsize=7)
    fig.suptitle("Accuracy / documented costs · compare within runtime and evaluation profile only")
    return _note(
        fig,
        report,
        [k for s in studies for k in s["keys"]],
        "Native timings exclude external policy; no summed end-to-end latency. Timing "
        "order/cache/thermal effects uncontrolled; no speedup claim.",
    )


def export_report(report, directory, *, figures, tables, settings):
    """Opt-in PNG/CSV bundle; refuse existing destinations or experimental paths."""
    import re
    from datetime import UTC, datetime
    from pathlib import Path

    from .common import file_hash, relative, write_json

    directory = Path(directory).resolve()
    derived = (report["root"] / "reports/comparison").resolve()
    if not directory.is_relative_to(derived):
        raise ValueError("Export only to a dedicated directory under reports/comparison/")
    if any(not re.fullmatch(r"[a-z0-9_]+", name) for name in (*figures, *tables)):
        raise ValueError("Use simple lowercase artifact names")
    if directory.exists():
        raise FileExistsError("Report destination already exists; choose a new directory")
    sources = {}
    for folder in report["paths"].values():
        for name in ("metrics.json", "paired_stats.npz"):
            path = folder / name
            if path.is_file():
                sources[relative(path, report["root"])] = file_hash(path)
    for ctx in report["contexts"].values():
        for name in ("config.yaml", "train/complete.json", "train/args.yaml", "train/results.csv"):
            path = ctx["run"] / name
            if path.is_file():
                sources[relative(path, report["root"])] = file_hash(path)
    for name in ("data.yaml", "splits.csv.gz", "comparison_evidence.json"):
        path = report["root"] / "configs" / name
        if path.is_file():
            sources[relative(path, report["root"])] = file_hash(path)
    directory.mkdir(parents=True)
    readme = [
        "# Generated development comparison\n",
        "See manifest.json for run IDs, profiles, checkpoint/source hashes, status and settings. "
        "Saved cohorts and pending evidence are descriptive, non-paired and not causal. "
        "External CPU timings are separate from native GPU timings. Final holdout sealed.\n",
    ]
    for name, figure in figures.items():
        figure.savefig(directory / f"{name}.png", dpi=120)
        readme.append(f"\n## {name.replace('_', ' ')}\n\n![{name}]({name}.png)\n")
    for name, table in tables.items():
        table.to_csv(directory / f"{name}.csv", index=False)
        readme.append(f"\n[{name}]({name}.csv)\n")
    (directory / "README.md").write_text("\n".join(readme))
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "split": "development_validation",
        "final_test_accessed": False,
        "settings": settings,
        "provenance": report["provenance"],
        "coverage": report["rows"],
        "sources_sha256": sources,
        "cells": {
            key: {
                "inputs": result["inputs"],
                "pending_attestation": key in report.get("unindexed", {})
                or not report["contexts"][key.split("/")[0]].get("verified", False),
            }
            for key, result in {**report["results"], **report.get("unindexed", {})}.items()
        },
        "artifacts_sha256": {p.name: file_hash(p) for p in sorted(directory.iterdir())},
    }
    write_json(directory / "manifest.json", manifest)
    return directory
