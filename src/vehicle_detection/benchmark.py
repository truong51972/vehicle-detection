"""Paired, frozen-checkpoint evaluation with atomic per-image resume records."""

import csv
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np

from .common import digest, environment, read_image, read_json, write_json
from .data import CLASSES, load_manifest
from .detector import YoloDetector
from .metrics import METRIC_ID, evaluate
from .preprocess import process, quality, stress


@contextmanager
def experiment_lock(output):
    path = output / ".lock"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ValueError(
            f"Benchmark locked: {path}. If the previous process died, "
            "confirm it has stopped, then remove this lock and use --resume"
        ) from exc
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        path.unlink()


def ensure_identity(output, identity, resume):
    path = output / "experiment.json"
    if path.exists():
        previous = read_json(path)
        if not resume:
            raise ValueError(f"Experiment exists: {output}; use --resume or a new output")
        if previous != identity:
            raise ValueError(
                "Resume identity mismatch (manifest, weights, config, code, "
                "software or hardware changed); use a new output directory"
            )
    elif resume:
        raise ValueError("Cannot resume without experiment.json; use a new output directory")
    elif any(p.name != ".lock" for p in output.iterdir()):
        raise ValueError(f"Experiment directory is not empty: {output}")
    else:
        write_json(path, identity)


def validate_predictions(predictions, width, height):
    result = np.asarray(predictions, dtype=float)
    if not result.size:
        return []
    if (
        result.ndim != 2
        or result.shape[1] != 6
        or not np.isfinite(result).all()
        or np.any(result[:, 0] != result[:, 0].astype(int))
        or np.any((result[:, 0] < 0) | (result[:, 0] >= len(CLASSES)))
        or np.any((result[:, 5] < 0) | (result[:, 5] > 1))
        or np.any(result[:, 1] < 0)
        or np.any(result[:, 2] < 0)
        or np.any(result[:, 3] > width)
        or np.any(result[:, 4] > height)
        or np.any(result[:, 3] <= result[:, 1])
        or np.any(result[:, 4] <= result[:, 2])
    ):
        raise ValueError("Detector returned invalid class/coordinates/confidence")
    return result.tolist()


def load_record(path, expected, identity):
    payload = read_json(path)
    checksum = payload.pop("checksum")
    if digest(payload) != checksum or payload.get("experiment") != identity:
        raise ValueError(f"Corrupt/incompatible resume record: {path}")
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"Paired alignment mismatch on {key}: {path}")
    validate_predictions(payload["predictions"], payload["width"], payload["height"])
    return payload


def save_sample(output, row, before, after):
    path = output / "samples" / row["condition"] / row["treatment"]
    path.mkdir(parents=True, exist_ok=True)
    panels = []
    for image, title in ((before, "Input / GT green"), (after, "Output / predictions red")):
        panel = image.copy()
        boxes = row["boxes"] if not panels else row["predictions"]
        color = (0, 255, 0) if not panels else (0, 0, 255)
        for box in boxes:
            if len(box) == 6 and box[5] < row["display_confidence"]:
                continue
            cls, x1, y1, x2, y2 = box[:5]
            cv2.rectangle(panel, (round(x1), round(y1)), (round(x2), round(y2)), color, 1)
            cv2.putText(
                panel,
                CLASSES[int(cls)],
                (round(x1), max(12, round(y1))),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )
        cv2.putText(panel, title, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        panels.append(panel)
    if not cv2.imwrite(str(path / f"{row['id']}.jpg"), np.concatenate(panels, axis=1)):
        raise ValueError(f"Could not write sample to {path}")


def benchmark(prepared, weights, output, config, resume=False, split="holdout", detector=None):
    manifest = load_manifest(prepared)
    records = manifest["partitions"][split]
    detector = detector or YoloDetector(weights, config)
    provenance = detector.training_provenance
    if provenance:
        checkpoint_hashes = {provenance[k]["sha256"] for k in ("best", "last")}
        if (
            provenance["manifest_identity"] != manifest["identity"]
            or detector.identity["weights_sha256"] not in checkpoint_hashes
        ):
            raise ValueError("Checkpoint training provenance does not match dataset/weights")
    bench = config["benchmark"]
    # Warm up actual inference device before recording its hardware identity or any timings.
    warm_image = read_image(records[0]["path"])
    for _ in range(bench["warmup"]):
        detector.predict(warm_image)
    detector.synchronize()
    identity = {
        "schema": 1,
        "manifest_identity": manifest["identity"],
        "config": config,
        "detector": detector.identity,
        "metric": METRIC_ID,
        "split": split,
        "sample_ids": [r["id"] for r in records],
        "environment": environment(),
        "hardware": detector.hardware(),
        "training_provenance": provenance,
    }
    experiment_id = digest(identity)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with experiment_lock(output):
        ensure_identity(output, identity, resume)
        for condition in bench["conditions"]:
            for index, record in enumerate(records):
                source = read_image(record["path"])
                input_image = stress(
                    source, condition, config["stress"], config["seed"], record["id"]
                )
                diagnostic_quality = quality(input_image)
                treatments = list(bench["treatments"])
                # Deterministic interleaving reduces systematic treatment-order/thermal bias.
                random.Random(digest([config["seed"], condition, record["id"]])).shuffle(treatments)
                for treatment in treatments:
                    path = output / "records" / condition / treatment / f"{record['id']}.json"
                    expected = {
                        "id": record["id"],
                        "condition": condition,
                        "treatment": treatment,
                        "boxes": record["boxes"],
                        "image_sha256": record["sha256"],
                        "width": record["width"],
                        "height": record["height"],
                    }
                    if path.exists():
                        row = load_record(path, expected, experiment_id)
                        if index < bench["samples"]:
                            processed, _, _ = process(input_image, treatment, config)
                            save_sample(output, row, input_image, processed)
                    else:
                        if index == 0:
                            for _ in range(bench["warmup"]):
                                warmed, _, _ = process(input_image, treatment, config)
                                detector.predict(warmed)
                            detector.synchronize()
                        detector.synchronize()
                        start = time.perf_counter_ns()
                        processed, scores, chosen = process(input_image, treatment, config)
                        pre_end = time.perf_counter_ns()
                        predictions = detector.predict(processed)
                        detector.synchronize()
                        end = time.perf_counter_ns()
                        predictions = validate_predictions(
                            predictions, record["width"], record["height"]
                        )
                        row = {
                            **expected,
                            "experiment": experiment_id,
                            "attributes": record["attributes"],
                            "quality": scores if scores is not None else diagnostic_quality,
                            "sequence": chosen,
                            "predictions": predictions,
                            "preprocess_ms": (pre_end - start) / 1e6,
                            "inference_ms": (end - pre_end) / 1e6,
                            "end_to_end_ms": (end - start) / 1e6,
                            "display_confidence": bench["precision_confidence"],
                        }
                        write_json(path, {**row, "checksum": digest(row)})
                        if index < bench["samples"]:
                            save_sample(output, row, input_image, processed)
                if (index + 1) % 100 == 0:
                    print(f"{condition}: {index + 1}/{len(records)} images", flush=True)
        return report(output, identity, records)


def latency(rows):
    result = {}
    for key in ("preprocess_ms", "inference_ms", "end_to_end_ms"):
        values = [r[key] for r in rows]
        result[key] = {
            "mean": float(np.mean(values)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
        }
    result["throughput_fps"] = 1000 / result["end_to_end_ms"]["mean"]
    return result


def aligned(groups, ids):
    expected = set(ids)
    for key, rows in groups.items():
        actual = [r["id"] for r in rows]
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError(f"Unpaired image IDs in {key}")
    boxes = {}
    for rows in groups.values():
        for row in rows:
            if row["id"] in boxes and boxes[row["id"]] != row["boxes"]:
                raise ValueError(f"Unpaired ground truth: {row['id']}")
            boxes[row["id"]] = row["boxes"]


def report(output, identity, manifest_records):
    config = identity["config"]
    bench = config["benchmark"]
    sources = {r["id"]: r for r in manifest_records}
    metrics, strata, failure_cases = [], [], []

    def load_rows(condition, treatment):
        return [
            read_json(output / "records" / condition / treatment / f"{sample_id}.json")
            for sample_id in identity["sample_ids"]
        ]

    # Hold only raw + one treatment in memory, even for the full condition matrix.
    with (output / "decisions.jsonl").open("w") as decisions:
        import json

        for condition in bench["conditions"]:
            baseline = load_rows(condition, "raw")
            baseline_by_id = {r["id"]: r for r in baseline}
            baseline_image_ap = {
                r["id"]: evaluate([r], bench["precision_confidence"])["map50_95"] for r in baseline
            }
            for treatment in bench["treatments"]:
                selected = baseline if treatment == "raw" else load_rows(condition, treatment)
                aligned({"raw": baseline, treatment: selected}, identity["sample_ids"])
                metrics.append(
                    {
                        "condition": condition,
                        "treatment": treatment,
                        **evaluate(selected, bench["precision_confidence"]),
                        "latency": latency(selected),
                    }
                )
                for row in selected:
                    decisions.write(
                        json.dumps(
                            {k: v for k, v in row.items() if k not in {"predictions", "checksum"}}
                        )
                        + "\n"
                    )
                for attribute in ("weather", "timeofday", "scene"):
                    for category in sorted({r["attributes"][attribute] for r in selected}):
                        subset = [r for r in selected if r["attributes"][attribute] == category]
                        paired_raw = [baseline_by_id[r["id"]] for r in subset]
                        result = evaluate(subset, bench["precision_confidence"])
                        base = evaluate(paired_raw, bench["precision_confidence"])
                        strata.append(
                            {
                                "condition": condition,
                                "treatment": treatment,
                                "attribute": attribute,
                                "value": category,
                                "small_sample": len(subset) < bench["min_stratum"],
                                **result,
                                "delta_map50_95": (
                                    result["map50_95"] - base["map50_95"]
                                    if result["map50_95"] is not None
                                    else None
                                ),
                            }
                        )
                if treatment == "raw":
                    continue
                differences = []
                for row in selected:
                    current = evaluate([row], bench["precision_confidence"])["map50_95"]
                    base = baseline_image_ap[row["id"]]
                    if current is not None:
                        differences.append((current - base, row))
                for delta, row in sorted(differences, key=lambda item: (item[0], item[1]["id"]))[
                    : bench["samples"]
                ]:
                    failure_cases.append(
                        {
                            "condition": condition,
                            "treatment": treatment,
                            "id": row["id"],
                            "single_image_delta": delta,
                            "interpretation": (
                                "harm"
                                if delta < 0
                                else "no improvement"
                                if delta == 0
                                else "improvement"
                            ),
                        }
                    )
                    source = read_image(sources[row["id"]]["path"])
                    stressed = stress(
                        source, condition, config["stress"], config["seed"], row["id"]
                    )
                    processed, _, _ = process(stressed, treatment, config)
                    save_sample(output, row, stressed, processed)
    raw = {r["condition"]: r for r in metrics if r["treatment"] == "raw"}
    for row in metrics:
        baseline = raw[row["condition"]]
        row["delta_map50_95"] = (
            row["map50_95"] - baseline["map50_95"] if row["map50_95"] is not None else None
        )
        row["overhead_ms"] = (
            row["latency"]["end_to_end_ms"]["mean"] - baseline["latency"]["end_to_end_ms"]["mean"]
        )
    write_json(
        output / "results.json",
        {
            "experiment": digest(identity),
            "overall": metrics,
            "strata": strata,
            "failure_cases": failure_cases,
        },
    )
    fields = [
        "condition",
        "treatment",
        "images",
        "targets",
        "map50",
        "map50_95",
        "precision",
        "recall",
        "delta_map50_95",
        "overhead_ms",
        "throughput_fps",
    ]
    timing_fields = [
        f"{key}_{stat}"
        for key in ("preprocess_ms", "inference_ms", "end_to_end_ms")
        for stat in ("mean", "p50", "p95")
    ]
    with (output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields + timing_fields)
        writer.writeheader()
        for row in metrics:
            writer.writerow(
                {
                    **{key: row[key] for key in fields if key in row},
                    "throughput_fps": row["latency"]["throughput_fps"],
                    **{
                        f"{key}_{stat}": row["latency"][key][stat]
                        for key in ("preprocess_ms", "inference_ms", "end_to_end_ms")
                        for stat in ("mean", "p50", "p95")
                    },
                }
            )
    render_report(output, metrics, strata, failure_cases, identity)
    return metrics


def render_report(output, metrics, strata, failures, identity):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    training_note = (
        "Recorded training selected the checkpoint on train-derived tuning data."
        if identity["training_provenance"]
        else "No training provenance supplied: training/model selection cannot be verified."
    )
    conditions = identity["config"]["benchmark"]["conditions"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for condition in conditions:
        selected = [r for r in metrics if r["condition"] == condition]
        axes[0].plot(
            [r["treatment"] for r in selected],
            [r["map50_95"] if r["map50_95"] is not None else np.nan for r in selected],
            "o-",
            label=condition,
        )
        axes[1].scatter(
            [r["latency"]["end_to_end_ms"]["mean"] for r in selected],
            [r["map50_95"] if r["map50_95"] is not None else np.nan for r in selected],
            label=condition,
        )
        for r in selected:
            if r["map50_95"] is not None:
                axes[1].annotate(
                    r["treatment"], (r["latency"]["end_to_end_ms"]["mean"], r["map50_95"])
                )
    axes[0].set(
        ylabel="mAP50:95", title="Paired accuracy (synthetic conditions labeled separately)"
    )
    axes[0].tick_params(axis="x", rotation=35)
    axes[1].set(xlabel="Mean end-to-end ms/image", ylabel="mAP50:95", title="Accuracy versus cost")
    for axis in axes:
        axis.legend()
    fig.tight_layout()
    fig.savefig(output / "comparison.png")
    plt.close(fig)
    lines = [
        "# BDD100K preprocessing experiment",
        "",
        f"Split: **{identity['split']}**, images: {len(identity['sample_ids'])}. "
        "Pilot/small-sample scores establish feasibility only, not research-grade conclusions.",
        "",
        f"Experiment: `{digest(identity)}`. Manifest: `{identity['manifest_identity']}`.",
        f"Frozen weights: `{identity['detector']['weights_sha256']}`.",
        "",
        "All treatments use identical IDs, boxes, confidence/NMS/resolution settings and weights. "
        f"{training_note} Holdout must never select policy thresholds or models. "
        "Defaults are uncalibrated heuristics.",
        "",
        "AP uses confidence-ordered one-to-one classwise IoU matching at 0.50:0.05:0.95, "
        "a precision envelope and 101 equally spaced recall points; no area/crowd weighting. "
        "Classes without GT have null AP and are excluded from macro averages. "
        "Precision/recall use IoU 0.50 and the configured precision confidence. "
        "This evaluator is not the official COCO/BDD leaderboard evaluator.",
        "",
        "Clean means original pixels. Dark/blur/noise/haze are deterministic synthetic stresses, "
        "separate from BDD weather/time metadata. No geometric transforms are applied.",
        "",
        "Batch size 1, FP32, fixed square inference resolution (see experiment.json). "
        "Warm-up excluded; CUDA/MPS synchronized. Preprocessing includes adaptive feature "
        "extraction/routing/operators. Inference includes detector resize/normalization, forward, "
        "NMS and CPU result transfer. End-to-end measures preprocessing + inference; excludes "
        "disk read/decode, synthetic stress, diagnostic-only quality extraction, metrics and "
        "artifact writes. Throughput is 1000/mean end-to-end ms, with no overlapped work. "
        "Resumed timings may span sessions; use fresh runs for final latency claims.",
        "",
        "![Accuracy and latency](comparison.png)",
        "",
        "| Condition | Treatment | N | mAP50 | mAP50:95 | P | R | Δ vs raw | Pre p50/p95 ms | "
        "Infer p50/p95 ms | E2E mean/p50/p95 ms | FPS |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics:

        def fmt(key, row=row):
            return "n/a" if row[key] is None else f"{row[key]:.4f}"

        timing = row["latency"]
        lines.append(
            f"| {row['condition']} | {row['treatment']} | {row['images']} | "
            f"{fmt('map50')} | {fmt('map50_95')} | {fmt('precision')} | {fmt('recall')} | "
            f"{fmt('delta_map50_95')} | "
            f"{timing['preprocess_ms']['p50']:.2f}/{timing['preprocess_ms']['p95']:.2f} | "
            f"{timing['inference_ms']['p50']:.2f}/{timing['inference_ms']['p95']:.2f} | "
            f"{timing['end_to_end_ms']['mean']:.2f}/{timing['end_to_end_ms']['p50']:.2f}/"
            f"{timing['end_to_end_ms']['p95']:.2f} | {timing['throughput_fps']:.2f} |"
        )
    lines.extend(
        [
            "",
            "Per-class AP/support and metadata strata are in results.json. "
            "Strata below the configured minimum sample count are flagged; do not draw "
            "strong conclusions from them. Compare each treatment to raw within the same "
            "condition. A positive accuracy delta does not establish acceptable overhead.",
            "",
        ]
    )
    lines.extend(
        [
            "| Condition | Treatment | Class | GT boxes | AP50 | AP50:95 |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in metrics:
        for name, values in row["per_class"].items():
            ap50 = "n/a" if values["map50"] is None else f"{values['map50']:.4f}"
            ap = "n/a" if values["map50_95"] is None else f"{values['map50_95']:.4f}"
            lines.append(
                f"| {row['condition']} | {row['treatment']} | {name} | "
                f"{values['targets']} | {ap50} | {ap} |"
            )
    lines.extend(
        [
            "",
            "Metadata strata with enough images for descriptive comparison "
            "(support still limits inference; complete strata are in results.json):",
            "",
            "| Condition | Treatment | Attribute | Value | N | mAP50:95 | Δ vs raw |",
            "|---|---|---|---|---:|---:|---:|",
        ]
    )
    for row in strata:
        if not row["small_sample"] and row["map50_95"] is not None:
            value = row["value"].replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {row['condition']} | {row['treatment']} | {row['attribute']} | "
                f"{value} | {row['images']} | {row['map50_95']:.4f} | "
                f"{row['delta_map50_95']:+.4f} |"
            )
    small = sum(row["small_sample"] for row in strata)
    lines.extend(
        [
            "",
            f"{small} strata rows fall below the sample-count threshold and are "
            "flagged in results.json.",
            "",
            "Failure cases (lowest per-image deltas, diagnostic only; aggregate AP is "
            "not a mean of image AP):",
            "",
        ]
    )
    lines.extend(
        f"- {r['condition']}/{r['treatment']}/{r['id']}: "
        f"{r['single_image_delta']:+.4f} ({r['interpretation']})"
        for r in failures
    )
    lines.extend(
        [
            "",
            "See samples/ for before/after panels; GT is green, predictions red. "
            "Sharpening cannot recover information lost to real blur; explicitly retain "
            "harm/no-improvement outcomes. Optional bootstrap CIs and repeated study "
            "seeds are not computed by this release.",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(lines))
