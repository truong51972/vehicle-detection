"""Run a REAL BDD100K YOLO fine-tune + evaluation, then verify the artifacts.

Requires locally installed BDD detection images/labels and detector dependencies.
Unlike CI unit tests, this intentionally does NOT mock the YOLO model.
"""

import argparse
import csv
import json
import math
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from vehicle_detection.common import digest, file_hash, read_json, write_json

CLASSES = ["car", "truck", "bus", "train", "motorcycle", "bicycle"]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def artifact(path):
    require(path.is_file() and path.stat().st_size > 0, f"Missing/empty artifact: {path}")
    return path


def verify(output):
    prepared = read_json(artifact(output / "prepared/manifest.json"))
    training = read_json(artifact(output / "train/training.json"))
    experiment = read_json(artifact(output / "benchmark/experiment.json"))
    results = read_json(artifact(output / "benchmark/results.json"))
    best = artifact(output / "train/weights/best.pt")
    last = artifact(output / "train/weights/last.pt")
    for name in (
        "train/results.csv",
        "benchmark/results.csv",
        "benchmark/report.md",
        "benchmark/comparison.png",
        "benchmark/decisions.jsonl",
    ):
        artifact(output / name)

    require(prepared["classes"] == CLASSES, "Wrong vehicle category mapping")
    partitions = prepared["partitions"]
    require(all(partitions[split] for split in ("train", "tune", "holdout")), "Empty partition")
    require(
        all(
            r["official_split"] == "train"
            for split in ("train", "tune")
            for r in partitions[split]
        ),
        "Training/tuning contains nontrain examples",
    )
    require(
        all(r["official_split"] == "val" for r in partitions["holdout"]),
        "Holdout must be official BDD validation",
    )

    best_hash = file_hash(best)
    require(best_hash == training["best"]["sha256"], "Best checkpoint hash mismatch")
    require(file_hash(last) == training["last"]["sha256"], "Last checkpoint hash mismatch")
    require(training["manifest_identity"] == prepared["identity"], "Training manifest differs")
    require(experiment["manifest_identity"] == prepared["identity"], "Evaluation manifest differs")
    require(experiment["detector"]["weights_sha256"] == best_hash, "Evaluation used wrong weights")
    require(experiment["split"] == "holdout", "Expected official-validation holdout evaluation")
    require(experiment["training_provenance"] is not None, "Missing real training provenance")
    require(experiment["training_provenance"]["best"]["sha256"] == best_hash, "Bad provenance")
    require(results["experiment"] == digest(experiment), "Results identity mismatch")
    with (output / "train/results.csv").open(newline="") as stream:
        epoch_count = sum(1 for _ in csv.DictReader(stream))
    require(epoch_count > 0, "No training epochs recorded")

    bench = experiment["config"]["benchmark"]
    expected = {(c, t) for c in bench["conditions"] for t in bench["treatments"]}
    actual = {(r["condition"], r["treatment"]) for r in results["overall"]}
    require(actual == expected and len(results["overall"]) == len(expected), "Incomplete matrix")
    sample_ids = set(experiment["sample_ids"])
    require(sample_ids == {r["id"] for r in partitions["holdout"]}, "Holdout IDs differ")
    for row in results["overall"]:
        require(row["images"] == len(sample_ids), "Inconsistent paired sample counts")
        for metric in ("map50", "map50_95", "precision", "recall"):
            value = row[metric]
            require(
                value is None or (math.isfinite(value) and 0 <= value <= 1),
                f"Invalid {metric}: {value}",
            )
        for sample_id in sample_ids:
            record = read_json(
                artifact(
                    output / "benchmark/records" / row["condition"]
                    / row["treatment"] / f"{sample_id}.json"
                )
            )
            require(record["id"] == sample_id, f"Unpaired prediction for {sample_id}")
            require(record["experiment"] == results["experiment"], "Mixed experiment records")

    summary = {
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "mode": "real BDD100K fine-tune and real Ultralytics evaluation, no mock detector",
        "manifest_identity": prepared["identity"],
        "weights_sha256": best_hash,
        "device": experiment["hardware"],
        "training_epochs_recorded": epoch_count,
        "partitions": {name: len(partitions[name]) for name in ("train", "tune", "holdout")},
        "evaluated_combinations": len(expected),
        "holdout_images": len(sample_ids),
        "metrics": [
            {
                k: r[k]
                for k in (
                    "condition", "treatment", "map50", "map50_95",
                    "precision", "recall", "delta_map50_95",
                )
            }
            for r in results["overall"]
        ],
    }
    write_json(output / "real_validation.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/bdd100k"))
    parser.add_argument("--out", type=Path, default=Path("runs/real-pilot"))
    parser.add_argument("--device", default="0", help="CUDA device index, cpu, or mps")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    output = args.out.resolve()

    if not args.verify_only:
        require(args.data.is_dir(), f"Local BDD100K dataset not found: {args.data}")
        require(not output.exists(), f"Output exists: {output}; use a fresh path")
        try:
            import torch
            import ultralytics  # noqa: F401
        except ImportError as exc:
            raise SystemExit("Install dependencies: uv sync --locked --extra detector") from exc
        if args.device.isdigit():
            require(torch.cuda.is_available(), "CUDA unavailable; specify --device cpu")
        cmd = [
            sys.executable, "-m", "vehicle_detection.cli", "run",
            "--preset", "pilot",
            "--data", str(args.data.resolve()), "--out", str(output),
            "--device", args.device, "--model", args.model,
            "--epochs", str(args.epochs), "--batch", str(args.batch),
            "--imgsz", str(args.imgsz),
        ]
        print("Executing unmocked BDD100K training and evaluation:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

    print(json.dumps(verify(output), indent=2))
    print(f"Verified real training and evaluation: {output / 'real_validation.json'}")


if __name__ == "__main__":
    main()
