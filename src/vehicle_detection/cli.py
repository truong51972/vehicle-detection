"""The vd CLI; surveys and tests never import PyTorch or download weights."""

import argparse
import json
from pathlib import Path

import yaml

from .benchmark import benchmark
from .common import read_json
from .config import load_config
from .data import discover, load_manifest, prepare, survey
from .detector import train


def parser():
    cli = argparse.ArgumentParser(description="BDD100K controlled preprocessing experiments")
    commands = cli.add_subparsers(dest="command", required=True)
    for command in ("survey", "prepare", "train", "benchmark", "run"):
        sub = commands.add_parser(
            command,
            help={
                "survey": "Profile detection annotations and sampled image health",
                "prepare": "Create disjoint manifests and raw YOLO export",
                "train": "Train raw baseline; select checkpoint on tuning partition only",
                "benchmark": "Compare frozen weights on paired tuning/holdout images",
                "run": "Survey -> prepare -> train -> benchmark -> report",
            }[command],
        )
        sub.add_argument("--preset", choices=("pilot", "full"), default="pilot")
        sub.add_argument("--config", type=Path, help="YAML overrides merged into preset")
        sub.add_argument("--seed", type=int)
        sub.add_argument(
            "--out",
            type=Path,
            required=True,
            help="Output directory (gitignored runs/ recommended)",
        )
        if command in ("survey", "prepare", "run"):
            sub.add_argument("--data", type=Path, default=Path("data/bdd100k"))
        if command in ("train", "benchmark"):
            sub.add_argument("--prepared", type=Path, required=True)
        if command in ("train", "benchmark", "run"):
            sub.add_argument("--device", help="cpu, mps, or single CUDA index, e.g. 0")
            sub.add_argument("--imgsz", type=int)
        if command in ("train", "run"):
            sub.add_argument("--model", help="Ultralytics checkpoint or model YAML")
            sub.add_argument("--epochs", type=int)
            sub.add_argument("--batch", type=int)
        if command in ("benchmark", "run"):
            sub.add_argument("--resume", action="store_true")
        if command == "benchmark":
            sub.add_argument("--weights", type=Path, required=True)
            sub.add_argument(
                "--split",
                choices=("tune", "holdout"),
                default="holdout",
                help="Use tune for policy calibration; holdout for locked final study",
            )
    return cli


def run(args, config):
    root = args.out.resolve()
    prepared, training, evaluation = root / "prepared", root / "train", root / "benchmark"
    if root.exists() and not args.resume:
        raise ValueError(f"Run directory exists: {root}; use --resume or a new directory")
    if not prepared.exists():
        print("Surveying dataset...", flush=True)
        discovered = discover(args.data)
        survey(args.data, root / "survey", config, discovered)
        print("Preparing raw dataset...", flush=True)
        prepare(args.data, prepared, config, discovered)
    manifest = load_manifest(prepared, verify_images=False)
    if (
        manifest["dataset_root"] != str(args.data.resolve())
        or manifest["data_config"] != config["data"]
        or manifest["seed"] != config["seed"]
    ):
        raise ValueError("Prepared dataset inputs/config changed; use a new run directory")
    record_path = training / "training.json"
    if record_path.exists() and args.resume:
        record = read_json(record_path)
        if record["config"] != config:
            raise ValueError("Run config changed; use a new run directory")
        weights = Path(record["best"]["path"])
    else:
        print("Training raw baseline...", flush=True)
        weights = train(prepared, training, config)
    print("Benchmarking frozen checkpoint...", flush=True)
    benchmark(
        prepared,
        weights,
        evaluation,
        config,
        resume=args.resume and (evaluation / "experiment.json").exists(),
    )
    print(f"Report: {evaluation / 'report.md'}")


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    overrides = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    training = {
        key: getattr(args, key)
        for key in ("model", "epochs", "batch", "imgsz", "device")
        if getattr(args, key, None) is not None
    }
    if training:
        overrides["training"] = training
    try:
        config = load_config(args.preset, args.config, overrides)
        if args.command == "survey":
            result = survey(args.data, args.out, config)
            print(
                json.dumps({"counts": result["counts"], "warnings": result["warnings"]}, indent=2)
            )
        elif args.command == "prepare":
            result = prepare(args.data, args.out, config)
            print(
                json.dumps(
                    {
                        "identity": result["identity"],
                        "images": {k: len(v) for k, v in result["partitions"].items()},
                    },
                    indent=2,
                )
            )
        elif args.command == "train":
            print(f"Frozen checkpoint: {train(args.prepared, args.out, config)}")
        elif args.command == "benchmark":
            benchmark(args.prepared, args.weights, args.out, config, args.resume, args.split)
            print(f"Report: {args.out / 'report.md'}")
        else:
            run(args, config)
    except (ValueError, OSError, KeyError, json.JSONDecodeError, yaml.YAMLError) as exc:
        cli.exit(2, f"vd: {exc}\n")


if __name__ == "__main__":
    main()
