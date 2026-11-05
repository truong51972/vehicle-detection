"""Bounded Round 2 calibration treatments; reuse native exports, matching and AP."""

import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from .adaptive import validate_policy
from .common import digest, file_hash, freeze_json, read_json
from .datasets import verify_label
from .evaluation_data import evaluation_sets, scenario_image
from .five_class import FIVE_CLASSES, SIX_TO_FIVE
from .offline import verify_alignment
from .preprocess import apply_recipe, validate_image, validate_recipes
from .protocol import summarize
from .stress import diagnostics, restore, validate_pilot, verify_stress


def candidates():
    """Two strengths per core operator, two existing alternatives, six ordered pairs."""
    singles = {
        "gamma_085": {"gamma": {"gamma": 0.85}},
        "gamma_070": {"gamma": {"gamma": 0.7}},
        "clahe_12": {"clahe": {"clip_limit": 1.2, "grid_size": 8}},
        "clahe_20": {"clahe": {"clip_limit": 2.0, "grid_size": 8}},
        "denoise_2": {"denoise": {"strength": 2.0}},
        "denoise_5": {"denoise": {"strength": 5.0}},
        "sharpen_015": {"sharpen": {"amount": 0.15, "sigma": 1.0}},
        "sharpen_040": {"sharpen": {"amount": 0.4, "sigma": 1.0}},
        "bilateral": {"bilateral": {}},
        "percentile": {"percentile": {}},
    }
    result = {"skip": {"steps": []}, "frozen_v2": {"v2": True}}
    result.update({name: {"steps": [step]} for name, step in singles.items()})
    for a, b in (
        ("gamma_070", "clahe_12"),
        ("clahe_12", "gamma_070"),
        ("denoise_2", "gamma_070"),
        ("gamma_070", "denoise_2"),
        ("bilateral", "sharpen_015"),
        ("sharpen_015", "bilateral"),
    ):
        result[f"{a}_then_{b}"] = {"steps": [singles[a], singles[b]]}
    # Thresholds are hypotheses fixed before detection screening, never scenario labels.
    result["route_dark"] = {
        "rules": [{"metric": "brightness", "below": 0.25, "steps": [singles["gamma_070"]]}]
    }
    result["route_noise_dark"] = {
        "rules": [
            {"metric": "noise_highpass", "above": 0.025, "steps": [singles["bilateral"]]},
            {"metric": "brightness", "below": 0.25, "steps": [singles["gamma_070"]]},
        ]
    }
    return result


def validate_candidate(candidate):
    if candidate == {"v2": True}:
        return
    if set(candidate) == {"steps"}:
        sequences = [candidate["steps"]]
    elif set(candidate) == {"rules"} and 1 <= len(candidate["rules"]) <= 2:
        sequences = []
        for rule in candidate["rules"]:
            bounds = set(rule) & {"above", "below"}
            if (
                len(bounds) != 1
                or set(rule) != {"metric", "steps", *bounds}
                or rule["metric"] not in {"brightness", "contrast", "noise_highpass", "sharpness"}
            ):
                raise ValueError("Invalid quality routing rule")
            value = rule[next(iter(bounds))]
            if type(value) not in (int, float) or not np.isfinite(value) or value < 0:
                raise ValueError("Invalid routing threshold")
            sequences.append(rule["steps"])
    else:
        raise ValueError("Unknown candidate configuration")
    for steps in sequences:
        if not isinstance(steps, list) or len(steps) > 2:
            raise ValueError("A candidate permits at most two ordered filters")
        for step in steps:
            if not isinstance(step, dict) or len(step) != 1:
                raise ValueError("Expected one operator per step")
            name, params = next(iter(step.items()))
            if name in {"bilateral", "percentile"}:
                if params != {}:
                    raise ValueError("Alternatives use the recorded stress operator settings")
            elif name not in {"gamma", "clahe", "denoise", "sharpen"}:
                raise ValueError("Unknown preprocessing operator")
            else:
                validate_recipes(step)


def apply_candidate(image, candidate, v2):
    """Route once from input quality, then apply ordered filters without JPEG or mutation."""
    validate_image(image)
    validate_candidate(candidate)
    if candidate == {"v2": True}:
        output, decisions, _ = apply_recipe(image, "adaptive", v2["recipes"])
        return output, decisions
    steps = candidate.get("steps", [])
    if "rules" in candidate:
        scores = diagnostics(image)
        for rule in candidate["rules"]:
            value = scores[rule["metric"]]
            if ("below" in rule and value < rule["below"]) or (
                "above" in rule and value > rule["above"]
            ):
                steps = rule["steps"]
                break
    output = image.copy()
    for step in steps:
        name = next(iter(step))
        output = (
            restore(output, name, v2["recipes"])
            if name in {"bilateral", "percentile"}
            else apply_recipe(output, name, step)[0]
        )
    return output, steps


def validate_manifest(manifest, protocol):
    expected = evaluation_sets(protocol, seed=manifest["seed"])
    if manifest["manifest_id"] != digest(
        {k: v for k, v in manifest.items() if k != "manifest_id"}
    ) or any(manifest.get(k) != v for k, v in expected.items() if k != "manifest_id"):
        raise ValueError("Frozen evaluation manifest changed; rerun its owning phase")


def calibration_sample(protocol, manifest, role, *, target_images=128):
    """Use whole groups within a saved half; image count is a bounded target, not exact."""
    if role not in {"calibration_exploration", "calibration_confirmation"}:
        raise ValueError("Only exploration/confirmation calibration roles may be screened")
    return _sample(protocol, manifest, role, target_images)


def development_sample(protocol, manifest, *, target_images=256):
    """Predeclare one development subset for transfer diagnostics, never candidate search."""
    return _sample(protocol, manifest, "development_validation", target_images)


def _sample(protocol, manifest, role, target_images):
    validate_manifest(manifest, protocol)
    if type(target_images) is not int or target_images < 1:
        raise ValueError("Sample target must be a positive integer")
    allowed = set(manifest["sets"][role])
    groups = defaultdict(list)
    source_role = "development_validation" if role == "development_validation" else "calibration"
    for row in protocol["splits"][source_role]:
        if row["id"] in allowed:
            groups[row["group"]].append(row)
    selected = []
    for group in sorted(groups, key=lambda g: (digest([manifest["seed"], role, g]), g)):
        selected.extend(groups[group])
        if len(selected) >= target_images:
            break
    pilot = {
        "version": "issue30-development-sample-v1"
        if role == "development_validation"
        else "issue30-calibration-sample-v1",
        "protocol_id": protocol["manifest_id"],
        "evaluation_manifest_id": manifest["manifest_id"],
        "sample_role": role,
        "role": source_role,
        "seed": manifest["seed"],
        "target_images": target_images,
        "rows": sorted(selected, key=lambda r: r["id"]),
        "summary": summarize(selected),
    }
    pilot["pilot_id"] = digest(pilot)
    return pilot if role == "development_validation" else validate_pilot(pilot, protocol)


def validate_sample(protocol, manifest, pilot):
    if pilot["role"] == "development_validation":
        expected = development_sample(protocol, manifest, target_images=pilot["target_images"])
    else:
        expected = calibration_sample(
            protocol, manifest, pilot["sample_role"], target_images=pilot["target_images"]
        )
        validate_pilot(pilot, protocol)
    if pilot != expected:
        raise ValueError("Experiment sample differs from the frozen role/group selection")


def verify_candidate_export(pilot, destination):
    """Use the historical calibration verifier, or the same inventory/box checks for transfer."""
    if pilot["role"] == "calibration":
        return verify_stress(pilot, destination)
    destination = Path(destination).resolve()
    meta = pd.read_csv(destination / "metadata.csv")
    export = read_json(destination / "export.json")
    expected = {r["id"]: r for r in pilot["rows"]}
    if (
        export["pilot_id"] != pilot["pilot_id"]
        or export["role"] != "development_validation"
        or len(meta) != len(expected)
        or set(meta.id) != set(expected)
        or set(meta.role) != {"development_validation"}
        or set(meta.split) != {"tune"}
    ):
        raise ValueError("Transfer export must retain exact frozen development IDs/role")
    labels = {}
    for row in meta.to_dict("records"):
        source = expected[row["id"]]
        if (
            Path(row["image"]) != destination / "images/tune" / f"{row['id']}.jpg"
            or row["group"] != source["group"]
            or row["source_sha256"] != source["sha256"]
            or (row["width"], row["height"]) != (source["width"], source["height"])
        ):
            raise ValueError("Transfer source/group/geometry changed")
        labels[row["id"]] = row["label_sha256"]
    if export["sources"] != digest(
        [(r["id"], r["group"], r["sha256"], labels[r["id"]]) for r in pilot["rows"]]
    ):
        raise ValueError("Transfer annotation identity changed")
    verify_alignment([destination])
    return meta


def export_candidate(
    protocol, manifest, pilot, data_root, converted, destination, scenario, candidate, v2
):
    """Publish paired five-class JPEG95 inputs for calibration or the fixed transfer sample."""
    validate_manifest(manifest, protocol)
    validate_sample(protocol, manifest, pilot)
    validate_policy(v2, protocol)
    validate_candidate(candidate)
    if scenario not in manifest["scenarios"]:
        raise ValueError("Candidate export differs from the frozen sample/scenario")
    data_root, converted = Path(data_root).resolve(), Path(converted).resolve()
    destination = Path(destination).absolute()
    if destination.resolve() != destination:
        raise ValueError("Unexpected output directory symlink")
    for protected in (data_root, converted):
        if (
            destination == protected
            or destination in protected.parents
            or protected in destination.parents
        ):
            raise ValueError("Candidate output overlaps preserved inputs")
    conversion = read_json(converted / "conversion.json")
    if conversion["classes"] != FIVE_CLASSES or conversion["mapping"] != {
        str(k): v for k, v in SIX_TO_FIVE.items()
    }:
        raise ValueError("Expected existing five-class label mapping")
    labels = {}
    for row in pilot["rows"]:
        path = converted / "labels/tune" / f"{row['id']}.txt"
        if path.resolve() != path:
            raise ValueError("Unexpected input label symlink")
        verify_label(path, row, SIX_TO_FIVE)
        labels[row["id"]] = file_hash(path)
        if file_hash(data_root / row["source_path"]) != row["sha256"]:
            raise ValueError("Calibration source changed")
    identity = {
        "version": "issue30-export-v1",
        "pilot_id": pilot["pilot_id"],
        "protocol_id": protocol["manifest_id"],
        "manifest_id": manifest["manifest_id"],
        "role": pilot["role"],
        "scenario": scenario,
        "candidate": candidate,
        "v2_policy_id": v2["policy_id"],
        "conversion": conversion,
        "encoding": manifest["encoding"],
        "code_sha256": file_hash(__file__),
        "sources": digest(
            [(r["id"], r["group"], r["sha256"], labels[r["id"]]) for r in pilot["rows"]]
        ),
    }
    if (
        destination.exists()
        and any(destination.iterdir())
        and not (destination / "export.json").is_file()
    ):
        raise ValueError("Unknown export directory; choose a new root")
    freeze_json(destination / "export.json", identity)
    if (destination / "metadata.csv").exists():
        return verify_candidate_export(pilot, destination)
    for split in ("train", "tune", "test"):
        for kind in ("images", "labels"):
            folder = destination / kind / split
            if folder.resolve() != folder:
                raise ValueError("Unexpected export directory symlink")
            folder.mkdir(parents=True, exist_ok=True)
    metadata = []
    for row in pilot["rows"]:
        pixels = scenario_image(row, data_root, scenario, seed=manifest["seed"])
        tick = time.perf_counter()
        output, decisions = apply_candidate(pixels, candidate, v2)
        elapsed = (time.perf_counter() - tick) * 1000
        if output.shape != pixels.shape or output.dtype != np.uint8:
            raise ValueError("Treatment changed geometry/dtype")
        image = destination / "images/tune" / f"{row['id']}.jpg"
        label = destination / "labels/tune" / f"{row['id']}.txt"
        if image.is_symlink() or label.is_symlink():
            raise ValueError("Unexpected output symlink")
        if not cv2.imwrite(str(image), output, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError("Candidate JPEG95 write failed")
        label.write_bytes((converted / "labels/tune" / label.name).read_bytes())
        metadata.append(
            {
                "id": row["id"],
                "split": "tune",
                "role": pilot["role"],
                "group": row["group"],
                "image": str(image),
                "source_sha256": row["sha256"],
                "width": row["width"],
                "height": row["height"],
                "output_sha256": file_hash(image),
                "label_sha256": labels[row["id"]],
                "restore_ms": elapsed,
                "decisions": json.dumps(decisions),
            }
        )
    (destination / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(destination),
                "train": "images/train",
                "val": "images/tune",
                "test": "images/test",
                "names": dict(enumerate(FIVE_CLASSES)),
            }
        )
    )
    pd.DataFrame(metadata).to_csv(destination / "metadata.csv", index=False)
    return verify_candidate_export(pilot, destination)


def detection_report(pilot, stats, raw_stats):
    """Descriptive native AP slices and error examples, with explicit class support."""
    from .evaluation_stats import class_ap
    from .five_class import remap_boxes

    expected = {r["id"]: sorted(int(b[0]) for b in remap_boxes(r["boxes"])) for r in pilot["rows"]}
    for values in (stats, raw_stats):
        if set(values) != set(expected) or any(
            sorted(map(int, values[i]["target_cls"])) != target for i, target in expected.items()
        ):
            raise ValueError("Native matching statistics differ from paired five-class GT")
    slices = [("all", pilot["rows"])]
    for axis in ("weather", "timeofday", "scene"):
        for value in sorted({r["attributes"][axis] for r in pilot["rows"]}):
            slices.append(
                (f"{axis}:{value}", [r for r in pilot["rows"] if r["attributes"][axis] == value])
            )
    report = []
    for name, rows in slices:
        ids = [r["id"] for r in rows]
        a, b = (class_ap(values, ids, detailed=True) for values in (raw_stats, stats))
        for cls, label in enumerate(FIVE_CLASSES):
            represented = [r for r in rows if cls in expected[r["id"]]]
            groups = len({r["group"] for r in represented})
            report.append(
                {
                    "slice": name,
                    "class": label,
                    "images": len(represented),
                    "groups": groups,
                    "status": "descriptive" if groups >= 5 else "insufficient_support",
                    "AP50_95": b.get(cls, {}).get("AP50_95"),
                    "delta_AP50_95": b[cls]["AP50_95"] - a[cls]["AP50_95"] if cls in a else None,
                    "AP50": b.get(cls, {}).get("AP50"),
                }
            )
    failures = []
    for sample_id in expected:
        # Count native IoU50 true positives at a fixed illustrative confidence; no AP replacement.
        raw, treated = (
            int(np.sum(v[sample_id]["tp"][:, 0] & (v[sample_id]["conf"] >= 0.25)))
            for v in (raw_stats, stats)
        )
        failures.append(
            {"id": sample_id, "raw_TP50": raw, "treated_TP50": treated, "delta": treated - raw}
        )
    return {
        "subgroups": report,
        "failure_examples": sorted(failures, key=lambda r: (r["delta"], r["id"]))[:8],
    }


def rank_candidates(table, scenarios):
    """Equal weight per clean/noise/low-light/blur family; unsupported five-class AP cannot win."""
    expected = {s["id"] for s in scenarios}
    families = {s["id"]: s["family"] for s in scenarios}
    if len(expected) != len(scenarios) or not np.isfinite(table.mAP50_95).all():
        raise ValueError("Ranking requires unique scenarios and finite native metrics")
    ranked = []
    for name, rows in table.groupby("candidate", sort=True):
        if (
            len(rows) != len(expected)
            or set(rows.scenario) != expected
            or not rows.full_class_support.all()
            or (rows.minimum_class_groups < 5).any()
            or any(families.get(r.scenario) != r.family for r in rows.itertuples())
        ):
            raise ValueError("Ranking requires every scenario and five groups per target class")
        score = float(rows.groupby("family").mAP50_95.mean().mean())
        ranked.append({"candidate": name, "score": score, "mean_ms": float(rows.mean_ms.mean())})
    if not {"skip", "frozen_v2"} <= {r["candidate"] for r in ranked}:
        raise ValueError("Ranking requires paired Skip and frozen v2 controls")
    return sorted(
        ranked, key=lambda r: (-r["score"], r["candidate"] != "skip", r["mean_ms"], r["candidate"])
    )


def confirmation_decision(table, scenarios, shortlist, configs, evidence):
    """One prelocked candidate versus two controls; failed confirmation never tries runners-up."""
    candidate = shortlist["candidate"]
    if set(table.candidate) != {"skip", "frozen_v2", candidate}:
        raise ValueError("Confirmation must evaluate only the prelocked candidate and controls")
    ranked = rank_candidates(table, scenarios)
    scores = {r["candidate"]: r["score"] for r in ranked}
    rows = table.set_index(["candidate", "scenario"])
    reasons = []
    for control in ("skip", "frozen_v2"):
        if scores[candidate] < scores[control]:
            reasons.append(f"Family-balanced AP below {control}")
        for scenario in scenarios:
            change = (
                rows.loc[(candidate, scenario["id"]), "mAP50_95"]
                - rows.loc[(control, scenario["id"]), "mAP50_95"]
            )
            limit = 0.005 if scenario["id"] == "clean" else 0.01
            if change < -limit:
                reasons.append(f"Regression beyond {limit} against {control}/{scenario['id']}")
            for a, b in zip(
                rows.loc[(candidate, scenario["id"]), "classes"],
                rows.loc[(control, scenario["id"]), "classes"],
                strict=True,
            ):
                if a["class"] != b["class"]:
                    raise ValueError("Confirmation class order changed")
                if a["AP50_95"] - b["AP50_95"] < -0.02:
                    reasons.append(f"Class regression beyond 0.02: {a['class']}/{scenario['id']}")
    decision = {
        "version": "issue30-policy-v3",
        "status": "confirmation_failed" if reasons else "selected_calibrated_candidate",
        "candidate": candidate,
        "config": configs[candidate],
        "evidence": evidence,
        "shortlist_id": digest(shortlist),
        "confirmation_ranking": ranked,
        "reasons": reasons,
        "limits": [
            "Pretrained calibration evidence does not prove best post-fine-tuning performance.",
            "Point-estimate gates are engineering tolerances, not statistical significance.",
            "Historical v2 used calibration; Round 2 halves alone are disjoint.",
            "Development transfer and final holdout have not been evaluated here.",
        ],
    }
    decision["policy_id"] = digest(decision)
    return decision


def replay_policy(image, policy):
    """Replay a confirmed config only with its frozen code and numeric library versions."""
    if (
        policy["version"] != "issue30-policy-v3"
        or policy["status"] != "selected_calibrated_candidate"
        or policy["policy_id"] != digest({k: v for k, v in policy.items() if k != "policy_id"})
    ):
        raise ValueError("Expected an intact confirmed v3 policy")
    plan = policy["evidence"]["plan"]
    if plan["plan_id"] != digest({k: v for k, v in plan.items() if k != "plan_id"}):
        raise ValueError("Policy plan checksum changed")
    if policy["config"] != plan["candidates"][policy["candidate"]]:
        raise ValueError("Policy differs from its preregistered candidate")
    if plan["numeric_versions"] != {"numpy": np.__version__, "opencv": cv2.__version__}:
        raise ValueError("Policy numeric library versions changed")
    for name in ("preprocessing_experiments.py", "preprocess.py", "stress.py"):
        if file_hash(Path(__file__).with_name(name)) != plan["implementation"][name]:
            raise ValueError("Policy implementation changed")
    return apply_candidate(image, policy["config"], {"recipes": policy["evidence"]["v2_recipes"]})


def run_comparison(
    protocol,
    manifest,
    pilot,
    data_root,
    converted,
    output,
    configs,
    v2,
    checkpoint,
    *,
    allow_inference=False,
):
    """Run/replay the existing native evaluator for one bounded, paired calibration matrix."""
    from .evaluation_stats import load_stats
    from .five_class import remap_boxes
    from .stress_eval import evaluate_calibration, evaluate_transfer

    output = Path(output)
    if list(configs)[:1] != ["skip"] or not {"skip", "frozen_v2"} <= configs.keys():
        raise ValueError("Paired comparison requires Skip first and frozen v2")
    minimum_class_groups = min(
        len({r["group"] for r in pilot["rows"] if cls in {b[0] for b in remap_boxes(r["boxes"])}})
        for cls in range(len(FIVE_CLASSES))
    )
    records = []
    for scenario in manifest["scenarios"]:
        for name, config in configs.items():
            directory = output / "metrics" / scenario["id"] / name
            dataset = output / "inputs" / scenario["id"] / name
            if not allow_inference and not (directory / "native_result.json").is_file():
                return None
            meta = export_candidate(
                protocol, manifest, pilot, data_root, converted, dataset, scenario, config, v2
            )
            if pilot["role"] == "development_validation":
                result = evaluate_transfer(
                    protocol,
                    manifest,
                    pilot,
                    dataset,
                    checkpoint,
                    directory,
                    allow_inference=allow_inference,
                )
            else:
                result = evaluate_calibration(
                    protocol, pilot, dataset, checkpoint, directory, allow_inference=allow_inference
                )
            record = {
                "candidate": name,
                "scenario": scenario["id"],
                "family": scenario["family"],
                "synthetic": scenario["synthetic"],
                **result["overall"],
                "full_class_support": all(c["instances"] > 0 for c in result["classes"]),
                "minimum_class_groups": minimum_class_groups,
                "mean_ms": float(meta.restore_ms.mean()),
                "p95_ms": float(meta.restore_ms.quantile(0.95)),
                "decisions": meta.decisions.value_counts().to_dict(),
                "classes": result["classes"],
                "native_result_sha256": file_hash(directory / "native_result.json"),
                "statistics_sha256": file_hash(directory / "paired_stats.npz"),
            }
            stats = load_stats(directory / "paired_stats.npz")
            raw = load_stats(output / "metrics" / scenario["id"] / "skip/paired_stats.npz")
            record.update(detection_report(pilot, stats, raw))
            records.append(record)
    return records
