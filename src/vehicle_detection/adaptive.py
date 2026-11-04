"""Calibration-only bounded Adaptive recipe and matched offline pilot exports."""

import copy
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .common import digest, file_hash, read_image, read_json, write_json
from .offline import export_variant, verify_alignment
from .preprocess import apply_recipe, quality, validate_recipes
from .protocol import summarize, validate_protocol

METRICS = ("brightness", "contrast", "noise", "sharpness", "black_fraction", "white_fraction")


def proxy_loss(scores):
    """Conservative visual diagnostic, explicitly not a detector objective."""
    return (
        max(0, 0.25 - scores["brightness"])
        + max(0, 0.12 - scores["contrast"])
        + 2 * max(0, scores["noise"] - 0.035)
        + 0.05 * max(0, 1 - scores["sharpness"] / 80)
        + 2 * (scores["black_fraction"] + scores["white_fraction"])
    )


def _jpeg_scores(image):
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise OSError("Calibration JPEG encoding failed")
    return quality(cv2.imdecode(encoded, cv2.IMREAD_COLOR))


def _implementation():
    return {
        "calibration_code": file_hash(__file__),
        "operator_code": file_hash(Path(__file__).with_name("preprocess.py")),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
    }


def calibrate_policy(
    protocol, dataset_root, *, seed=2026, sample_size=256, per_operator=12, min_support=12
):
    """Fit v2 on seeded route samples; insufficiently supported routes remain uncalibrated."""
    validate_protocol(protocol)
    for value in (sample_size, per_operator, min_support):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("Calibration sample sizes must be positive integers")
    rows = sorted(protocol["splits"]["calibration"], key=lambda r: r["id"])
    rng = random.Random(seed)
    selected = sorted(rng.sample(rows, min(sample_size, len(rows))), key=lambda r: r["id"])
    images, measurements = {}, []
    root = Path(dataset_root).resolve()
    for row in selected:
        path = root / row["source_path"]
        if file_hash(path) != row["sha256"]:
            raise ValueError("Calibration source changed")
        image = read_image(path)
        if image.shape != (row["height"], row["width"], 3):
            raise ValueError("Calibration geometry changed")
        images[row["id"]] = image
        measurements.append({"id": row["id"], **quality(image)})
    table = pd.DataFrame(measurements)
    recipes = {
        "raw": {},
        "gamma": {"gamma": 0.95},
        "clahe": {"clip_limit": 1.2, "grid_size": 8},
        "denoise": {"strength": 2.0},
        "sharpen": {"amount": 0.15, "sigma": 1.0},
        "adaptive": {
            "brightness_below": float(np.clip(table.brightness.quantile(0.2), 0.12, 0.28)),
            "contrast_below": float(np.clip(table.contrast.quantile(0.2), 0.06, 0.12)),
            "noise_above": float(np.clip(table.noise.quantile(0.9), 0.035, 0.12)),
            "sharpness_min": 10.0,
            "sharpness_below": float(np.clip(table.sharpness.quantile(0.2), 10, 80)),
            "max_filters": 1,
        },
    }
    candidates = {
        "gamma": [{"gamma": v} for v in (0.95, 0.85)],
        "clahe": [{"clip_limit": v, "grid_size": 8} for v in (1.2, 1.5)],
        "denoise": [{"strength": v} for v in (2.0, 3.0)],
        "sharpen": [{"amount": v, "sigma": 1.0} for v in (0.15, 0.3)],
    }
    noise_diagnostics = {
        "status": "unvalidated_proxy",
        "sample_count": len(table),
        "zero_count": int((table.noise == 0).sum()),
        "maximum": float(table.noise.max()),
        "initial_routing_threshold": recipes["adaptive"]["noise_above"],
    }
    tuning, route_support = [], {}
    for operator, settings in candidates.items():
        # Evaluate the actual priority route, including any previously disabled operator.
        eligible = []
        for row in selected:
            _, decisions, _ = apply_recipe(images[row["id"]], "adaptive", recipes)
            if decisions and decisions[0]["name"] == operator:
                eligible.append(row["id"])
        candidate_ids = sorted(rng.sample(eligible, min(per_operator, len(eligible))))
        supported = len(candidate_ids) >= min_support
        before = {i: _jpeg_scores(images[i]) for i in candidate_ids} if supported else {}
        best_delta, best_setting = 0.0, None  # Skip wins ties and unsupported routes.
        for setting in settings:
            trial = {**recipes, operator: setting}
            deltas = []
            for sample_id in candidate_ids if supported else []:
                output, _, _ = apply_recipe(images[sample_id], operator, trial)
                after = _jpeg_scores(output)
                deltas.append(proxy_loss(after) - proxy_loss(before[sample_id]))
            delta = float(np.mean(deltas)) if deltas else None
            tuning.append(
                {
                    "operator": operator,
                    "settings": setting,
                    "ids": candidate_ids,
                    "mean_proxy_loss_delta": delta,
                }
            )
            if delta is not None and delta < best_delta:
                best_delta, best_setting = delta, setting
        if best_setting is not None:
            recipes[operator] = best_setting
        else:
            disable = {
                "gamma": {"brightness_below": 0.0},
                "clahe": {"contrast_below": 0.0},
                "denoise": {"noise_above": 1.0},
                "sharpen": {"sharpness_min": 0.0, "sharpness_below": 0.0},
            }
            recipes["adaptive"].update(disable[operator])
        route_support[operator] = {
            "eligible_count": len(eligible),
            "candidate_count": len(candidate_ids),
            "candidate_ids": candidate_ids,
            "minimum_support": min_support,
            "status": "uncalibrated_insufficient_support"
            if not supported
            else (
                "proxy_calibrated_filter" if best_setting is not None else "proxy_calibrated_skip"
            ),
        }
    validate_recipes(recipes)
    policy = {
        "version": "issue6-policy-v2",
        "protocol_id": protocol["manifest_id"],
        "implementation": _implementation(),
        "seed": seed,
        "sample_size": sample_size,
        "per_operator": per_operator,
        "minimum_support": min_support,
        "candidate_selection": "seeded uniform image sample over all eligible route cases",
        "route_support": route_support,
        "noise_diagnostics": noise_diagnostics,
        "calibration_rows_sha256": digest(selected),
        "calibration_ids": [r["id"] for r in selected],
        "sample_summary": summarize(selected),
        "measurements": measurements,
        "candidates": tuning,
        "recipes": recipes,
        "encoding": {"format": "jpg", "jpeg_quality": 95},
        "limits": [
            "Calibration uses a bounded seeded sample, not every calibration frame.",
            "At most one filter; fixed priority Gamma, CLAHE, Denoise, Sharpen, otherwise skip.",
            "Quality loss is a heuristic proxy; lower loss does not prove higher detection AP.",
            "Small candidate support can miss adverse cases; inspect paired pilot examples.",
            "Minimum support is an engineering gate, not a statistical adequacy guarantee.",
            "Noise proxy is unvalidated and can collapse to zero; unsupported routes stay off.",
            "No development-validation or final-test outcome enters recipe fitting.",
        ],
    }
    policy["policy_id"] = digest(policy)
    return validate_policy(policy, protocol)


def validate_policy(policy, protocol):
    validate_protocol(protocol)
    if policy["policy_id"] != digest({k: v for k, v in policy.items() if k != "policy_id"}):
        raise ValueError("Policy checksum changed")
    if policy["version"] != "issue6-policy-v2":
        raise ValueError(
            "Frozen v1 policy requires its recorded source revision; use a new v2 root"
        )
    if (
        policy["protocol_id"] != protocol["manifest_id"]
        or policy["implementation"] != _implementation()
    ):
        raise ValueError("Policy protocol/implementation changed; use a new version/root")
    by_id = {r["id"]: r for r in protocol["splits"]["calibration"]}
    ids = policy["calibration_ids"]
    if (
        not ids
        or len(set(ids)) != len(ids)
        or set(ids) - by_id.keys()
        or digest([by_id[i] for i in ids]) != policy["calibration_rows_sha256"]
    ):
        raise ValueError("Policy must use only frozen calibration IDs")
    validate_recipes(policy["recipes"])
    minimum = policy["minimum_support"]
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
        raise ValueError("Policy minimum support must be a positive integer")
    if set(policy["route_support"]) != {"gamma", "clahe", "denoise", "sharpen"}:
        raise ValueError("Missing policy route support")
    settings = policy["recipes"]["adaptive"]
    disabled = {
        "gamma": settings["brightness_below"] == 0,
        "clahe": settings["contrast_below"] == 0,
        "denoise": settings["noise_above"] == 1,
        "sharpen": settings["sharpness_min"] == settings["sharpness_below"],
    }
    for operator, support in policy["route_support"].items():
        candidates = support["candidate_ids"]
        supported = len(candidates) >= minimum
        valid_statuses = (
            {"proxy_calibrated_filter", "proxy_calibrated_skip"}
            if supported
            else {"uncalibrated_insufficient_support"}
        )
        if (
            len(set(candidates)) != len(candidates)
            or set(candidates) - set(ids)
            or support["candidate_count"] != len(candidates)
            or support["eligible_count"] < len(candidates)
            or support["minimum_support"] != minimum
            or support["status"] not in valid_statuses
            or (support["status"] != "proxy_calibrated_filter" and not disabled[operator])
        ):
            raise ValueError("Policy route support/status mismatch")
    if policy["recipes"]["adaptive"]["max_filters"] != 1:
        raise ValueError("Issue #6 policy permits at most one operator")
    return policy


def freeze_policy(path, policy, protocol):
    validate_policy(policy, protocol)
    path = Path(path)
    if path.exists() and read_json(path) != policy:
        raise ValueError("Immutable policy changed; use a new version/root")
    if not path.exists():
        write_json(path, policy)


def export_pair(partitions, converted, destination, policy, protocol, subsets, name, *, classes=None):
    """Bind both native exports to the frozen subset and calibration recipe before writing."""
    from .subsets import validate_subsets

    validate_policy(policy, protocol)
    validate_subsets(subsets, protocol)
    expected = subsets["subsets"][name]["ids"]
    if set(partitions) != {"train"} or sorted(r["id"] for r in partitions["train"]) != expected:
        raise ValueError("Pair export must contain exactly the selected training IDs")
    frozen = {r["id"]: r for r in protocol["splits"]["training_pool"]}
    for row in partitions["train"]:
        source = frozen[row["id"]]
        if any(row[k] != source[k] for k in ("sha256", "width", "height")) or (
            digest(row["labels"]) != source["labels_sha256"]
        ):
            raise ValueError("Paired source annotations/geometry changed")
    destination = Path(destination)
    identity = {
        "protocol_id": protocol["manifest_id"],
        "subsets_id": subsets["manifest_id"],
        "subset": name,
        "policy_id": policy["policy_id"],
        "records_sha256": subsets["subsets"][name]["records_sha256"],
        "exporter_code": file_hash(Path(__file__).with_name("offline.py")),
        "encoding": policy["encoding"],
    }
    if classes is not None:
        identity["classes"] = list(classes)
        identity["conversion"] = read_json(Path(converted) / "conversion.json")
    identity_path = destination / "pair.json"
    if identity_path.exists():
        if read_json(identity_path) != identity:
            raise ValueError("Pair identity changed; use a fresh root")
    elif destination.exists() and any(destination.iterdir()):
        raise ValueError("Unknown existing pair directory")
    else:
        write_json(identity_path, identity)
    tables = {}
    for branch, variant in [("raw", "raw_reencoded"), ("adaptive", "adaptive")]:
        tables[branch] = export_variant(
            partitions,
            converted,
            destination / branch,
            variant,
            copy.deepcopy(policy["recipes"]),
            output_format="jpg",
            jpeg_quality=95,
            classes=classes,
        )
    verified_count = verify_alignment([destination / b for b in tables], classes=classes)
    if verified_count != len(expected):
        raise ValueError(
            f"Paired export count mismatch: expected {len(expected)}, got {verified_count}"
        )
    return tables


def paired_quality(destination, *, classes=None):
    """Measure decoded JPEGs from both branches, so reported deltas share compression."""
    destination = Path(destination)
    verify_alignment([destination / b for b in ("raw", "adaptive")], classes=classes)
    raw = pd.read_csv(destination / "raw/metadata.csv").set_index("id")
    adaptive = pd.read_csv(destination / "adaptive/metadata.csv").set_index("id")
    rows = []
    for sample_id in sorted(raw.index):
        before = quality(read_image(raw.loc[sample_id, "image"]))
        after = quality(read_image(adaptive.loc[sample_id, "image"]))
        rows.append(
            {
                "id": sample_id,
                "proxy_loss_delta": proxy_loss(after) - proxy_loss(before),
                **{f"raw_{k}": before[k] for k in METRICS},
                **{f"adaptive_{k}": after[k] for k in METRICS},
                **{f"delta_{k}": after[k] - before[k] for k in METRICS},
            }
        )
    return pd.DataFrame(rows)
