"""Round 2 data audits and lazy synthetic scenarios; no training or final-test pixels."""

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .adaptive import validate_policy
from .common import digest, file_hash, read_image, read_json
from .data import scan_bdd
from .datasets import verify_label, verify_role
from .five_class import FIVE_CLASSES, SIX_TO_FIVE, support_audit
from .offline import encoding_policy, verify_alignment
from .protocol import annotation_profile, group_key, validate_protocol
from .stress import CORRUPTIONS, corrupt
from .subsets import validate_subsets


def audit_data(protocol, data_root, five_root, subsets, conversions, policy):
    """Read-only audit of frozen sources and the existing S10/development five-class pairs."""
    validate_protocol(protocol)
    validate_subsets(subsets, protocol)
    validate_policy(policy, protocol)
    data_root, five_root = Path(data_root).resolve(), Path(five_root).resolve()
    summary = read_json(five_root / "summary.json")
    if (
        summary["source_protocol_id"] != protocol["manifest_id"]
        or summary["source_subset_id"] != subsets["manifest_id"]
        or summary["taxonomy"] != FIVE_CLASSES
    ):
        raise ValueError("Stale five-class summary: protocol/subsets/taxonomy differ")
    artifacts = {str(five_root / "summary.json"): file_hash(five_root / "summary.json")}
    for source in protocol["annotation_sources"].values():
        path = data_root / source["path"]
        if file_hash(path) != source["sha256"]:
            raise ValueError(f"Stale original annotations: {path}")
        artifacts[str(path)] = source["sha256"]
    records, discovery, _ = scan_bdd(data_root, strict=False)
    lookup = {r["id"]: r for r in records}
    for role, rows in protocol["splits"].items():
        for row in rows:
            record = lookup.get(row["id"])
            path = data_root / row["source_path"]
            if (
                record is None
                or Path(record["path"]).resolve() != path
                or record["official_split"] != row["official_split"]
                or digest(record["labels"]) != row["labels_sha256"]
                or group_key(record) != row["source_group"]
                or {
                    key: str(record["attributes"].get(key) or "<missing>")
                    for key in row["attributes"]
                }
                != row["attributes"]
                or annotation_profile(record, row["width"], row["height"])[0] != row["boxes"]
            ):
                raise ValueError(f"Missing/stale frozen source or labels: {row['id']}")
            if role != "final_test":
                if file_hash(path) != row["sha256"]:
                    raise ValueError(f"Stale source image: {row['id']}")
                with Image.open(path) as image:
                    if image.size != (row["width"], row["height"]):
                        raise ValueError(f"Source geometry changed: {row['id']}")
    role_support = {role: support_audit(rows) for role, rows in protocol["splits"].items()}
    if summary["role_support_annotation_only"] != role_support:
        raise ValueError("Stale five-class role support")
    train_ids = set(subsets["subsets"]["S10"]["ids"])
    populations = {
        "training": [r for r in protocol["splits"]["training_pool"] if r["id"] in train_ids],
        "development": protocol["splits"]["development_validation"],
    }
    exports = {}
    for name, rows in populations.items():
        split = "train" if name == "training" else "tune"
        conversion = five_root / "conversion" / ("S10" if name == "training" else name)
        identity = read_json(conversion / "conversion.json")
        original = Path(conversions[name]).resolve()
        if (
            identity != summary[f"{name}_conversion"]
            or identity["classes"] != FIVE_CLASSES
            or identity["mapping"] != {str(k): v for k, v in SIX_TO_FIVE.items()}
            or identity["role_ids_sha256"] != digest({split: sorted(r["id"] for r in rows)})
            or identity["source_conversion_sha256"] != file_hash(original / "conversion.json")
        ):
            raise ValueError(f"Stale five-class {name} conversion")
        for row in rows:
            label_name = f"{row['id']}.txt"
            original_label = original / "labels" / split / label_name
            derived_label = conversion / "labels" / split / label_name
            if any(p.resolve() != p or not p.is_file() for p in (original_label, derived_label)):
                raise ValueError(f"Missing/unsafe frozen label: {row['id']}")
            verify_label(original_label, row)
            old = original_label.read_bytes()
            lines = old.splitlines(keepends=True)
            if [line.split()[0] for line in lines] != [str(b[0]).encode() for b in row["boxes"]]:
                raise ValueError(f"Historical label classes changed: {row['id']}")
            expected = b"".join(
                line.replace(line.split()[0], str(SIX_TO_FIVE[int(line.split()[0])]).encode(), 1)
                for line in lines
                if SIX_TO_FIVE[int(line.split()[0])] is not None
            )
            if derived_label.read_bytes() != expected:
                raise ValueError(f"Five-class label bytes changed: {row['id']}")
        pair = five_root / "datasets/S10" if name == "training" else five_root / name
        pair_identity = read_json(pair / "pair.json")
        if (
            pair_identity != summary[f"{name}_pair"]
            or pair_identity["protocol_id"] != protocol["manifest_id"]
            or pair_identity["records_sha256"] != digest(rows)
            or pair_identity["policy_id"] != policy["policy_id"]
        ):
            raise ValueError(f"Stale {name} pair")
        for branch, variant in (("raw", "raw_reencoded"), ("adaptive", "adaptive")):
            directory = pair / branch
            export = verify_role(
                directory,
                rows,
                split,
                variant,
                policy["recipes"] if branch == "adaptive" else None,
                write_receipt=False,
            )
            if export["export"]["conversion"] != identity:
                raise ValueError(f"Wrong conversion in {directory}")
            for row in rows:
                filename = f"{row['id']}.txt"
                if (directory / "labels" / split / filename).read_bytes() != (
                    conversion / "labels" / split / filename
                ).read_bytes():
                    raise ValueError(f"Export label bytes changed: {row['id']}")
            exports[f"{name}/{branch}"] = {
                "images": len(rows),
                "files_identity": export["files_identity"],
            }
            for filename in ("export.json", "metadata.csv", "verified.json", "data.yaml"):
                artifacts[str(directory / filename)] = file_hash(directory / filename)
        verify_alignment([pair / "raw", pair / "adaptive"], deep=False)
        for path in (
            conversion / "conversion.json",
            original / "conversion.json",
            pair / "pair.json",
        ):
            artifacts[str(path)] = file_hash(path)
    audit = {
        "version": "issue28-data-audit-v1",
        "protocol_id": protocol["manifest_id"],
        "subsets_id": subsets["manifest_id"],
        "classes": FIVE_CLASSES,
        "role_support_annotation_only": role_support,
        "discovery": discovery,
        "exports": exports,
        "artifacts": artifacts,
        "final_test": "metadata only; no image opens, hashes, headers or materialized labels",
    }
    audit["audit_id"] = digest(audit)
    return audit


def validate_audit(audit, protocol):
    validate_protocol(protocol)
    if (
        audit["audit_id"] != digest({k: v for k, v in audit.items() if k != "audit_id"})
        or audit["protocol_id"] != protocol["manifest_id"]
        or audit["classes"] != FIVE_CLASSES
    ):
        raise ValueError("Data audit checksum/protocol/taxonomy changed; rerun notebook 01")
    for path, expected in audit["artifacts"].items():
        if file_hash(path) != expected:
            raise ValueError(f"Audited prerequisite changed; rerun notebook 01: {path}")
    return audit


def evaluation_sets(protocol, *, seed=2026):
    """Partition calibration groups within their frozen role; retain all development IDs."""
    validate_protocol(protocol)
    if type(seed) is not int or seed < 0:
        raise ValueError("Seed must be a nonnegative integer")
    groups = defaultdict(list)
    for row in protocol["splits"]["calibration"]:
        groups[row["group"]].append(row["id"])
    if len(groups) < 2:
        raise ValueError("Exploration/confirmation require at least two calibration groups")
    ordered = sorted(groups, key=lambda g: (digest([seed, g]), g))
    exploration = set(ordered[: len(ordered) // 2])
    sets = {
        "calibration_exploration": sorted(
            i for g in ordered if g in exploration for i in groups[g]
        ),
        "calibration_confirmation": sorted(
            i for g in ordered if g not in exploration for i in groups[g]
        ),
        "development_validation": sorted(
            r["id"] for r in protocol["splits"]["development_validation"]
        ),
    }
    scenarios = [
        {"id": "clean", "family": "clean", "severity": 0, "synthetic": False, "params": {}}
    ]
    for family in ("underexposure", "shot_read", "motion"):
        scenarios.extend(
            {
                "id": f"synthetic_{family}_s{severity}",
                "family": family,
                "severity": severity,
                "synthetic": True,
                "params": params,
            }
            for severity, params in enumerate(CORRUPTIONS[family], 1)
        )
    manifest = {
        "version": "issue28-evaluation-sets-v1",
        "protocol_id": protocol["manifest_id"],
        "classes": FIVE_CLASSES,
        "mapping": {str(k): v for k, v in SIX_TO_FIVE.items()},
        "seed": seed,
        "selection": "seeded SHA256 order of whole calibration groups; equal group halves",
        "sets": sets,
        "scenarios": scenarios,
        "encoding": encoding_policy("jpg", 95),
        "numpy": np.__version__,
        "corruption_code_sha256": file_hash(Path(__file__).with_name("stress.py")),
        "preparation_code_sha256": file_hash(__file__),
        "pixel_pipeline": "original source -> corruption (or copy for clean) -> JPEG95 once",
        "limitations": [
            "Synthetic conditions are controlled approximations, not real camera physics.",
            "Confirmation is disjoint within Round 2; historical v2 used the calibration role.",
            "Equal group halves need not have equal image counts or balanced class support.",
            "Development is evaluation only; confirmation is for shortlisted calibration choices.",
            "No final-test IDs or pixels are included; final evaluation belongs to issue 8.",
        ],
    }
    manifest["manifest_id"] = digest(manifest)
    return manifest


def scenario_image(row, data_root, scenario, *, seed=2026):
    """Prepare one image in memory from an original train source; never write bulk exports."""
    if row["official_split"] != "train" or not row["sha256"]:
        raise ValueError("Only non-final frozen train-role images may be prepared")
    family, severity = scenario["family"], scenario["severity"]
    if (
        type(seed) is not int
        or seed < 0
        or type(severity) is not int
        or (family == "clean" and severity != 0)
        or (
            family != "clean"
            and (
                family not in {"underexposure", "shot_read", "motion"} or severity not in (1, 2, 3)
            )
        )
    ):
        raise ValueError("Unknown scenario family/severity/seed")
    expected = {} if family == "clean" else CORRUPTIONS[family][severity - 1]
    if scenario["params"] != expected or scenario["synthetic"] != (family != "clean"):
        raise ValueError("Scenario parameters/synthetic label differ from the frozen corruption")
    root = Path(data_root).resolve()
    path = root / row["source_path"]
    if not path.is_relative_to(root) or path.resolve() != path:
        raise ValueError("Unsafe source image path")
    if file_hash(path) != row["sha256"]:
        raise ValueError(f"Source image changed: {row['id']}")
    image = read_image(path)
    if image.shape != (row["height"], row["width"], 3):
        raise ValueError("Source image geometry changed")
    result = corrupt(image, family, severity, row["id"], seed)
    ok, encoded = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise OSError("Scenario JPEG95 encoding failed")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None or decoded.shape != image.shape:
        raise ValueError("Scenario encoding changed image geometry")
    return decoded
