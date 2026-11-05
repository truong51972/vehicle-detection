"""P4: prepare or verify frozen Raw/v2 S10 inputs without changing historical exports."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import pandas as pd

from .adaptive import export_pair, validate_policy
from .common import digest, file_hash, read_json
from .data import scan_bdd
from .datasets import verify_role
from .evaluation_data import scenario_source
from .five_class import FIVE_CLASSES, derive_labels, support_audit
from .offline import export_variant, verify_alignment
from .preprocess import apply_recipe
from .protocol import validate_protocol
from .subsets import training_records, validate_subsets


def prepare_training_data(
    protocol, subsets, policy, data_root, prepared, train_conversion, dev_conversion
):
    """Reuse byte-equivalent inputs; create missing exports only in an independent root."""
    validate_protocol(protocol)
    validate_subsets(subsets, protocol)
    validate_policy(policy, protocol)
    data_root, prepared = Path(data_root).resolve(), Path(prepared).resolve()
    if prepared.is_relative_to(data_root) or data_root.is_relative_to(prepared):
        raise ValueError("Preparation root must be independent of original data")
    for source in protocol["annotation_sources"].values():
        if file_hash(data_root / source["path"]) != source["sha256"]:
            raise ValueError("Original annotation sources changed")
    ids = set(subsets["subsets"]["S10"]["ids"])
    rows = [r for r in protocol["splits"]["training_pool"] if r["id"] in ids]
    dev_rows = protocol["splits"]["development_validation"]
    pair, development = prepared / "datasets/S10", prepared / "development/raw"
    # Existing directories are audited, never repaired or regenerated in place.
    if not pair.exists() or not development.exists():
        records, _, sources = scan_bdd(data_root, strict=False)
        if any(
            sources[s]["sha256"] != identity["sha256"]
            for s, identity in protocol["annotation_sources"].items()
        ):
            raise ValueError("Original annotation sources changed")
        if not pair.exists():
            conversion = prepared / "conversion/S10"
            derive_labels(train_conversion, conversion, {"train": rows})
            partitions = training_records(protocol, subsets, "S10", records, data_root)
            export_pair(partitions, conversion, pair, policy, protocol, subsets, "S10")
        if not development.exists():
            conversion = prepared / "conversion/development"
            derive_labels(dev_conversion, conversion, {"tune": dev_rows})
            export_variant(
                {
                    "tune": [
                        {
                            "id": r["id"],
                            "path": str(data_root / r["source_path"]),
                            "sha256": r["sha256"],
                        }
                        for r in dev_rows
                    ]
                },
                conversion,
                development,
                "raw_reencoded",
                policy["recipes"],
            )
            verify_alignment([development])
    binding = read_json(pair / "pair.json")
    expected = {
        "protocol_id": protocol["manifest_id"],
        "subsets_id": subsets["manifest_id"],
        "policy_id": policy["policy_id"],
        "subset": "S10",
        "records_sha256": subsets["subsets"]["S10"]["records_sha256"],
        "encoding": policy["encoding"],
    }
    if any(binding.get(k) != value for k, value in expected.items()):
        raise ValueError("S10 pair differs from frozen protocol/subsets/policy")
    exports, artifacts, routes = {}, {}, []
    for name, directory, population, split, variant in (
        ("raw", pair / "raw", rows, "train", "raw_reencoded"),
        ("adaptive", pair / "adaptive", rows, "train", "adaptive"),
        ("development", development, dev_rows, "tune", "raw_reencoded"),
    ):
        exports[name] = verify_role(
            directory,
            population,
            split,
            variant,
            policy["recipes"] if name == "adaptive" else None,
            write_receipt=False,
        )
        if exports[name]["export"]["conversion"]["classes"] != FIVE_CLASSES:
            raise ValueError("P4 requires the fixed five-class mapping")
        verify_alignment([directory], deep=False)
        for filename in ("export.json", "metadata.csv", "verified.json", "data.yaml"):
            artifacts[str((directory / filename).relative_to(prepared))] = file_hash(
                directory / filename
            )
    verify_alignment([pair / "raw", pair / "adaptive"], deep=False)
    artifacts["datasets/S10/pair.json"] = file_hash(pair / "pair.json")
    tables = {
        name: pd.read_csv(Path(export["path"]) / "metadata.csv").set_index("id")
        for name, export in exports.items()
    }
    # Replay from original pixels proves encoding and actual routes, beyond metadata claims.
    for population, branches in ((rows, ("raw", "adaptive")), (dev_rows, ("development",))):
        for row in population:
            image = scenario_source(row, data_root)
            for branch in branches:
                variant = "adaptive" if branch == "adaptive" else "raw"
                output, decisions, _ = apply_recipe(image, variant, policy["recipes"])
                ok, encoded = cv2.imencode(".jpg", output, [cv2.IMWRITE_JPEG_QUALITY, 95])
                metadata = tables[branch].loc[row["id"]]
                if (
                    not ok
                    or hashlib.sha256(encoded.tobytes()).hexdigest() != metadata.output_sha256
                    or json.loads(metadata.decisions) != decisions
                    or Path(metadata.source) != data_root / row["source_path"]
                ):
                    raise ValueError(f"Original-pixel encoding/route replay differs: {row['id']}")
                if branch == "adaptive":
                    routes.append(
                        {
                            "id": row["id"],
                            "source_group": row["source_group"],
                            "source_sha256": row["sha256"],
                            "raw_sha256": tables["raw"].loc[row["id"], "output_sha256"],
                            "adaptive_sha256": metadata.output_sha256,
                            "label_sha256": metadata.label_sha256,
                            "route": "/".join(d["name"] for d in decisions) or "skip",
                        }
                    )
    receipt = {
        "version": "round2-training-data-v1",
        **expected,
        "classes": FIVE_CLASSES,
        "policy_sha256": digest(policy),
        "exports": exports,
        "artifacts": artifacts,
        "routes_sha256": digest(routes),
        "route_counts": dict(Counter(r["route"] for r in routes)),
        "training_support": support_audit(rows),
        "original_pixel_replay": True,
        "paired_labels_geometry_ids": True,
        "final_test_accessed": False,
    }
    receipt["preparation_id"] = digest(receipt)
    return receipt, routes


def validate_preparation(receipt, prepared):
    """P5 checks saved P4 evidence and inventories without launching preparation."""
    if (
        receipt["version"] != "round2-training-data-v1"
        or receipt["preparation_id"]
        != digest({k: v for k, v in receipt.items() if k != "preparation_id"})
        or receipt["classes"] != FIVE_CLASSES
    ):
        raise ValueError("P4 receipt changed; run notebook 04")
    prepared = Path(prepared).resolve()
    for relative, expected in receipt["artifacts"].items():
        path = prepared / relative
        if not path.resolve().is_relative_to(prepared) or file_hash(path) != expected:
            raise ValueError("Prepared artifact changed; run notebook 04")
    for export in receipt["exports"].values():
        if not Path(export["path"]).is_relative_to(prepared):
            raise ValueError("Prepared export escapes configured root")
        verify_alignment([export["path"]], deep=False)
    return receipt
