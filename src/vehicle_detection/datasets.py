"""Frozen role/box checks shared by training and development evaluation."""

from pathlib import Path

import numpy as np
import pandas as pd

from .common import read_json
from .offline import verify_alignment


def verify_role(directory, records, split, variant, recipes=None, *, deep=True):
    """Reject wrong roles before decoding pixels, then verify owned files and frozen boxes."""
    directory = Path(directory).resolve()
    meta = pd.read_csv(directory / "metadata.csv")
    selected = {r["id"]: r for r in records}
    if meta.id.duplicated().any() or set(meta.id) != set(selected) or set(meta.split) != {split}:
        raise ValueError("Export must contain exactly the frozen role IDs, with no duplicates")
    export = read_json(directory / "export.json")
    if (
        export["variant"] != variant
        or export["encoding"]["format"] != "jpg"
        or export["encoding"]["jpeg_quality"] != 95
        or (variant == "raw_reencoded" and export["recipe"] != {})
        or (recipes is not None and export["recipe"] != recipes)
    ):
        raise ValueError("Export treatment/encoding differs from the frozen policy")
    for row in meta.to_dict("records"):
        source = selected[row["id"]]
        image = directory / "images" / split / f"{row['id']}.jpg"
        label = directory / "labels" / split / f"{row['id']}.txt"
        if (
            Path(row["image"]) != image
            or image.resolve() != image
            or label.resolve() != label
            or row["source_sha256"] != source["sha256"]
            or (row["width"], row["height"]) != (source["width"], source["height"])
        ):
            raise ValueError(
                "Export path, source hash or geometry changed; re-export after relocation"
            )
        actual = [list(map(float, line.split())) for line in label.read_text().splitlines()]
        expected = [
            [
                c,
                (x1 + x2) / 2 / source["width"],
                (y1 + y2) / 2 / source["height"],
                (x2 - x1) / source["width"],
                (y2 - y1) / source["height"],
            ]
            for c, x1, y1, x2, y2 in source["boxes"]
        ]
        if len(actual) != len(expected) or any(len(box) != 5 for box in actual):
            raise ValueError("Export labels differ from the frozen boxes")
        # Native labels preserve source order; sorting rounded centers can swap tied boxes.
        if actual and not np.allclose(actual, expected, atol=1e-6, rtol=0):
            raise ValueError("Export labels differ from the frozen boxes")
    verify_alignment([directory], deep=deep)
    return {
        "path": str(directory),
        "export": export,
        "files_identity": read_json(directory / "verified.json")["files_identity"],
        "ids": sorted(selected),
    }


def export_development_pair(protocol, policy, raw, converted, destination):
    """Prepare Adaptive dev from original sources, sharing frozen Raw labels/JPEG95."""
    from .adaptive import validate_policy
    from .common import digest, freeze_json
    from .offline import export_variant

    validate_policy(policy, protocol)
    raw, converted, destination = map(Path, (raw, converted, destination))
    rows = protocol["splits"]["development_validation"]
    reference = verify_role(raw, rows, "tune", "raw_reencoded")
    freeze_json(
        destination / "pair.json",
        {
            "protocol_id": protocol["manifest_id"],
            "role": "development_validation",
            "policy_id": policy["policy_id"],
            "raw": reference,
            "records_sha256": digest(rows),
        },
    )
    meta = pd.read_csv(raw / "metadata.csv")
    records = [
        {"id": r["id"], "path": r["source"], "sha256": r["source_sha256"]}
        for r in meta.to_dict("records")
    ]
    export_variant(
        {"tune": records}, converted, destination / "adaptive", "adaptive", policy["recipes"]
    )
    verify_role(destination / "adaptive", rows, "tune", "adaptive", policy["recipes"])
    verify_alignment([raw, destination / "adaptive"], deep=False)
    return {"raw": raw, "adaptive": destination / "adaptive"}
