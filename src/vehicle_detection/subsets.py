"""Nested whole-group training subsets of the immutable issue #5 protocol."""

import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .common import digest, file_hash, read_json, write_json
from .data import CLASSES
from .protocol import CONDITIONS, features, summarize, validate_protocol


def subset_features(row):
    counts = features(row)
    classes = Counter(int(box[0]) for box in row["boxes"])
    for axis in CONDITIONS:
        for cls, count in classes.items():
            key = f"{CLASSES[cls]}|{axis}:{row['attributes'][axis]}"
            counts[f"joint_presence:{key}"] = 1
            counts[f"joint_instances:{key}"] = count
    return counts


def build_subsets(protocol, *, seed=2026, attempts=128, include_s40=False):
    """Select one nested prefix sequence by train-only multi-label distribution drift."""
    validate_protocol(protocol)
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
        raise ValueError("attempts must be a positive integer")
    pool = protocol["splits"]["training_pool"]
    fractions = {"S10": 0.1, "S20": 0.2}
    if include_s40:
        fractions["S40"] = 0.4
    groups = defaultdict(list)
    for row in sorted(pool, key=lambda r: r["id"]):
        groups[row["group"]].append(row)
    keys = sorted(groups)
    if len(keys) < len(fractions):
        raise ValueError("Too few independent groups for strictly nested subsets")
    group_counts = [sum((subset_features(r) for r in groups[k]), Counter()) for k in keys]
    totals = sum(group_counts, Counter())
    for cls in CLASSES:
        for prefix in ("presence", "instances"):
            totals.setdefault(f"{prefix}:{cls}", 0)
    columns = sorted(totals)
    matrix = np.array([[c.get(k, 0) for k in columns] for c in group_counts], dtype=float)
    sizes = matrix[:, columns.index("images")]
    total = matrix.sum(axis=0)
    rng, best = random.Random(seed), None
    for _ in range(attempts):
        order = list(range(len(keys)))
        rng.shuffle(order)
        cumulative = np.cumsum(sizes[order])
        cuts, previous = {}, 0
        for i, (name, fraction) in enumerate(fractions.items()):
            # Approximate image percentages while preserving every group and strict nesting.
            upper = len(keys) - (len(fractions) - i - 1)
            cut = min(
                range(previous + 1, upper + 1),
                key=lambda c: abs(cumulative[c - 1] - len(pool) * fraction),
            )
            cuts[name], previous = cut, cut
        score = 0.0
        for name, cut in cuts.items():
            observed = matrix[order[:cut]].sum(axis=0)
            expected = total * fractions[name]
            score += float(np.mean((observed - expected) ** 2 / (expected + 1)))
            score += int(np.sum((observed == 0) & (expected >= 2)))
        if best is None or score < best[0]:
            best = score, {name: [keys[j] for j in order[:cut]] for name, cut in cuts.items()}
    subsets, diagnostics = {}, []
    for name, selected_groups in best[1].items():
        rows = sorted([r for k in selected_groups for r in groups[k]], key=lambda r: r["id"])
        counts = sum((subset_features(r) for r in rows), Counter())
        subsets[name] = {
            "ids": [r["id"] for r in rows],
            "groups": sorted(selected_groups),
            "records_sha256": digest(rows),
            "summary": summarize(rows),
            "target_fraction": fractions[name],
            "actual_fraction": len(rows) / len(pool),
        }
        for key, support in sorted(totals.items()):
            actual = counts.get(key, 0)
            diagnostics.append(
                {
                    "subset": name,
                    "feature": key,
                    "support": actual,
                    "pool_support": support,
                    "expected": support * len(rows) / len(pool),
                    "delta_per_image": actual / len(rows) - support / len(pool),
                    "rare_support": actual < 5,
                    "zero_support": actual == 0,
                }
            )
    manifest = {
        "version": "issue6-subsets-v1",
        "protocol_id": protocol["manifest_id"],
        "pool_sha256": digest(sorted(pool, key=lambda r: r["id"])),
        "seed": seed,
        "algorithm": {
            "name": "minimum joint/marginal drift over nested seeded whole-group prefixes",
            "attempts": attempts,
            "score": best[0],
            "code_sha256": file_hash(__file__),
        },
        "subsets": subsets,
        "diagnostics": diagnostics,
        "limitations": protocol["limitations"]
        + [
            "Percentages approximate image counts; groups are never split.",
            "Rare support <5 requires parent-marginal reporting, not oversampling.",
        ],
    }
    manifest["manifest_id"] = digest(manifest)
    validate_subsets(manifest, protocol)
    return manifest


def validate_subsets(manifest, protocol):
    validate_protocol(protocol)
    if manifest["manifest_id"] != digest({k: v for k, v in manifest.items() if k != "manifest_id"}):
        raise ValueError("Subset manifest checksum changed")
    pool = sorted(protocol["splits"]["training_pool"], key=lambda r: r["id"])
    if manifest["protocol_id"] != protocol["manifest_id"] or manifest["pool_sha256"] != digest(
        pool
    ):
        raise ValueError("Subset parent protocol changed")
    names = list(manifest["subsets"])
    if names != ["S10", "S20"] and names != ["S10", "S20", "S40"]:
        raise ValueError("Expected ordered S10/S20 and optional S40")
    previous = set()
    for name in names:
        subset = manifest["subsets"][name]
        ids, groups = set(subset["ids"]), set(subset["groups"])
        rows = [r for r in pool if r["id"] in ids]
        if (
            not previous < ids
            or len(ids) != len(subset["ids"])
            or ids != {r["id"] for r in pool if r["group"] in groups}
            or digest(rows) != subset["records_sha256"]
            or summarize(rows) != subset["summary"]
        ):
            raise ValueError("Subset IDs, whole groups, nesting or annotations changed")
        previous = ids
    return manifest


def freeze_subsets(path, manifest, protocol):
    validate_subsets(manifest, protocol)
    path = Path(path)
    if path.exists() and read_json(path) != manifest:
        raise ValueError("Immutable subsets changed; choose a new version/root")
    if not path.exists():
        write_json(path, manifest)


def training_records(protocol, subsets, name, records, dataset_root):
    """Recover only selected training inputs, verifying source annotations and paths."""
    validate_subsets(subsets, protocol)
    ids = set(subsets["subsets"][name]["ids"])
    by_id = {r["id"]: r for r in records}
    root, selected = Path(dataset_root).resolve(), []
    for row in protocol["splits"]["training_pool"]:
        if row["id"] not in ids:
            continue
        record = by_id[row["id"]]
        if (
            record["official_split"] != "train"
            or digest(record["labels"]) != row["labels_sha256"]
            or Path(record["path"]).resolve() != root / row["source_path"]
        ):
            raise ValueError("Training source/labels changed")
        selected.append({**record, **{k: row[k] for k in ("width", "height", "sha256")}})
    return {"train": sorted(selected, key=lambda r: r["id"])}
