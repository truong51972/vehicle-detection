"""Issue #5: immutable, train-only-selected data roles; no final-test pixel access."""

import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from .common import digest, file_hash, read_image, read_json, write_json
from .data import ALIASES, CLASSES

ROLES = ("training_pool", "calibration", "development_validation", "final_test")
FRACTIONS = {"training_pool": 0.7, "calibration": 0.1, "development_validation": 0.2}
CONDITIONS = ("weather", "timeofday", "scene")


def group_key(record):
    if record["group"] != record["id"] or record.get("group_source") == "videoName":
        return "video:" + record["group"]
    if re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{8}", record["id"]):
        return "bdd_prefix:" + record["id"].split("-")[0]
    return "sample:" + record["id"]


def annotation_profile(record, width, height):
    boxes, excluded, errors = [], Counter(), Counter()
    for label in record["labels"]:
        attrs = label.get("attributes") or {}
        if attrs.get("crowd") or attrs.get("ignored"):
            excluded["crowd_or_ignored"] += 1
            continue
        try:
            cls = CLASSES.index(ALIASES.get(label["category"], label["category"]))
            coords = [float(label["box2d"][k]) for k in ("x1", "y1", "x2", "y2")]
        except (KeyError, TypeError, ValueError, OverflowError):
            errors["malformed_box"] += 1
            continue
        x1, y1, x2, y2 = coords
        if not np.isfinite(coords).all():
            errors["nonfinite_box"] += 1
        elif not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            errors["invalid_geometry"] += 1
        else:
            boxes.append([cls, *coords])
    unique = list(dict.fromkeys(map(tuple, boxes)))
    excluded["duplicate_boxes"] = len(boxes) - len(unique)
    return [list(b) for b in unique], dict(excluded), dict(errors)


def features(row):
    counts = Counter({"images": 1})
    instances, sizes = Counter(), Counter()
    for cls, x1, y1, x2, y2 in row["boxes"]:
        instances[CLASSES[cls]] += 1
        area = (x2 - x1) * (y2 - y1)
        sizes["small" if area < 32**2 else "medium" if area < 96**2 else "large"] += 1
    counts.update({f"instances:{k}": v for k, v in instances.items()})
    counts.update({f"presence:{k}": 1 for k in instances})
    counts.update({f"size_instances:{k}": v for k, v in sizes.items()})
    counts.update({f"size_presence:{k}": 1 for k in sizes})
    counts["negative"] = int(not row["boxes"])
    attrs = row["attributes"]
    for key in CONDITIONS:
        counts[f"{key}:{attrs[key]}"] = 1
        counts[f"missing_metadata:{key}"] = int(
            attrs[key].lower() in {"<missing>", "undefined", "unknown", "n/a", "none"}
        )
    counts["intersection:" + "|".join(attrs[k] for k in CONDITIONS)] = 1
    return counts


def summarize(rows):
    counts = Counter()
    for row in rows:
        counts.update(features(row))
    for name in CLASSES:
        for prefix in ("presence", "instances"):
            counts.setdefault(f"{prefix}:{name}", 0)
    return dict(sorted(counts.items()))


def _allocate(rows, seed, attempts):
    groups = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row)
    keys = sorted(groups)
    if len(keys) < 3:
        raise ValueError("Need at least three independent training groups")
    totals = summarize(rows)
    columns = sorted(totals)
    group_counts = [summarize(groups[k]) for k in keys]
    matrix = np.array([[counts.get(c, 0) for c in columns] for counts in group_counts])
    sizes = matrix[:, columns.index("images")]
    total = matrix.sum(axis=0)
    rng, best = random.Random(seed), None
    for _ in range(attempts):
        order = list(range(len(keys)))
        rng.shuffle(order)
        cumulative = np.cumsum(sizes[order])
        cut1 = max(1, min(len(keys) - 2, np.searchsorted(cumulative, len(rows) * 0.2) + 1))
        cut2 = max(cut1 + 1, min(len(keys) - 1, np.searchsorted(cumulative, len(rows) * 0.3) + 1))
        allocation = {
            "development_validation": order[:cut1],
            "calibration": order[cut1:cut2],
            "training_pool": order[cut2:],
        }
        score = 0.0
        for role, indices in allocation.items():
            observed, expected = matrix[indices].sum(axis=0), total * FRACTIONS[role]
            score += float(np.mean((observed - expected) ** 2 / (expected + 1)))
            score += int(np.sum((observed == 0) & (expected >= 2)))
        if best is None or score < best[0]:
            best = score, allocation
    return {
        role: sorted([r for i in indices for r in groups[keys[i]]], key=lambda r: r["id"])
        for role, indices in best[1].items()
    }, best[0]


def build_protocol(
    records,
    sources,
    dataset_root,
    excluded_test_ids,
    *,
    seed=2026,
    attempts=64,
    historical=None,
    discovery=None,
):
    """Inspect train pixels only; group risks, select natural proportions and freeze evidence."""
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
        raise ValueError("attempts must be a positive integer")
    root = Path(dataset_root).resolve()
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("Duplicate source IDs")
    if any(r["official_split"] not in {"train", "val"} for r in records):
        raise ValueError("Unknown official split")
    source_identity = {
        split: {**source, "path": str(Path(source["path"]).resolve().relative_to(root))}
        for split, source in sources.items()
    }
    for source in source_identity.values():
        if file_hash(root / source["path"]) != source["sha256"]:
            raise ValueError("Annotation source changed during discovery")
    profiles, exclusions, label_counts = [], [], Counter()
    for record in sorted(records, key=lambda r: r["id"]):
        # Resolve/stat of val paths is allowed; only official train is opened or decoded.
        path = Path(record["path"]).resolve(strict=True)
        relative = str(path.relative_to(root))
        train = record["official_split"] == "train"
        image_hash = near_hash = None
        width, height = 1280, 720  # BDD2018 annotation canvas; val geometry is unverified.
        if train:
            try:
                image = read_image(path)
                height, width = image.shape[:2]
                image_hash = file_hash(path)
                if path.suffix.lower() in {".jpg", ".jpeg"}:
                    with path.open("rb") as stream:
                        stream.seek(-2, 2)
                        if stream.read() != b"\xff\xd9":
                            raise ValueError("Invalid JPEG end marker")
                gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (9, 8))
                near_hash = np.packbits(gray[:, 1:] > gray[:, :-1]).tobytes().hex()
            except (OSError, ValueError, cv2.error) as exc:
                exclusions.append(
                    {"id": record["id"], "reason": "unusable_train_image", "detail": str(exc)}
                )
                continue
        boxes, ignored, errors = annotation_profile(record, width, height)
        label_counts.update(ignored)
        label_counts.update(errors)
        if errors:
            exclusions.append({"id": record["id"], "reason": "invalid_boxes", "detail": errors})
            continue
        attrs = record.get("attributes") or {}
        profiles.append(
            {
                "id": record["id"],
                "source_path": relative,
                "official_split": record["official_split"],
                "source_group": group_key(record),
                "width": width,
                "height": height,
                "geometry_verified": train,
                "sha256": image_hash,
                "dhash64": near_hash,
                "labels_sha256": digest(record["labels"]),
                "boxes": boxes,
                "attributes": {k: str(attrs.get(k) or "<missing>") for k in CONDITIONS},
            }
        )
    training = [r for r in profiles if r["official_split"] == "train"]
    # Merge sequence keys, byte duplicates and equal-dHash near-duplicate candidates transitively.
    parents = {r["source_group"]: r["source_group"] for r in training}

    def find(key):
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    owners, collisions = {}, Counter()
    for row in training:
        for kind in ("sha256", "dhash64"):
            token = (kind, row[kind])
            if token in owners:
                collisions[kind] += 1
                a, b = find(row["source_group"]), find(owners[token])
                parents[max(a, b)] = min(a, b)
            else:
                owners[token] = row["source_group"]
    for row in training:
        row["group"] = find(row["source_group"])
    official_val_ids = {r["id"] for r in records if r["official_split"] == "val"}
    missing_val = (discovery or {}).get("missing_image_ids", {}).get("val", [])
    excluded_test_ids = sorted(set(excluded_test_ids))
    if set(excluded_test_ids) - official_val_ids - set(missing_val):
        raise ValueError("Exposure exclusions absent from official val annotations")
    exposed_groups = {group_key(r) for r in records if r["id"] in excluded_test_ids}
    train_groups = {r["source_group"] for r in training}
    final_test = []
    for row in profiles:
        if row["official_split"] != "val":
            continue
        row["group"] = row["source_group"]
        reason = (
            "prior_exposure"
            if row["id"] in excluded_test_ids
            else "group_related_to_exposure"
            if row["group"] in exposed_groups
            else "group_overlap_official_train"
            if row["group"] in train_groups
            else None
        )
        if reason:
            exclusions.append({"id": row["id"], "reason": reason})
        else:
            final_test.append(row)
    splits, score = _allocate(training, seed, attempts)
    splits["final_test"] = final_test
    summaries = {role: summarize(rows) for role, rows in splits.items()}
    train_summary = summarize(training)
    diagnostics = []
    for role in FRACTIONS:
        n = len(splits[role])
        for feature, support in train_summary.items():
            if feature == "images":
                continue
            actual = summaries[role].get(feature, 0)
            expected = support * n / len(training)
            diagnostics.append(
                {
                    "role": role,
                    "feature": feature,
                    "support": actual,
                    "expected": expected,
                    "delta_per_image": actual / n - support / len(training),
                    "inadequate_support": actual < 5,
                    "fallback": "report parent marginal; no separate inference"
                    if actual < 5
                    else None,
                }
            )
    manifest = {
        "version": "issue5-v1",
        "classes": CLASSES,
        "seed": seed,
        "algorithm": {
            "name": "seeded group permutations, minimum train-only marginal drift",
            "attempts": attempts,
            "fractions": FRACTIONS,
            "score": score,
            "code_sha256": file_hash(__file__),
        },
        "annotation_sources": source_identity,
        "discovery": discovery or {},
        "excluded_test_ids": excluded_test_ids,
        "exclusions": sorted(exclusions, key=lambda r: (r["id"], r["reason"])),
        "label_audit": dict(label_counts),
        "splits": splits,
        "summaries": {
            **summaries,
            "official_train": train_summary,
            "official_val": summarize([r for r in profiles if r["official_split"] == "val"]),
        },
        "diagnostics": diagnostics,
        "group_audit": {
            "training_groups": len({r["group"] for r in training}),
            "source_group_kinds": dict(Counter(r["source_group"].split(":")[0] for r in profiles)),
            "duplicate_candidates": dict(collisions),
        },
        "limitations": [
            "BDD filename prefixes are inferred sequence safeguards, not verified video IDs.",
            "Equal dHash groups candidates only; nonzero-distance near duplicates may remain.",
            "Final-test pixels are unopened; cross-official content duplicates are unverified.",
            "Val geometry uses the BDD2018 1280x720 annotation canvas, not decoded headers.",
            "Rare supports below 5 use parent-marginal reporting; absent classes remain absent.",
        ],
    }
    if historical is not None:
        old_roles = {i: s for s, ids in historical["splits"].items() for i in ids}
        manifest["historical_diff"] = {
            "selection_id": historical["selection_id"],
            "old_sizes": {s: len(ids) for s, ids in historical["splits"].items()},
            "transitions": dict(
                Counter(
                    f"{old_roles.get(r['id'], 'absent')}->{role}"
                    for role, rows in splits.items()
                    if role != "final_test"
                    for r in rows
                )
            ),
        }
    manifest["manifest_id"] = digest(manifest)
    validate_protocol(manifest)
    return manifest


def validate_protocol(manifest):
    """Reusable downstream role/content-integrity guard; opens no source images."""
    if manifest["manifest_id"] != digest({k: v for k, v in manifest.items() if k != "manifest_id"}):
        raise ValueError("Split manifest checksum changed")
    if manifest["classes"] != CLASSES or set(manifest["splits"]) != set(ROLES):
        raise ValueError("Wrong class order or data roles")
    owners = {}
    for role, rows in manifest["splits"].items():
        if not rows:
            raise ValueError(f"Empty data role: {role}")
        for row in rows:
            if row["official_split"] != ("val" if role == "final_test" else "train"):
                raise ValueError("Wrong source population for role")
            if role == "final_test" and (
                row["id"] in manifest["excluded_test_ids"] or row["sha256"] or row["dhash64"]
            ):
                raise ValueError("Exposed or pixel-inspected final test")
            path = Path(row["source_path"])
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Source path must be dataset-relative")
            for key in ("id", "group", "source_group", "sha256", "dhash64"):
                if row[key] is None:
                    continue
                token = (key, row[key])
                if token in owners and (key == "id" or owners[token] != role):
                    raise ValueError(f"Split leakage on {key}: {row['id']}")
                owners[token] = role
        if summarize(rows) != manifest["summaries"][role]:
            raise ValueError("Summary differs from manifest records")
    return manifest


def freeze_protocol(path, manifest):
    validate_protocol(manifest)
    path = Path(path)
    if path.exists():
        if read_json(path) != manifest:
            raise ValueError("Immutable protocol changed; choose a new version/root")
    else:
        write_json(path, manifest)


def development_records(manifest, records, dataset_root):
    """Recover converter inputs only for frozen development IDs, checking paths and labels."""
    validate_protocol(manifest)
    by_id = {r["id"]: r for r in records}
    selected = []
    root = Path(dataset_root).resolve()
    for row in manifest["splits"]["development_validation"]:
        record = by_id[row["id"]]
        if (
            record["official_split"] != "train"
            or digest(record["labels"]) != row["labels_sha256"]
            or Path(record["path"]).resolve() != root / row["source_path"]
        ):
            raise ValueError("Development source/labels changed")
        selected.append({**record, **{k: row[k] for k in ("width", "height", "sha256")}})
    return {"tune": selected}
