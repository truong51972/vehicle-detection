"""Issue #11: calibration-only, geometry-preserving synthetic robustness experiments."""

import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from .common import digest, file_hash, read_image, read_json, write_json
from .data import CLASSES
from .offline import encoding_policy, verify_alignment
from .preprocess import apply_recipe, clahe, denoise, gamma, quality, sharpen, validate_image
from .protocol import summarize, validate_protocol
from .subsets import subset_features

# These are controlled stress approximations, not calibrated BDD camera physics.
CORRUPTIONS = {
    "shot_read": [
        {"photons": p, "read_sigma": s} for p, s in [(800, 0.003), (200, 0.008), (50, 0.02)]
    ],
    "motion": [{"size": s} for s in (5, 11, 21)],
    "defocus": [{"radius": r} for r in (1, 2, 4)],
    "underexposure": [{"factor": f} for f in (0.7, 0.45, 0.25)],
    "overexposure": [{"factor": f} for f in (1.3, 1.7, 2.3)],
    "veil": [{"alpha": a} for a in (0.1, 0.25, 0.4)],
}
OPERATORS = {
    "skip": {},
    "gamma": {"gamma": 0.7},
    "darken": {"gamma": 1.3},
    "clahe": {"clip_limit": 1.5, "grid_size": 8},
    "nlmeans": {"strength": 5.0},
    "bilateral": {"d": 5, "sigmaColor": 25.0, "sigmaSpace": 3.0},
    "unsharp": {"amount": 0.4, "sigma": 1.0},
    "percentile": {"low": 1.0, "high": 99.0},
    "adaptive_v2": {},
}
SCREEN = {
    "shot_read": ["nlmeans", "bilateral"],
    "motion": ["unsharp", "clahe"],
    "defocus": ["unsharp", "clahe"],
    "underexposure": ["gamma", "clahe"],
    "overexposure": ["darken", "percentile"],
    "veil": ["clahe", "percentile"],
}


def corrupt(image, family, severity, sample_id, seed=2026):
    validate_image(image)
    if family == "clean":
        if severity != 0:
            raise ValueError("Clean requires severity zero")
        return image.copy()
    if family not in CORRUPTIONS or type(severity) is not int or severity not in (1, 2, 3):
        raise ValueError("Unknown corruption/severity")
    params = CORRUPTIONS[family][severity - 1]
    rng = np.random.default_rng(int(digest([seed, sample_id, family, severity])[:16], 16))
    x = image.astype(np.float32) / 255
    if family == "shot_read":
        linear = x**2.2
        noisy = rng.poisson(linear * params["photons"]) / params["photons"]
        noisy += rng.normal(0, params["read_sigma"], image.shape)
        result = np.clip(noisy, 0, 1) ** (1 / 2.2)
    elif family == "motion":
        size = params["size"]
        kernel = np.zeros((size, size), np.float32)
        # Per-ID orientation is held constant across severities.
        angle = int(digest([seed, sample_id, "motion-angle"])[:8], 16) % 4
        if angle == 0:
            kernel[size // 2, :] = 1
        elif angle == 1:
            kernel[:, size // 2] = 1
        elif angle == 2:
            np.fill_diagonal(kernel, 1)
        else:
            np.fill_diagonal(kernel[:, ::-1], 1)
        result = cv2.filter2D(x, -1, kernel / kernel.sum(), borderType=cv2.BORDER_REFLECT_101)
    elif family == "defocus":
        radius = params["radius"]
        y, z = np.ogrid[-radius : radius + 1, -radius : radius + 1]
        kernel = (y * y + z * z <= radius * radius).astype(np.float32)
        result = cv2.filter2D(x, -1, kernel / kernel.sum(), borderType=cv2.BORDER_REFLECT_101)
    elif family in {"underexposure", "overexposure"}:
        result = np.clip(x**2.2 * params["factor"], 0, 1) ** (1 / 2.2)
    else:
        result = x * (1 - params["alpha"]) + 0.9 * params["alpha"]
    return np.rint(np.clip(result, 0, 1) * 255).astype(np.uint8)


def restore(image, operator, v2_recipes):
    validate_image(image)
    if operator not in OPERATORS:
        raise ValueError("Unknown restoration operator")
    params = OPERATORS[operator]
    if operator == "skip":
        return image.copy()
    if operator in {"gamma", "darken"}:
        return gamma(image, **params)
    if operator == "clahe":
        return clahe(image, **params)
    if operator == "nlmeans":
        return denoise(image, **params)
    if operator == "bilateral":
        return cv2.bilateralFilter(image, **params)
    if operator == "unsharp":
        return sharpen(image, **params)
    if operator == "adaptive_v2":
        return apply_recipe(image, "adaptive", v2_recipes)[0]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    low, high = np.percentile(lab[:, :, 0], [params["low"], params["high"]])
    if high <= low:
        return image.copy()
    lab[:, :, 0] = np.clip((lab[:, :, 0].astype(float) - low) * 255 / (high - low), 0, 255)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def diagnostics(image):
    scores = quality(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    response = cv2.filter2D(gray, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
    scores["noise_highpass"] = float(np.abs(response[1:-1, 1:-1]).mean() / 6 * np.sqrt(np.pi / 2))
    return scores


def freeze(path, value):
    path = Path(path)
    if path.exists() and read_json(path) != value:
        raise ValueError("Immutable stress identity changed; use a new version/root")
    if not path.exists():
        write_json(path, value)


def select_calibration(protocol, *, image_count=256, seed=2026, attempts=64):
    validate_protocol(protocol)
    if any(type(v) is not int or v < 1 for v in (image_count, attempts)):
        raise ValueError("Pilot size/attempts must be positive integers")
    pool = protocol["splits"]["calibration"]
    groups = defaultdict(list)
    for row in sorted(pool, key=lambda r: r["id"]):
        groups[row["group"]].append(row)
    keys = sorted(groups)
    counts = [sum((subset_features(r) for r in groups[k]), Counter()) for k in keys]
    totals = sum(counts, Counter())
    columns = sorted(totals)
    matrix = np.array([[c.get(k, 0) for k in columns] for c in counts], dtype=float)
    sizes = np.array([len(groups[k]) for k in keys])
    rng, best = random.Random(seed), None
    for _ in range(attempts):
        order = list(range(len(keys)))
        rng.shuffle(order)
        cumulative = np.cumsum(sizes[order])
        cut = int(np.argmin(abs(cumulative - min(image_count, len(pool)))) + 1)
        observed = matrix[order[:cut]].sum(axis=0)
        expected = matrix.sum(axis=0) * cumulative[cut - 1] / len(pool)
        score = float(np.mean((observed - expected) ** 2 / (expected + 1)))
        score += int(np.sum((observed == 0) & (expected >= 2)))
        if best is None or score < best[0]:
            best = score, [keys[i] for i in order[:cut]]
    selected = sorted([r for k in best[1] for r in groups[k]], key=lambda r: r["id"])
    pilot = {
        "version": "issue11-pilot-v1",
        "protocol_id": protocol["manifest_id"],
        "role": "calibration",
        "seed": seed,
        "target_images": image_count,
        "attempts": attempts,
        "score": best[0],
        "rows": selected,
        "summary": summarize(selected),
        "calibration_summary": summarize(pool),
        "selection_code": file_hash(__file__),
    }
    pilot["pilot_id"] = digest(pilot)
    return validate_pilot(pilot, protocol)


def validate_pilot(pilot, protocol):
    validate_protocol(protocol)
    if pilot["pilot_id"] != digest({k: v for k, v in pilot.items() if k != "pilot_id"}):
        raise ValueError("Pilot checksum changed")
    pool = {r["id"]: r for r in protocol["splits"]["calibration"]}
    rows = pilot["rows"]
    groups = {r["group"] for r in rows}
    if (
        pilot["protocol_id"] != protocol["manifest_id"]
        or pilot["role"] != "calibration"
        or not rows
        or len({r["id"] for r in rows}) != len(rows)
        or any(r != pool.get(r["id"]) for r in rows)
        or {r["id"] for r in rows} != {r["id"] for r in pool.values() if r["group"] in groups}
        or summarize(rows) != pilot["summary"]
    ):
        raise ValueError("Pilot must preserve complete frozen calibration groups/annotations")
    return pilot


def converter_records(pilot, protocol, records, root):
    validate_pilot(pilot, protocol)
    lookup, selected = {r["id"]: r for r in records}, []
    for row in pilot["rows"]:
        record = lookup[row["id"]]
        if (
            record["official_split"] != "train"
            or digest(record["labels"]) != row["labels_sha256"]
            or Path(record["path"]).resolve() != Path(root).resolve() / row["source_path"]
        ):
            raise ValueError("Calibration source/labels changed")
        selected.append({**record, **{k: row[k] for k in ("width", "height", "sha256")}})
    return {"tune": selected}


def export_stress(protocol, pilot, root, converted, destination, family, severity, operator, v2):
    validate_pilot(pilot, protocol)
    if family not in {"clean", *CORRUPTIONS} or operator not in OPERATORS:
        raise ValueError("Unknown experiment treatment")
    # Validate before any source image access, including on cache reuse.
    corrupt(np.zeros((4, 4, 3), np.uint8), family, severity, "validation", pilot["seed"])
    root, converted, destination = (
        Path(root).resolve(),
        Path(converted).resolve(),
        Path(destination).resolve(),
    )
    if destination == root or root in destination.parents or destination in root.parents:
        raise ValueError("Stress export must be outside source data")
    for row in pilot["rows"]:
        if file_hash(root / row["source_path"]) != row["sha256"]:
            raise ValueError("Calibration source changed")
    labels = {
        r["id"]: file_hash(converted / "labels/tune" / f"{r['id']}.txt") for r in pilot["rows"]
    }
    identity = {
        "version": "issue11-export-v1",
        "pilot_id": pilot["pilot_id"],
        "protocol_id": protocol["manifest_id"],
        "role": "calibration",
        "family": family,
        "severity": severity,
        "operator": operator,
        "corruption": {} if family == "clean" else CORRUPTIONS[family][severity - 1],
        "operator_params": OPERATORS[operator],
        "v2_policy_id": v2["policy_id"],
        "v2_recipes": v2["recipes"],
        "sources": digest(
            [(r["id"], r["group"], r["sha256"], labels[r["id"]]) for r in pilot["rows"]]
        ),
        "encoding": encoding_policy("jpg", 95),
        "numpy": np.__version__,
        "code_sha256": file_hash(__file__),
        "operator_code": file_hash(Path(__file__).with_name("preprocess.py")),
        "conversion": read_json(converted / "conversion.json"),
    }
    if (
        not (destination / "export.json").exists()
        and destination.exists()
        and any(destination.iterdir())
    ):
        raise ValueError("Unknown stress export directory")
    freeze(destination / "export.json", identity)
    for split in ("train", "tune", "test"):
        for kind in ("images", "labels"):
            folder = destination / kind / split
            if folder.resolve() != folder:
                raise ValueError("Unexpected stress directory symlink")
            folder.mkdir(parents=True, exist_ok=True)
    if (destination / "metadata.csv").exists():
        verify_stress(pilot, destination)
        return pd.read_csv(destination / "metadata.csv")
    metadata = []
    for row in pilot["rows"]:
        source = root / row["source_path"]
        if file_hash(source) != row["sha256"]:
            raise ValueError("Calibration source changed")
        started = time.perf_counter()
        clean = read_image(source)
        if clean.shape != (row["height"], row["width"], 3):
            raise ValueError("Source geometry changed")
        tick = time.perf_counter()
        damaged = corrupt(clean, family, severity, row["id"], pilot["seed"])
        corruption_ms = (time.perf_counter() - tick) * 1000
        tick = time.perf_counter()
        output = restore(damaged, operator, v2["recipes"])
        restore_ms = (time.perf_counter() - tick) * 1000
        if output.shape != clean.shape or output.dtype != clean.dtype:
            raise ValueError("Stress transformation changed geometry/dtype")
        target = destination / "images/tune" / f"{row['id']}.jpg"
        label = destination / "labels/tune" / f"{row['id']}.txt"
        if target.is_symlink() or label.is_symlink():
            raise ValueError("Unexpected stress output symlink")
        temporary = target.with_suffix(".tmp.jpg")
        if temporary.is_symlink():
            raise ValueError("Unexpected temporary symlink")
        if not cv2.imwrite(str(temporary), output, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError("Stress JPEG write failed")
        temporary.replace(target)
        label.write_bytes((converted / "labels/tune" / label.name).read_bytes())
        metadata.append(
            {
                "id": row["id"],
                "split": "tune",
                "role": "calibration",
                "group": row["group"],
                "image": str(target),
                "source_sha256": row["sha256"],
                "width": row["width"],
                "height": row["height"],
                "output_sha256": file_hash(target),
                "label_sha256": labels[row["id"]],
                "corruption_ms": corruption_ms,
                "restore_ms": restore_ms,
                "preparation_ms": (time.perf_counter() - started) * 1000,
                **diagnostics(read_image(target)),
            }
        )
    (destination / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(destination),
                "train": "images/train",
                "val": "images/tune",
                "test": "images/test",
                "names": dict(enumerate(CLASSES)),
            }
        )
    )
    table = pd.DataFrame(metadata)
    table.to_csv(destination / "metadata.csv", index=False)
    verify_stress(pilot, destination)
    return table


def verify_stress(pilot, destination):
    destination = Path(destination).resolve()
    meta = pd.read_csv(destination / "metadata.csv")
    expected = {r["id"]: r for r in pilot["rows"]}
    export = read_json(destination / "export.json")
    if (
        export["pilot_id"] != pilot["pilot_id"]
        or export["role"] != "calibration"
        or set(meta.id) != set(expected)
        or len(meta) != len(expected)
        or set(meta.split) != {"tune"}
        or set(meta.role) != {"calibration"}
    ):
        raise ValueError("Stress export must retain exact calibration IDs/role")
    labels = {}
    for row in meta.to_dict("records"):
        source = expected[row["id"]]
        image = Path(row["image"])
        if (
            image != destination / "images/tune" / f"{row['id']}.jpg"
            or image.is_symlink()
            or image.parent.resolve() != image.parent
            or row["group"] != source["group"]
            or row["source_sha256"] != source["sha256"]
            or (row["width"], row["height"]) != (source["width"], source["height"])
        ):
            raise ValueError("Stress source/group/geometry changed")
        labels[row["id"]] = row["label_sha256"]
    if export["sources"] != digest(
        [(r["id"], r["group"], r["sha256"], labels[r["id"]]) for r in pilot["rows"]]
    ):
        raise ValueError("Stress annotation identity changed")
    count = verify_alignment([destination])
    if count != len(expected):
        raise ValueError("Stress verified image count mismatch")
    return meta
