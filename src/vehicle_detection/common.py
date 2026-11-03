"""File utilities shared by the independent study notebooks."""

import hashlib
import json
import os
from pathlib import Path

import cv2


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def read_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError(f"Unreadable/empty image: {path}")
    return image


def reuse_evaluation(path, identity, *, force=False):
    """Read a completed native-val summary only when its inputs and saved plots still match."""
    path = Path(path)
    if force or not path.exists():
        return None
    cached = read_json(path)
    result = cached["result"]
    if cached["identity"] != identity or cached["checksum"] != digest(result):
        raise ValueError(
            "Evaluation cache identity/checksum changed; force reevaluation or use a new root"
        )
    for name, expected in result["artifacts"].items():
        if Path(name).name != name or not (path.parent / name).is_file():
            raise ValueError("Missing evaluation artifact; force reevaluation")
        if file_hash(path.parent / name) != expected:
            raise ValueError("Evaluation artifact changed; force reevaluation")
    return result


def freeze_json(path, value):
    """Publish immutable JSON once; repeated equal writes leave evidence untouched."""
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError("Immutable identity changed; use a new version/root")
    else:
        write_json(path, value)


def require_paths(**paths):
    """Fail before phase work when declared input artifacts are missing."""
    missing = [f"{name}: {path}" for name, path in paths.items() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required artifacts (prepare the owning phase): " + "; ".join(missing)
        )
