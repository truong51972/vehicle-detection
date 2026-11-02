"""File utilities for the two study notebooks."""

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
