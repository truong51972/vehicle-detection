"""Small file/provenance utilities shared by commands."""

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
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


def environment():
    versions = {}
    for name in (
        "vehicle-detection",
        "numpy",
        "opencv-python",
        "ijson",
        "PyYAML",
        "matplotlib",
        "ultralytics",
        "torch",
        "torchvision",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    source = Path(__file__).parent
    processor = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if not processor and cpuinfo.is_file():
        processor = next(
            (
                line.split(":", 1)[1].strip()
                for line in cpuinfo.read_text().splitlines()
                if line.startswith("model name")
            ),
            "unknown",
        )
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": processor,
        "cpu_count": os.cpu_count(),
        "opencv_version": cv2.__version__,
        "opencv_threads": cv2.getNumThreads(),
        "versions": versions,
        "source_hash": digest(
            {str(p.relative_to(source)): file_hash(p) for p in sorted(source.rglob("*.py"))}
        ),
    }
