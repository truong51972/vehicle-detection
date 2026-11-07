"""Verify Issue #51's packaged evidence without models, pixels or resampling."""

import gzip
import hashlib
import json
from pathlib import Path

from .common import file_hash, read_json
from .full_resolution_training import STUDY, read_report


def verify(root):
    root = Path(root).resolve()
    report = root / f"docs/{STUDY}"
    summary = read_report(root)
    packaging = read_json(report / "packaging.json")
    validation_path = root / "docs/issue51/execution_validation.json"
    acquired_path = report / "evidence.acquired.json"
    acquired_hash = file_hash(acquired_path)
    if (
        acquired_hash != packaging["original_evidence_sha256"]
        or acquired_hash != read_json(validation_path)["evidence_index_sha256"]
        or file_hash(validation_path) != packaging["original_validation_sha256"]
    ):
        raise ValueError("Original validation/manifest binding changed")
    acquired = read_json(acquired_path)
    if set(packaging["archives"]) != {"result.json", "membership.csv"}:
        raise ValueError("Unexpected packaged evidence files")
    originals = {}
    for name, expected in acquired.items():
        if Path(name).name != name:
            raise ValueError("Unexpected original evidence path")
        if name in packaging["archives"]:
            path = report / (name + ".gz")
            record = packaging["archives"][name]
            raw = gzip.decompress(path.read_bytes())
            observed = hashlib.sha256(raw).hexdigest()
            if (
                file_hash(path) != record["gzip_sha256"]
                or observed != record["uncompressed_sha256"]
            ):
                raise ValueError("Archive hashes changed")
            originals[name] = raw
        else:
            observed = file_hash(report / name)
        if observed != expected:
            raise ValueError(f"Original evidence changed: {name}")
    original = json.loads(originals["result.json"])
    for interval in [original["paired_interval"], *original["cohort_intervals"].values()]:
        del interval["draws"]
    if summary != original:
        raise ValueError("Summary differs beyond removal of draw arrays")
    return summary


if __name__ == "__main__":
    verify(Path.cwd())
    print("Original hashes, lossless archives and exact summary equivalence verified.")
