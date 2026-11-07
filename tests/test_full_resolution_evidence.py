"""Packaging retains the exact original evidence, even after current hashes are refreshed."""

import copy
import gzip
import hashlib
import json
import shutil
import sys

import pytest

from vehicle_detection.common import file_hash, read_json, write_json
from vehicle_detection.full_resolution_evidence import verify
from vehicle_detection.full_resolution_training import MODULE, NOTEBOOK, PLAN, RECEIPT, STUDY


@pytest.fixture
def packaged(tmp_path):
    for name in (MODULE, NOTEBOOK, PLAN, RECEIPT, "docs/issue51/execution_validation.json"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, path)
    shutil.copytree(f"docs/{STUDY}", tmp_path / f"docs/{STUDY}")
    return tmp_path


def test_packaged_summary_round_trips_to_exact_original_bytes_without_study_modules(
    packaged, monkeypatch
):
    for name in ("torch", "ultralytics", "vehicle_detection.evaluation_stats"):
        monkeypatch.setitem(sys.modules, name, None)
    summary = verify(packaged)
    report = packaged / f"docs/{STUDY}"
    acquired = read_json(report / "evidence.acquired.json")
    raw = gzip.decompress((report / "result.json.gz").read_bytes())
    original = json.loads(raw)
    restored = copy.deepcopy(summary)
    restored["paired_interval"]["draws"] = original["paired_interval"]["draws"]
    for name, interval in original["cohort_intervals"].items():
        restored["cohort_intervals"][name]["draws"] = interval["draws"]
    assert (json.dumps(restored, indent=2, sort_keys=True, allow_nan=False) + "\n").encode() == raw
    for name in ("result.json", "membership.csv"):
        raw = gzip.decompress((report / (name + ".gz")).read_bytes())
        assert hashlib.sha256(raw).hexdigest() == acquired[name]


@pytest.mark.parametrize(
    ("changed", "message"),
    [
        ("summary", "Summary differs"),
        ("result.json", "Original evidence changed"),
        ("membership.csv", "Original evidence changed"),
        ("manifest", "Original validation/manifest binding changed"),
    ],
)
def test_refreshed_packaging_hashes_cannot_hide_changed_evidence(packaged, changed, message):
    report = packaged / f"docs/{STUDY}"
    packaging = read_json(report / "packaging.json")
    if changed == "summary":
        result = read_json(report / "result.json")
        result["metrics"]["960/960"]["all"]["mAP50_95"] += 0.001
        write_json(report / "result.json", result)
    elif changed == "manifest":
        acquired = read_json(report / "evidence.acquired.json")
        acquired["result.json"] = "0" * 64
        write_json(report / "evidence.acquired.json", acquired)
        packaging["original_evidence_sha256"] = file_hash(report / "evidence.acquired.json")
    else:
        path = report / (changed + ".gz")
        raw = gzip.decompress(path.read_bytes())
        if changed == "result.json":
            original = json.loads(raw)
            original["paired_interval"]["draws"][0]["recovery"] += 0.001
            raw = json.dumps(original).encode()
        else:
            raw += b"\n"
        path.write_bytes(gzip.compress(raw, mtime=0))
        packaging["archives"][changed] = {
            "gzip_sha256": file_hash(path),
            "uncompressed_sha256": hashlib.sha256(raw).hexdigest(),
        }
    write_json(report / "packaging.json", packaging)
    write_json(
        report / "evidence.json",
        {p.name: file_hash(p) for p in report.iterdir() if p.name != "evidence.json"},
    )
    with pytest.raises(ValueError, match=message):
        verify(packaged)
