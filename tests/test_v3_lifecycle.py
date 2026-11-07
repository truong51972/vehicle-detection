"""Verify the stopped lifecycle's compact evidence against its complete archives."""

import gzip
import json
from pathlib import Path

import pandas as pd

from vehicle_detection.common import digest, file_hash, read_json
from vehicle_detection.v3_adaptive import STUDY, assess, read_report


def test_saved_native_metrics_intervals_and_scientific_stop_agree():
    root = Path.cwd()
    report = root / f"docs/{STUDY}"
    result = read_report(root)
    assert not result["decision"]["advance"]
    assert result["decision"] == assess(result["metrics"], result["runtime"], result["intervals"])
    assert not any(
        result[k]
        for k in (
            "policy_frozen",
            "v3_trained",
            "confirmation_accessed",
            "development_accessed",
            "final_test_accessed",
        )
    )
    assert not (root / "configs/v3.yaml").exists()
    assert not (report / "confirmation").exists()
    for arm, metrics in result["metrics"].items():
        assert metrics == read_json(report / f"{arm}_metrics.json")
        native = result["measurements"][arm]["native"]
        assert abs(native["mAP50_95"] - metrics["all"]["mAP50_95"]) < 1e-8
        assert abs(native["mAP50"] - metrics["all"]["mAP50"]) < 1e-8
    for name, summary in result["intervals"].items():
        full = json.loads(gzip.decompress((report / f"{name}_interval.json.gz").read_bytes()))
        draws = full.pop("draws")
        assert full == summary and len(draws) == summary["resamples"] == 1000
        assert sum(d["valid"] for d in draws) == summary["valid_draws"]


def test_complete_read_and_route_archives_match_the_frozen_original_population():
    report = Path("docs") / STUDY
    rows = pd.read_csv(report / "membership.csv.gz", keep_default_na=False).to_dict("records")
    result = read_report(Path.cwd())
    features = json.loads(gzip.decompress((report / "features.json.gz").read_bytes()))
    reads = json.loads(gzip.decompress((report / "source_reads.json.gz").read_bytes()))
    assert set(features) == {r["id"] for r in rows}
    assert reads == sorted(r["source_path"] for r in rows)
    assert result["source_reads"] == {"count": len(rows), "paths_sha256": digest(reads)}
    assert sum(result["routing"].values()) == len(rows) == 1479
    assert result["inference_image_passes"] == 3 * len(rows)
    receipt = read_json(report / "preflight_validation.json")
    assert (
        file_hash(receipt["historical_notebook_archive"]) == receipt["historical_notebook_sha256"]
    )
