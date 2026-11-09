"""Route-report safeguards and saved evidence consistency; no detector acquisition."""

import shutil
from pathlib import Path

import numpy as np
import pytest

from vehicle_detection.common import file_hash, write_json
from vehicle_detection.five_class import FIVE_CLASSES
from vehicle_detection.posthoc_routes import (
    REPORT,
    cohort_record,
    differences,
    partition,
    read_report,
)
from vehicle_detection.posthoc_training import STUDY
from vehicle_detection.posthoc_training import read_report as original_report


def test_partition_requires_unique_complete_frozen_boolean_routes():
    rows = [{"id": "a"}, {"id": "b"}]
    records = [
        ["val", "a", "raw", "adaptive", "label", True],
        ["val", "b", "raw", "raw", "label", False],
    ]
    assert partition(rows, records) == {"apply": [rows[0]], "skip": [rows[1]]}
    for changed in (records[:1], records + [records[0]], [records[0][:-1] + [1], records[1]]):
        with pytest.raises(ValueError):
            partition(rows, changed)


def test_missing_classes_are_not_zero_imputed_and_weak_support_stays_labelled():
    pytest.importorskip("ultralytics")
    row = {"id": "one", "source_group": "source", "group": "group", "boxes": [[4, 0, 0, 1, 1]]}
    stats = {
        "one": {
            "target_cls": np.array([4]),
            "pred_cls": np.array([4]),
            "conf": np.array([0.9]),
            "tp": np.ones((1, 10), dtype=bool),
        }
    }
    record = cohort_record(stats, [row])
    assert record["mAP50_95"] is None and record["mAP50"] is None
    assert record["classes"]["car"]["AP50_95"] is None
    assert record["classes"]["car"]["support"] == "missing"
    bicycle = record["classes"]["bicycle"]
    assert bicycle["AP50_95"] > 0 and bicycle["instances"] == bicycle["source_groups"] == 1
    assert bicycle["support"] == "weak (<5 source groups)"
    contrast = differences(record, record)
    assert contrast["mAP50_95"] is None
    assert contrast["classes"]["car"]["AP50_95"] is None
    assert contrast["classes"]["bicycle"]["AP50_95"] == 0


def test_saved_routes_partition_original_gt_and_contrasts_without_averaging_ap():
    root = Path(__file__).resolve().parents[1]
    result, original = read_report(root), original_report(root)
    assert result["original_evidence_index_sha256"] == file_hash(
        root / f"docs/{STUDY}/evidence.json"
    )
    assert {k: v["images"] for k, v in result["cohorts"].items()} == {"apply": 5653, "skip": 8322}
    assert result["statistics"] == original["statistics"]
    assert result["prepared_receipt_sha256"] == original["data"]["prepared_sha256"]
    assert result["inputs"]["src/vehicle_detection/posthoc_routes.py"] == file_hash(
        root / "src/vehicle_detection/posthoc_routes.py"
    )
    for key in ("new_training_runs", "new_inference_image_passes", "new_source_pixel_reads"):
        assert result[key] == 0
    for cell, records in result["metrics"].items():
        for cls in FIVE_CLASSES:
            for count in ("GT", "TP50", "FP50", "FN50"):
                assert (
                    sum(v["counts"][cls][count] for v in records.values())
                    == original["metrics"][cell]["all"]["counts"][cls][count]
                )
        for cohort, record in records.items():
            assert not record["missing_classes"]
            for metric, class_metric in (("mAP50_95", "AP50_95"), ("mAP50", "AP50")):
                assert record[metric] == pytest.approx(
                    np.mean([c[class_metric] for c in record["classes"].values()])
                )
            for cls, support in record["classes"].items():
                baseline = result["metrics"]["raw/raw"][cohort]["classes"][cls]
                assert support["instances"] == baseline["instances"] == record["counts"][cls]["GT"]
                assert support["source_groups"] == baseline["source_groups"]
                assert support["images"] == baseline["images"]
                assert support["support"] == "≥5 source groups"
            for label, before in (("D-B", "raw/adaptive"), ("D-A", "raw/raw")):
                assert result["contrasts"][cohort][label] == differences(
                    result["metrics"][before][cohort],
                    result["metrics"]["adaptive/adaptive"][cohort],
                )
    # Identical Skip bytes plus a fixed checkpoint must preserve the saved AP/counts.
    for a, b in (("raw/raw", "raw/adaptive"), ("adaptive/raw", "adaptive/adaptive")):
        assert result["metrics"][a]["skip"] == result["metrics"][b]["skip"]


def test_route_report_detects_modified_compact_evidence(tmp_path):
    report = tmp_path / REPORT
    write_json(report / "result.json", {"scope": "synthetic"})
    write_json(report / "evidence.json", {"result.json": file_hash(report / "result.json")})
    assert read_report(tmp_path)["scope"] == "synthetic"
    write_json(report / "result.json", {})
    with pytest.raises(ValueError, match="Route evidence changed"):
        read_report(tmp_path)


def test_complete_comparison_with_routes_is_read_only_in_a_fresh_kernel(tmp_path):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").touch()
    shutil.copytree("docs", root / "docs")
    before = {str(p.relative_to(root)): file_hash(p) for p in root.rglob("*") if p.is_file()}
    notebook = nbformat.read("notebooks/experiments/compare.ipynb", as_version=4)
    notebook.cells.insert(
        1,
        nbformat.v4.new_code_cell("""import builtins
import io
import sys
from pathlib import Path
from unittest.mock import patch
sys.modules["torch"] = None
sys.modules["ultralytics"] = None
sys.modules["vehicle_detection.evaluation_stats"] = None
patch("vehicle_detection.posthoc_routes.build",
      side_effect=AssertionError("Analysis replay")).start()
_root = Path.cwd()
def guarded(original):
    def wrapped(path, mode="r", *args, **kwargs):
        if isinstance(path, (str, Path)):
            path = Path(path).absolute()
            if path.is_relative_to(_root):
                if any(c in mode for c in "wax+") or not path.is_relative_to(_root / "docs"):
                    raise AssertionError("Comparison accessed execution state or wrote files")
        return original(path, mode, *args, **kwargs)
    return wrapped
patch("builtins.open", guarded(builtins.open)).start()
patch("io.open", guarded(io.open)).start()
"""),
    )
    nbclient.NotebookClient(
        notebook, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(root)}}
    ).execute()
    assert all(c.execution_count is not None for c in notebook.cells if c.cell_type == "code")
    assert before == {
        str(p.relative_to(root)): file_hash(p) for p in root.rglob("*") if p.is_file()
    }
