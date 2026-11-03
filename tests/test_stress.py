import copy
from pathlib import Path

import numpy as np
import pytest
from test_study import bdd as bdd
from test_study import recipes as recipes

from vehicle_detection import stress
from vehicle_detection.common import digest, file_hash, read_json, write_json
from vehicle_detection.data import scan_bdd
from vehicle_detection.protocol import build_protocol, summarize
from vehicle_detection.stress import (
    CORRUPTIONS,
    OPERATORS,
    converter_records,
    corrupt,
    diagnostics,
    export_stress,
    freeze,
    restore,
    select_calibration,
    validate_pilot,
    verify_stress,
)


@pytest.mark.parametrize("family", CORRUPTIONS)
@pytest.mark.parametrize("severity", [1, 2, 3])
def test_corruptions_reproducible_geometry_and_source_safe(family, severity):
    image = np.random.default_rng(0).integers(0, 256, (48, 64, 3), dtype=np.uint8)
    before = image.copy()
    result = corrupt(image, family, severity, "sample")
    np.testing.assert_array_equal(result, corrupt(image, family, severity, "sample"))
    np.testing.assert_array_equal(image, before)
    assert result.shape == image.shape and result.dtype == image.dtype
    assert not np.array_equal(result, image)
    if family == "shot_read":
        assert not np.array_equal(result, corrupt(image, family, severity, "another"))
    with pytest.raises(ValueError, match="severity"):
        corrupt(image, family, 0, "sample")


@pytest.mark.parametrize("operator", OPERATORS)
def test_operator_invariants_and_skip(operator, recipes):
    image = np.random.default_rng(1).integers(0, 256, (32, 40, 3), dtype=np.uint8)
    before = image.copy()
    output = restore(image, operator, recipes)
    assert output.shape == image.shape and output.dtype == image.dtype
    np.testing.assert_array_equal(image, before)
    np.testing.assert_array_equal(output, restore(image, operator, recipes))
    if operator == "skip":
        np.testing.assert_array_equal(output, image)
    np.testing.assert_array_equal(corrupt(image, "clean", 0, "sample"), image)


def test_synthetic_noise_and_blur_reference_sensitivity():
    flat = np.full((128, 160, 3), 100, np.uint8)
    noise = [
        diagnostics(corrupt(flat, "shot_read", s, "sample"))["noise_highpass"] for s in (1, 2, 3)
    ]
    assert 0 < noise[0] < noise[1] < noise[2]
    texture = np.random.default_rng(2).integers(0, 256, flat.shape, dtype=np.uint8)
    sharpness = [
        diagnostics(corrupt(texture, "defocus", s, "sample"))["sharpness"] for s in (1, 2, 3)
    ]
    assert sharpness[0] > sharpness[1] > sharpness[2]
    with pytest.raises(ValueError):
        restore(flat, "unknown", {})


@pytest.fixture
def pilot(bdd):
    records, _, sources = scan_bdd(bdd)
    protocol = build_protocol(records, sources, bdd, [])
    return protocol, select_calibration(protocol, image_count=2, attempts=4), records


def test_pilot_reproducible_group_safe_and_role_isolated(pilot, tmp_path):
    protocol, selected, _ = pilot
    assert selected == select_calibration(protocol, image_count=2, attempts=4)
    path = tmp_path / "pilot.json"
    freeze(path, selected)
    freeze(path, selected)
    changed = copy.deepcopy(selected)
    changed["rows"] = [protocol["splits"]["final_test"][0]]
    changed["summary"] = summarize(changed["rows"])
    changed["pilot_id"] = digest({k: v for k, v in changed.items() if k != "pilot_id"})
    with pytest.raises(ValueError, match="calibration groups"):
        validate_pilot(changed, protocol)
    with pytest.raises(ValueError, match="Immutable"):
        freeze(path, changed)


def test_pilot_keeps_cogrouped_frames_atomic(pilot):
    protocol, _, _ = pilot
    protocol = copy.deepcopy(protocol)
    template = protocol["splits"]["calibration"][0]
    protocol["splits"]["calibration"] = [{**template, "id": f"co-group-{i}"} for i in range(2)]
    protocol["summaries"]["calibration"] = summarize(protocol["splits"]["calibration"])
    protocol["manifest_id"] = digest({k: v for k, v in protocol.items() if k != "manifest_id"})
    selected = select_calibration(protocol, image_count=1, attempts=2)
    assert len(selected["rows"]) == 2
    selected["rows"].pop()
    selected["summary"] = summarize(selected["rows"])
    selected["pilot_id"] = digest({k: v for k, v in selected.items() if k != "pilot_id"})
    with pytest.raises(ValueError, match="complete frozen calibration groups"):
        validate_pilot(selected, protocol)


def test_exports_paired_reuse_no_holdout_or_stale_identity(
    pilot, bdd, recipes, tmp_path, monkeypatch
):
    protocol, selected, records = pilot
    original_read = stress.read_image
    allowed_sources = {str(bdd / r["source_path"]) for r in selected["rows"]}

    def safe_read(path):
        if str(path).startswith(str(bdd)):
            assert str(path) in allowed_sources
        assert "val" not in Path(path).parts
        return original_read(path)

    monkeypatch.setattr(stress, "read_image", safe_read)
    partitions = converter_records(selected, protocol, records, bdd)
    converted = tmp_path / "conversion"
    write_json(converted / "conversion.json", {"fixture": True})
    for row in partitions["tune"]:
        p = converted / "labels/tune" / f"{row['id']}.txt"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("0 0.5 0.5 0.2 0.2\n")
    v2 = {"policy_id": "fixture", "recipes": recipes}
    raw = tmp_path / "raw"
    restored = tmp_path / "restored"
    one = export_stress(protocol, selected, bdd, converted, raw, "shot_read", 2, "skip", v2)
    two = export_stress(
        protocol, selected, bdd, converted, restored, "shot_read", 2, "bilateral", v2
    )
    assert one.id.tolist() == two.id.tolist()
    assert one.label_sha256.tolist() == two.label_sha256.tolist()
    hashes = [file_hash(p) for p in one.image]
    export_stress(protocol, selected, bdd, converted, raw, "shot_read", 2, "skip", v2)
    assert hashes == [file_hash(p) for p in one.image]
    assert not list((raw / "images/test").iterdir())
    with pytest.raises(ValueError, match="Immutable"):
        export_stress(protocol, selected, bdd, converted, raw, "shot_read", 3, "skip", v2)
    changed_policy = copy.deepcopy(v2)
    changed_policy["recipes"]["gamma"]["gamma"] = 0.5
    with pytest.raises(ValueError, match="Immutable"):
        export_stress(
            protocol, selected, bdd, converted, raw, "shot_read", 2, "skip", changed_policy
        )
    target = Path(one.iloc[0].image)
    target.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        verify_stress(selected, raw)
    source = bdd / selected["rows"][0]["source_path"]
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="source changed"):
        export_stress(protocol, selected, bdd, converted, raw, "shot_read", 2, "skip", v2)
    assert read_json(raw / "export.json")["role"] == "calibration"


def test_native_statistics_pairing_and_bootstrap(pilot):
    pytest.importorskip("ultralytics")
    from vehicle_detection.stress_eval import native_ap, paired_interval

    _, selected, _ = pilot
    before = {
        r["id"]: {
            "tp": np.ones((1, 10), bool),
            "conf": np.array([0.9]),
            "pred_cls": np.array([0]),
            "target_cls": np.array([0]),
        }
        for r in selected["rows"]
    }
    assert native_ap(before, list(before)) > 0.9
    interval = paired_interval(before, before, selected, resamples=5)
    assert interval["recovery_ci_low"] == interval["recovery_ci_high"] == 0
    with pytest.raises(ValueError, match="identical"):
        paired_interval(before, {}, selected)
    changed = copy.deepcopy(before)
    changed[next(iter(changed))]["target_cls"] = np.array([1])
    with pytest.raises(ValueError, match="ground truth"):
        paired_interval(before, changed, selected)
