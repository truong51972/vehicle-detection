"""Non-frozen review checks; synthetic pixels are not new study acquisition."""

import builtins
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vehicle_detection.common import file_hash, read_json
from vehicle_detection.v3_acceptance import REPORT, failed_gates, review, route_audit
from vehicle_detection.v3_adaptive import STUDY, local_contrast, read_report


def test_signed_pixel_bound_source_integrity_and_luminance_extremes():
    image = np.random.default_rng(53).integers(0, 256, (96, 128, 3), dtype=np.uint8)
    image[0, :4] = [(0, 0, 0), (0, 8, 0), (255, 247, 255), (255, 255, 255)]
    original = image.copy()
    output = local_contrast(image)
    difference = output.astype(np.int16) - image.astype(np.int16)
    assert output.dtype == np.uint8 and output.shape == image.shape
    assert np.array_equal(image, original)
    assert np.abs(difference).max() <= 12
    assert np.array_equal(output[0, :4], original[0, :4])


def test_saturated_channel_clipping_does_not_preserve_channel_differences():
    image = np.full((64, 64, 3), (0, 160, 255), dtype=np.uint8)
    output = local_contrast(image)
    difference = output.astype(np.int16) - image.astype(np.int16)
    assert np.all(difference[:, :, 0] > 0)
    assert np.array_equal(difference[:, :, 0], difference[:, :, 1])
    assert np.all(output[:, :, 2] == 255) and np.all(difference[:, :, 2] == 0)
    assert np.abs(difference).max() <= 12
    image = np.random.default_rng(47).integers(0, 256, (96, 128, 3), dtype=np.uint8)
    image[:, :, 0] = 0
    output = local_contrast(image)
    difference = output.astype(np.int16) - image.astype(np.int16)
    darkened = (difference[:, :, 1] < 0) & (difference[:, :, 2] < 0)
    assert darkened.any()
    assert np.all(output[:, :, 0][darkened] == 0)
    assert np.abs(difference).max() <= 12


def test_route_failure_counts_preserve_overlaps_and_the_frozen_strict_boundaries():
    safe = dict(contrast=0.1, brightness=0.2, noise=0, black_fraction=0, white_fraction=0)
    assert failed_gates(safe) == []
    for key, value, expected in (
        ("contrast", 0.2, "contrast_ge_0.20"),
        ("brightness", 0.04, "brightness_le_0.04"),
        ("brightness", 0.8, "brightness_ge_0.80"),
        ("noise", 0.010001, "noise_gt_0.01"),
        ("black_fraction", 0.3, "black_fraction_ge_0.30"),
        ("white_fraction", 0.1, "white_fraction_ge_0.10"),
    ):
        assert failed_gates({**safe, key: value}) == [expected]
    assert failed_gates({**safe, "noise": 0.01}) == []
    features = {"applied": safe, "skipped": {**safe, "contrast": 0.2, "white_fraction": 0.1}}
    stats = {
        "applied": {"target_cls": np.array([3, 4])},
        "skipped": {"target_cls": np.array([3, 3, 4])},
    }
    audit = route_audit(features, stats)
    motorcycle = audit["motorcycle"]
    assert (motorcycle["GT"], motorcycle["apply_GT"], motorcycle["skip_GT"]) == (3, 1, 2)
    assert sum(motorcycle["overlapping_failed_gate_GT"].values()) == 4
    assert sum(motorcycle["exclusive_failed_gate_combinations_GT"].values()) == 2
    with pytest.raises(ValueError, match="IDs differ"):
        route_audit(features, {"applied": stats["applied"]})


def test_historical_closure_deltas_keep_their_own_samples_and_objectives():
    family = read_json("docs/round2-preprocessing-v2/family_scores.json")
    scores = {(r["sample"], r["candidate"]): r["score"] for r in family}
    for sample, expected in (("exploration", 0.6736), ("confirmation", -0.4623)):
        assert 100 * (scores[sample, "denoise_5"] - scores[sample, "skip"]) == pytest.approx(
            expected, abs=0.00005
        )
    assert (
        read_json("docs/round2-preprocessing-v2/decision.json")["status"] == "confirmation_failed"
    )
    quality = read_json("docs/round2-quality-v2/summary.json")
    assert quality["status"] == "pending_fresh_confirmation" and not quality["v3_released"]
    slices = pd.DataFrame(read_json("docs/round2-research-v3/slices.json"))
    scores = (
        slices.groupby(["slice", "candidate", "family"])
        .mAP50_95.mean()
        .groupby(level=[0, 1])
        .mean()
    )
    for sample, expected in (("old", 0.59575), ("new", -0.01621)):
        assert 100 * (scores[sample, "pr38_locked"] - scores[sample, "frozen_v2"]) == pytest.approx(
            expected, abs=0.000005
        )
    shadow = read_json("docs/issue45/summary.json")["metrics"]
    assert 100 * (
        shadow["local_shadow"]["all"]["mAP50_95"] - shadow["raw"]["all"]["mAP50_95"]
    ) == pytest.approx(0.1145, abs=0.00005)
    trained = read_json("docs/issue45-s20/summary.json")["records"]
    assert 100 * (
        trained["v2"]["native"]["overall"]["mAP50_95"]
        - trained["v0"]["native"]["overall"]["mAP50_95"]
    ) == pytest.approx(-0.1194, abs=0.00005)


def test_saved_acceptance_report_binds_the_original_inputs_and_known_gate_coverage():
    audit = read_json(REPORT)
    for name, expected in audit["inputs"].items():
        assert file_hash(Path("docs") / STUDY / name) == expected
    assert file_hash("src/vehicle_detection/v3_acceptance.py") == audit["analysis_sha256"]
    assert audit["statistics"] == read_report(Path.cwd())["statistics"]
    for cls, gt, applied, skipped in (("motorcycle", 52, 17, 35), ("bicycle", 156, 26, 130)):
        record = audit["rare_class_routing"][cls]
        assert (record["GT"], record["apply_GT"], record["skip_GT"]) == (gt, applied, skipped)
        assert record["overlapping_failed_gate_GT"]["contrast_ge_0.20"] == skipped
        assert sum(record["exclusive_failed_gate_combinations_GT"].values()) == skipped
    assert all(
        audit[k] == 0
        for k in ("new_source_pixel_reads", "new_inference_passes", "new_bootstrap_draws")
    )


def test_local_saved_match_audit_cannot_open_pixels_weights_or_construct_models(monkeypatch):
    root = Path.cwd()
    result = read_report(root)
    allowed = {root / r["path"] for r in result["statistics"].values()}
    if not all(p.exists() for p in allowed):
        pytest.skip("Ignored native matching caches are available only on the acquisition host")
    ultralytics = pytest.importorskip("ultralytics")

    def reject(*args, **kwargs):
        raise AssertionError("Acceptance attempted source/model/inference access")

    for name in ("__init__", "train", "val"):
        monkeypatch.setattr(ultralytics.YOLO, name, reject)
    monkeypatch.setattr("torch.load", reject)
    monkeypatch.setattr("cv2.imread", reject)

    def guarded(original):
        def opening(path, mode="r", *args, **kwargs):
            if isinstance(path, (str, Path)):
                target = Path(path).absolute()
                if target.is_relative_to(root):
                    if any(c in mode for c in "wax+") or any(
                        target.is_relative_to(root / d) for d in ("data", "weights")
                    ):
                        reject()
                    if target.is_relative_to(root / "runs") and target not in allowed:
                        reject()
            return original(path, mode, *args, **kwargs)

        return opening

    monkeypatch.setattr(builtins, "open", guarded(builtins.open))
    monkeypatch.setattr(io, "open", guarded(io.open))
    assert review(root) == read_json(root / REPORT)
