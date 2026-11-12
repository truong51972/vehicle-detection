"""Contracts for the 13-cell Raw-vs-policy comparison (no production inference)."""

from pathlib import Path

import numpy as np
import pytest

from vehicle_detection.common import write_json
from vehicle_detection.comparison import (
    _cached,
    cell_name,
    comparison_pairs,
    contrasts,
)


def test_exact_thirteen_cells_and_no_policy_to_policy_mismatch():
    pairs = comparison_pairs()
    assert len(pairs) == len(set(pairs)) == 13
    assert set(pairs) == {
        ("v0", "v0"),
        ("v0", "v1"),
        ("v1", "v0"),
        ("v1", "v1"),
        ("v0", "v2"),
        ("v2", "v0"),
        ("v2", "v2"),
        ("v0", "v3"),
        ("v3", "v0"),
        ("v3", "v3"),
        ("v0", "v4"),
        ("v4", "v0"),
        ("v4", "v4"),
    }
    assert comparison_pairs(("v3",)) == [("v0", "v0"), ("v0", "v3"), ("v3", "v0"), ("v3", "v3")]
    with pytest.raises(ValueError, match="Unsupported"):
        cell_name("v1", "v2")
    with pytest.raises(ValueError, match="distinct"):
        comparison_pairs(("v1", "v1"))
    with pytest.raises(ValueError, match="distinct"):
        comparison_pairs(("v5",))


def test_factorial_contrast_definition_and_zero_baseline():
    values = contrasts(0.25, 0.28, 0.26, 0.32)
    assert values == pytest.approx(
        {
            "inference_on_raw_model": 0.03,
            "training_on_processed_input": 0.04,
            "training_on_raw_input": 0.01,
            "inference_on_policy_model": 0.06,
            "joint_pipeline": 0.07,
            "interaction": 0.03,
        }
    )
    assert all(abs(x) < 1e-12 for x in contrasts(0.2, 0.2, 0.2, 0.2).values())


def test_cache_identity_npz_membership_and_incomplete_are_fail_closed(tmp_path):
    folder = tmp_path / "runs/comparisons/v0__v1"
    expected = {"model_run": "v0-2026-11-09", "settings": {"imgsz": 640}}
    assert _cached(folder, expected, {"a", "b"}) == ("missing", None)
    folder.mkdir(parents=True)
    (folder / "native_plot.png").write_bytes(b"partial")
    assert _cached(folder, expected, {"a", "b"}) == ("incomplete", None)
    write_json(folder / "metrics.json", {"inputs": expected, "overall": {"mAP50_95": 0.25}})
    assert _cached(folder, expected, {"a", "b"}) == ("incomplete", None)
    np.savez_compressed(folder / "paired_stats.npz", ids=np.array(["a", "b"]))
    status, payload = _cached(folder, expected, {"a", "b"})
    assert status == "cached"
    assert payload["overall"]["mAP50_95"] == 0.25
    assert _cached(folder, {"changed": True}, {"a", "b"}) == ("stale", None)
    assert _cached(folder, expected, {"a", "c"}) == ("stale", None)
    np.savez_compressed(folder / "paired_stats.npz", ids=np.array(["a", "a"]))
    assert _cached(folder, expected, {"a", "b"}) == ("stale", None)


def test_raw_vs_policy_paths_do_not_modify_training_layout(tmp_path):
    from vehicle_detection.comparison import _output

    model = tmp_path / "runs/v0-2026-11-09"
    inputs = tmp_path / "runs/v3-2026-11-09"
    path = _output(Path(tmp_path), model, inputs)
    assert path == tmp_path / "runs/comparisons/v0-2026-11-09__v3-2026-11-09"
    assert path != model / "eval"
    assert path != inputs / "eval"


@pytest.mark.parametrize("policy", ["v1", "v4"])
def test_shared_draw_factorial_bootstrap_and_class_cis(policy):
    pytest.importorskip("ultralytics")
    from vehicle_detection.evaluation_stats import factorial_bootstrap

    rng = np.random.default_rng(2026)
    ids = [f"i{i}" for i in range(40)]
    groups = {sample: f"grp{i // 2}" for i, sample in enumerate(ids)}
    base = {}
    for sample in ids:
        tp = (rng.random((4, 10)) < 0.4).astype(bool)
        base[sample] = {
            "tp": tp,
            "conf": rng.random(4),
            "pred_cls": np.array([0, 1, 0, 1]),
            "target_cls": np.array([0, 1]),
        }
    arms = {key: base for key in ("v0/v0", f"v0/{policy}", f"{policy}/v0", f"{policy}/{policy}")}
    same = factorial_bootstrap(arms, groups, [0, 1], policy=policy, resamples=30)
    assert same["groups"] == 20
    for contrast in same["contrasts"].values():
        assert contrast["delta"] == pytest.approx(0)
        assert contrast["ci95"] == [0.0, 0.0]
        assert contrast["valid_draws"] == 30
    assert all(
        record["ci95"] == [0.0, 0.0]
        for by_contrast in same["per_class"].values()
        for record in by_contrast.values()
    )
    altered = {key: dict(base) for key in arms}
    altered[f"{policy}/{policy}"] = {sample: dict(record) for sample, record in base.items()}
    altered[f"{policy}/{policy}"]["i0"]["target_cls"] = np.array([1])
    with pytest.raises(ValueError, match="ground truth"):
        factorial_bootstrap(altered, groups, [0, 1], policy=policy, resamples=10)
