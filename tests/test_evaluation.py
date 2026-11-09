import numpy as np
import pandas as pd
import pytest

pytest.importorskip("ultralytics")

from vehicle_detection.common import write_json  # noqa: E402
from vehicle_detection.evaluation import cohort_metrics, saved_result  # noqa: E402
from vehicle_detection.evaluation_stats import class_ap, paired_bootstrap  # noqa: E402


def synthetic(ids, *, quality, seed):
    rng = np.random.default_rng(seed)
    stats = {}
    for i, sample_id in enumerate(ids):
        targets = np.array([i % 2, 1 - i % 2, 0], dtype=float)
        hits = rng.random((4, 10)) < quality
        stats[sample_id] = {
            "tp": hits,
            "conf": rng.random(4),
            "pred_cls": np.array([0, 1, 0, 1], dtype=float),
            "target_cls": targets,
        }
    return stats


def test_cohorts_cover_dark_bright_and_routes():
    ids = [f"i{i}" for i in range(12)]
    stats = synthetic(ids, quality=0.6, seed=0)
    metadata = pd.DataFrame(
        {
            "id": ids + ["train_only"],
            "split": ["test"] * 12 + ["train"],
            "brightness": [0.05] * 4 + [0.5] * 8 + [0.01],
            "applied": ["gamma"] * 4 + [""] * 9,
        }
    )
    cohorts = cohort_metrics(stats, metadata)
    assert {k: v["images"] for k, v in cohorts.items()} == {
        "dark": 4,
        "bright": 8,
        "apply": 4,
        "skip": 8,
    }
    assert set(cohorts["bright"]["classes"]) == {"car", "truck"}
    unrouted = metadata.assign(applied="")
    assert set(cohort_metrics(stats, unrouted)) == {"dark", "bright"}


def test_paired_bootstrap_is_paired_and_detects_improvement():
    ids = [f"i{i}" for i in range(40)]
    before = synthetic(ids, quality=0.3, seed=1)
    after = synthetic(ids, quality=0.8, seed=1)
    groups = {sample_id: f"g{i // 2}" for i, sample_id in enumerate(ids)}
    result = paired_bootstrap(before, after, groups, [0, 1], resamples=30)
    assert result["delta"] > 0 and result["ci95"][0] > 0 and result["groups"] == 20
    same = paired_bootstrap(before, before, groups, [0, 1], resamples=5)
    assert same["delta"] == 0 and same["ci95"] == [0.0, 0.0]
    changed = {k: {**v, "target_cls": v["target_cls"][:1]} for k, v in after.items()}
    with pytest.raises(ValueError, match="ground truth"):
        paired_bootstrap(before, changed, groups, [0, 1])
    assert class_ap(before, []) == {}


def test_saved_result_picks_newest_evaluated_run(tmp_path):
    for date in ("2026-01-01", "2026-02-01", "2026-03-01"):
        (tmp_path / f"runs/v0-{date}").mkdir(parents=True)
    for date in ("2026-01-01", "2026-02-01"):
        write_json(tmp_path / f"runs/v0-{date}/eval/metrics.json", {"date": date})
    run, metrics = saved_result(tmp_path, "v0")
    assert run.name == "v0-2026-02-01" and metrics == {"date": "2026-02-01"}
    assert saved_result(tmp_path, "v0", "2026-01-01")[1] == {"date": "2026-01-01"}
    assert saved_result(tmp_path, "v1") is None
