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


@pytest.fixture
def report_archive(tmp_path):
    """Only small saved evidence: deliberately no images, exports, weights or NPZs."""
    import shutil

    repo = Path(__file__).resolve().parents[1]
    root = tmp_path / "archive"
    for relative in ("configs",):
        shutil.copytree(repo / relative, root / relative)
    for folder in [
        *sorted((repo / "runs").glob("v[0-4]-*")),
        repo / "docs/v4/evidence/matched_raw",
    ]:
        for name in (
            "config.yaml",
            "eval/metrics.json",
            "metrics.json",
            "train/complete.json",
            "train/args.yaml",
            "train/results.csv",
        ):
            source = folder / name
            if source.is_file():
                target = root / source.relative_to(repo)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
    for source in (repo / "runs/comparisons").glob("*/metrics.json"):
        target = root / source.relative_to(repo)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    (root / "pyproject.toml").write_text('[project]\nname = "vehicle-detection"\n')
    return root


def test_report_without_exports_weights_npzs_keeps_v4_and_matched_control(report_archive):
    from vehicle_detection.comparison import (
        class_deltas,
        compare_policies,
        evidence_summary,
        inspect_report,
        matched_studies,
        policy_cohorts,
        training_histories,
    )

    report = inspect_report(report_archive)
    assert len(report["rows"]) == 13
    assert sum(row["status"] == "cached" for row in report["rows"]) == 11
    assert set(row["status"] for row in report["rows"] if "v4" in (row["model"], row["input"])) == {
        "cached",
        "blocked",
    }
    assert "matched_raw/matched_raw" in report["results"]
    studies = matched_studies(report)
    assert [s["profile"] for s in studies] == ["historical-batch32", "matched-batch16"]
    assert studies[1]["keys"] == ["matched_raw/matched_raw", "v4/v4"]
    assert not any("v4" in key for key in studies[0]["keys"])
    a = report["results"]["matched_raw/matched_raw"]["overall"]
    d = report["results"]["v4/v4"]["overall"]
    assert (d["mAP50_95"] - a["mAP50_95"]) * 100 == pytest.approx(-0.1507118409)
    assert (d["mAP50"] - a["mAP50"]) * 100 == pytest.approx(-0.5615593582)
    classes = class_deltas(report, studies[1], "mAP50").set_index("class")
    assert classes.loc["motorcycle", "delta_pp"] == pytest.approx(-1.849415)
    assert classes.loc["bicycle", "delta_pp"] == pytest.approx(-1.435488)
    effects, intervals = compare_policies(report, bootstrap_resamples=2)
    assert len(effects) == 18 and intervals == {}
    assert set(effects.policy) == {"v1", "v2", "v3"}
    assert "ci_low_pp" not in effects
    assert policy_cohorts(report, policy="v1").empty
    histories, unavailable = training_histories(report)
    assert set(histories) == {"v0", "v1", "v2", "v3", "v4", "matched_raw"}
    assert unavailable == {}
    text = "\n".join(evidence_summary(report, effects))
    assert "11/13" in text and "-0.1507 pp" in text
    assert "v4: incomplete" in text and "CI unavailable" in text


def test_report_rejects_using_archive_as_inference_preflight(report_archive):
    from vehicle_detection.comparison import evaluate_matrix, inspect_report

    report = inspect_report(report_archive)
    with pytest.raises(ValueError, match="cannot authorize inference"):
        evaluate_matrix(report_archive, preflight=report, allow_inference=True)


@pytest.mark.parametrize(
    "change", ["hash", "checkpoint", "settings", "role", "final", "cross_input", "npz_ids"]
)
def test_report_fails_closed_on_changed_evidence(report_archive, change):
    from vehicle_detection.common import file_hash, read_json
    from vehicle_detection.comparison import inspect_report

    key = "v0/v1" if change == "cross_input" else "v1/v1"
    folder = report_archive / (
        "runs/comparisons/v0-2026-11-09__v1-2026-11-09"
        if change == "cross_input"
        else "runs/v1-2026-11-09/eval"
    )
    path = folder / "metrics.json"
    saved = read_json(path)
    if change == "hash":
        saved["overall"]["mAP50"] += 0.01
    elif change == "checkpoint":
        saved["inputs"]["checkpoint_sha256"] = "wrong"
    elif change == "settings":
        saved["inputs"]["settings"]["batch"] = 16
    elif change == "role":
        saved["role"] = "final_test"
    elif change == "final":
        saved["final_test_accessed"] = True
    elif change == "cross_input":
        saved["inputs"]["input_files_sha256"] = "wrong"
    else:
        np.savez_compressed(folder / "paired_stats.npz", ids=np.array(["wrong"]))
    write_json(path, saved)
    # Refresh only the metrics pin to exercise semantic identity checks too.
    if change != "hash":
        index_path = report_archive / "configs/comparison_evidence.json"
        index = read_json(index_path)
        index["artifacts"][path.relative_to(report_archive).as_posix()] = file_hash(path)
        if change == "npz_ids":
            stats = folder / "paired_stats.npz"
            index["artifacts"][stats.relative_to(report_archive).as_posix()] = file_hash(stats)
        write_json(index_path, index)
    before = {p: file_hash(p) for p in report_archive.rglob("*") if p.is_file()}
    report = inspect_report(report_archive)
    assert key not in report["results"]
    row = next(r for r in report["rows"] if f"{r['model']}/{r['input']}" == key)
    assert row["status"] == "stale"
    assert before == {p: file_hash(p) for p in before}


def test_missing_v4_does_not_hide_other_studies_and_dates_are_honored(report_archive):
    from vehicle_detection.comparison import compare_policies, inspect_report

    report = inspect_report(report_archive, run_dates={"v4": "2026-11-12"})
    assert "v4/v4" not in report["results"]
    assert "matched_raw/matched_raw" in report["results"]
    assert len(compare_policies(report)[0]) == 18
    statuses = {f"{r['model']}/{r['input']}": r["status"] for r in report["rows"]}
    assert statuses["v4/v4"] == "missing"
    assert statuses["v0/v4"] == statuses["v4/v0"] == "blocked"


def test_unindexed_runs_are_descriptive_only_and_incomplete_design_has_no_effects(report_archive):
    from vehicle_detection.comparison import compare_policies, inspect_report, matched_studies

    index = report_archive / "configs/comparison_evidence.json"
    index.unlink()
    report = inspect_report(report_archive)
    assert "v4/v4" in report["results"]
    assert not any(item["baseline_eligible"] for item in report["provenance"])
    assert matched_studies(report) == []
    assert compare_policies(report)[0].empty


def test_partial_cache_missing_cohort_and_zero_class_support(report_archive):
    from vehicle_detection.comparison import (
        class_deltas,
        compare_policies,
        inspect_report,
        matched_studies,
    )

    folder = report_archive / "runs/comparisons/v0-2026-11-09__v1-2026-11-09"
    (folder / "metrics.json").unlink()
    (folder / "partial.txt").write_text("partial")
    report = inspect_report(report_archive)
    row = next(r for r in report["rows"] if (r["model"], r["input"]) == ("v0", "v1"))
    assert row["status"] == "incomplete"
    assert set(compare_policies(report)[0].policy) == {"v2", "v3"}
    study = matched_studies(report)[1]
    report["results"]["v4/v4"]["classes"][0]["Instances"] = 0
    values = class_deltas(report, study).set_index("class")
    assert np.isnan(values.loc["car", "delta_pp"])
    assert np.isfinite(values.loc["truck", "delta_pp"])


def test_changed_history_is_unavailable_not_plotted_as_valid_cost(report_archive):
    from vehicle_detection.comparison import inspect_report, training_histories

    path = report_archive / "runs/v4-2026-11-10/train/results.csv"
    path.write_text("epoch,time\n1,10\n")
    report = inspect_report(report_archive)
    assert "v4/v4" in report["results"]
    histories, unavailable = training_histories(report)
    assert "v4" not in histories and "v4" in unavailable


def test_all_chart_families_render_on_agg_with_missing_diagnostics(report_archive):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from vehicle_detection import comparison_plots as plots
    from vehicle_detection.comparison import compare_policies, inspect_report, training_histories

    report = inspect_report(report_archive)
    effects, _ = compare_policies(report)
    histories, _ = training_histories(report)
    calls = [
        (plots.plot_coverage, (report,)),
        (plots.plot_overall, (report,)),
        (plots.plot_factorial, (report,)),
        (plots.plot_effects, (report, effects)),
        (plots.plot_classes, (report,)),
        (plots.plot_cohorts, (report, {})),
        (plots.plot_convergence, (report, histories)),
        (plots.plot_cost, (report, histories)),
    ]
    for function, args in calls:
        figure = function(*args)
        figure.canvas.draw()
        assert len(figure.axes) > 0
        text = " ".join(t.get_text() for ax in figure.axes for t in ax.texts)
        if function is plots.plot_effects:
            assert "CI unavailable" in " ".join(
                t.get_text() for t in figure.axes[0].get_yticklabels()
            )
        if function is plots.plot_factorial:
            assert "Incomplete" in text
        plt.close(figure)

    # No history, no matched baseline, and no complete factorial must render too.
    report["results"] = {}
    for function, args in (
        (plots.plot_convergence, (report, {})),
        (plots.plot_cost, (report, {})),
        (plots.plot_overall, (report,)),
        (plots.plot_classes, (report,)),
        (plots.plot_effects, (report, effects.iloc[:0])),
    ):
        figure = function(*args)
        figure.canvas.draw()
        plt.close(figure)


def test_compare_notebook_fresh_kernel_without_native_runtime_or_dataset(report_archive):
    import shutil

    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    root = report_archive
    target = root / "notebooks/experiments/compare.ipynb"
    target.parent.mkdir(parents=True)
    shutil.copyfile("notebooks/experiments/compare.ipynb", target)
    notebook = nbformat.read(target, as_version=4)
    # Guard imports and pixel reads in the kernel, not just in this pytest process.
    notebook.cells.insert(
        1,
        nbformat.v4.new_code_cell("""import builtins
from unittest.mock import patch
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'ultralytics'}:
        raise AssertionError('Native runtime imported: ' + name)
    return original_import(name, *args, **kwargs)
patch.object(builtins, '__import__', side_effect=guarded_import).start()
import cv2
patch.object(cv2, 'imread', side_effect=AssertionError('Pixel access')).start()
"""),
    )
    nbclient.NotebookClient(
        notebook, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(root)}}
    ).execute()
    text = "".join(
        "".join(output.get("text", "")) for c in notebook.cells for output in c.get("outputs", [])
    )
    assert "11/13" in text and "-0.1507 pp" in text and "Inference enabled: False" in text
    assert (
        len(
            [
                o
                for c in notebook.cells
                for o in c.get("outputs", [])
                if "image/png" in o.get("data", {})
            ]
        )
        >= 5
    )


@pytest.mark.parametrize(
    "field",
    ["training", "evaluation", "checkpoint_sha256", "seed", "runtime", "ultralytics", "verified"],
)
def test_comparability_requires_recipe_identity_and_runtime(report_archive, field):
    from copy import deepcopy

    from vehicle_detection.comparison import compatible_runs, inspect_report

    report = inspect_report(report_archive)
    a = report["contexts"]["v0"]
    b = deepcopy(report["contexts"]["v1"])
    assert compatible_runs(a, b)
    if field in {"training", "evaluation"}:
        b["config"][field]["batch"] = 16
    elif field in {"checkpoint_sha256", "seed"}:
        b["config"][field] = "changed"
    elif field == "verified":
        b[field] = False
    else:
        b[field] = "another runtime"
    assert not compatible_runs(a, b)


def test_fixed_cohort_membership_gt_support_and_changed_gt(tmp_path):
    pytest.importorskip("ultralytics")
    import pandas as pd

    from vehicle_detection.comparison import policy_cohorts

    metadata = pd.DataFrame(
        {
            "id": ["a", "b"],
            "source_group": ["a", "b"],
            "split": ["test", "test"],
            "brightness": [0.04, 0.4],
            "applied": ["gamma", ""],
        }
    ).set_index("id", drop=False)
    config = {"training": {}, "evaluation": {}, "checkpoint_sha256": "init", "seed": 2026}
    report = {
        "root": tmp_path,
        "mode": "report",
        "contexts": {
            v: {"config": config, "metadata": metadata, "routing_metadata": metadata}
            for v in ("v0", "v1")
        },
        "results": {},
        "paths": {},
    }
    arrays = {"ids": np.array(["a", "b"])}
    for i in range(2):
        arrays.update(
            {
                f"tp_{i}": np.ones((1, 10), dtype=bool),
                f"conf_{i}": np.array([0.9]),
                f"pred_cls_{i}": np.array([0]),
                f"target_cls_{i}": np.array([0]),
            }
        )
    for key in ("v0/v0", "v0/v1", "v1/v0", "v1/v1"):
        folder = tmp_path / key.replace("/", "__")
        folder.mkdir()
        np.savez_compressed(folder / "paired_stats.npz", **arrays)
        report["paths"][key] = folder
        report["results"][key] = {}
    frame = policy_cohorts(report, policy="v1")
    assert len(frame) == 16
    assert set(frame.cohort) == {"dark", "bright", "apply", "skip"}
    assert set(frame.images) == set(frame.gt_instances) == {1}
    for _, group in frame.groupby("cohort"):
        assert group.mAP50_95_pct.nunique() == 1
    arrays["target_cls_0"] = np.array([1])
    np.savez_compressed(report["paths"]["v0/v1"] / "paired_stats.npz", **arrays)
    with pytest.raises(ValueError, match="ground truth changed"):
        policy_cohorts(report, policy="v1")
