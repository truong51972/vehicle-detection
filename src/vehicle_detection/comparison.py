"""Read-only first cross-evaluation plan, then opt-in native YOLO validation.

Only Raw-vs-policy 2x2 comparisons are allowed. Training and data exports are
never started by this module. The final holdout is never accessed.
"""

from importlib.metadata import version as package_version
from pathlib import Path

import numpy as np
import pandas as pd

from .common import file_hash, read_json, relative, write_json
from .data import FIVE_CLASSES
from .experiment_data import load_experiment, run_dir, verify_data

POLICIES = ("v1", "v2", "v3", "v4")
VERSIONS = ("v0", *POLICIES)
CONTRASTS = (
    "inference_on_raw_model",
    "training_on_processed_input",
    "training_on_raw_input",
    "inference_on_policy_model",
    "joint_pipeline",
    "interaction",
)


def comparison_pairs(policies=POLICIES):
    """Thirteen valid cells: one shared Raw cell and three per policy."""
    if not policies or len(set(policies)) != len(policies) or set(policies) - set(POLICIES):
        raise ValueError("Select distinct policies from v1, v2, v3, v4")
    return [("v0", "v0")] + [
        pair for policy in policies for pair in (("v0", policy), (policy, "v0"), (policy, policy))
    ]


def contrasts(a, b, c, d):
    """2x2 factorial contrasts in metric units (multiply by 100 for pp)."""
    return {
        "inference_on_raw_model": b - a,
        "training_on_processed_input": d - b,
        "training_on_raw_input": c - a,
        "inference_on_policy_model": d - c,
        "joint_pipeline": d - a,
        "interaction": (d - c) - (b - a),
    }


def cell_name(model, inputs):
    if (model, inputs) not in comparison_pairs():
        raise ValueError(f"Unsupported cross-evaluation: {model} x {inputs}")
    return f"{model}/{inputs}"


def _output(root, model_run, input_run):
    return root / "runs/comparisons" / f"{model_run.name}__{input_run.name}"


def _metadata(run):
    frame = pd.read_csv(run / "data/metadata.csv", keep_default_na=False)
    test = frame.loc[frame.split == "test"].copy()
    if test.id.duplicated().any() or test.empty:
        raise ValueError(f"Duplicate or empty development IDs in {run}")
    return test.set_index("id", drop=False)


def _runs(root, run_dates):
    from .training import completed_training, training_settings

    contexts, blocked = {}, {}
    for v in VERSIONS:
        try:
            run = run_dir(root, v, run_dates.get(v))
        except FileNotFoundError as exc:
            blocked[v] = str(exc)
            continue
        config = load_experiment(root, v)
        try:
            # Verify all exported bytes once per run, not once per matrix cell.
            prepared = verify_data(root, v, run.name.removeprefix(f"{v}-"))
            if completed_training(root, run, training_settings(config)) is None:
                blocked[v] = f"{run.name}: training is not completed"
                continue
            checkpoint = run / "train/weights/best.pt"
            contexts[v] = {
                "version": v,
                "run": run,
                "config": config,
                "prepared": prepared,
                "checkpoint": checkpoint,
                "checkpoint_sha256": file_hash(checkpoint),
                "metadata": _metadata(run),
            }
        except FileNotFoundError as exc:
            blocked[v] = str(exc)
    ready = list(contexts.values())
    if ready:
        reference = ready[0]
        ref_ids = set(reference["metadata"].index)
        ref_groups = reference["metadata"].source_group.to_dict()
        ref_eval = reference["config"]["evaluation"]
        ref_splits = reference["prepared"]["splits_sha256"]
        for ctx in ready[1:]:
            if (
                set(ctx["metadata"].index) != ref_ids
                or ctx["metadata"].source_group.to_dict() != ref_groups
                or ctx["prepared"]["splits_sha256"] != ref_splits
                or ctx["config"]["evaluation"] != ref_eval
            ):
                raise ValueError(
                    "Matrix runs disagree on IDs, source groups, splits or evaluation args"
                )
    return contexts, blocked


def _identity(model, inputs):
    settings = model["config"]["evaluation"]
    return {
        "model_run": model["run"].name,
        "input_run": inputs["run"].name,
        "checkpoint_sha256": model["checkpoint_sha256"],
        "input_files_sha256": inputs["prepared"]["files_sha256"],
        "input_metadata_sha256": inputs["prepared"]["metadata_sha256"],
        "settings": settings,
        "ultralytics": package_version("ultralytics"),
        "role": "development_validation",
        "classes": FIVE_CLASSES,
    }


def _diagonal_identity(ctx):
    return {
        "run": relative(ctx["run"], ctx["run"].parents[1]),
        "files_sha256": ctx["prepared"]["files_sha256"],
        "checkpoint_sha256": ctx["checkpoint_sha256"],
        "settings": ctx["config"]["evaluation"],
        "ultralytics": package_version("ultralytics"),
    }


def _cached(path, expected, expected_ids):
    metrics_path = path / "metrics.json"
    if not metrics_path.is_file():
        if path.exists() and any(path.iterdir()):
            return "incomplete", None
        return "missing", None
    result = read_json(metrics_path)
    if result.get("inputs") != expected:
        return "stale", None
    stats_path = path / "paired_stats.npz"
    if not stats_path.is_file():
        return "incomplete", None
    if result.get("stats_sha256") and result["stats_sha256"] != file_hash(stats_path):
        return "stale", None
    # Inspect only the IDs here; full match statistics are loaded once for analysis.
    with np.load(stats_path, allow_pickle=False) as archive:
        ids = archive["ids"].tolist()
    if len(ids) != len(set(ids)) or set(ids) != expected_ids:
        return "stale", None
    return "cached", result


def inspect_matrix(root, *, run_dates=None, policies=POLICIES):
    """Preflight statuses, without inference or training. Incomplete v3 is blocked."""
    root = Path(root).resolve()
    contexts, blocked = _runs(root, run_dates or {})
    rows, results = [], {}
    for model_version, input_version in comparison_pairs(policies):
        key = cell_name(model_version, input_version)
        absent = [v for v in (model_version, input_version) if v not in contexts]
        if absent:
            rows.append(
                {
                    "model": model_version,
                    "input": input_version,
                    "status": "blocked",
                    "detail": "; ".join(blocked.get(v, v) for v in dict.fromkeys(absent)),
                }
            )
            continue
        model, inputs = contexts[model_version], contexts[input_version]
        diagonal = model_version == input_version
        target = model["run"] / "eval" if diagonal else _output(root, model["run"], inputs["run"])
        identity = _diagonal_identity(model) if diagonal else _identity(model, inputs)
        status, result = _cached(target, identity, set(inputs["metadata"].index))
        rows.append(
            {
                "model": model_version,
                "input": input_version,
                "status": status,
                "detail": relative(target, root),
            }
        )
        if result is not None:
            results[key] = result
    return {
        "root": root,
        "contexts": contexts,
        "blocked": blocked,
        "rows": rows,
        "results": results,
    }


def _evaluate_cross(root, model, inputs):
    from ultralytics import YOLO

    from .evaluation import ExperimentValidator, cohort_metrics
    from .evaluation_stats import load_stats

    target = _output(root, model["run"], inputs["run"])
    if target.exists() and any(target.iterdir()):
        raise ValueError(f"Refusing to overwrite incomplete evaluation: {target}")
    detector = YOLO(str(model["checkpoint"]))
    if detector.names != dict(enumerate(FIVE_CLASSES)):
        raise ValueError("Cross-evaluation detector taxonomy differs from five-class GT")
    settings = model["config"]["evaluation"]
    if settings.get("half") or settings.get("augment"):
        raise ValueError("Cross-evaluation requires native FP32 validation, no TTA")
    native = detector.val(
        validator=ExperimentValidator,
        data=str(inputs["run"] / "data/data.yaml"),
        split="test",
        project=str(target.parent),
        name=target.name,
        exist_ok=True,
        seed=model["config"]["seed"],
        **settings,
    )
    stats_path = target / "paired_stats.npz"
    stats = load_stats(stats_path)
    if set(stats) != set(inputs["metadata"].index):
        raise ValueError("Cross-evaluation returned incomplete/incorrect IDs")
    p, r, ap50, ap = map(float, native.mean_results())
    result = {
        "inputs": _identity(model, inputs),
        "overall": {"precision": p, "recall": r, "mAP50": ap50, "mAP50_95": ap},
        "classes": __import__("json").loads(native.to_json(decimals=8)),
        "cohorts": cohort_metrics(stats, inputs["metadata"].reset_index(drop=True)),
        "speed_ms": native.speed,
        "role": "development_validation",
        "final_test_accessed": False,
        "stats_sha256": file_hash(stats_path),
    }
    write_json(target / "metrics.json", result)
    return result


def evaluate_matrix(
    root, *, run_dates=None, policies=POLICIES, allow_inference=False, preflight=None
):
    """Reuse valid caches; only infer missing cells on explicit opt-in.

    Stale/incomplete cells are fail-closed: never delete, overwrite or silently
    recompute them. The diagonal is always owned by each version's 04_eval.
    """
    report = (
        preflight
        if preflight is not None
        else inspect_matrix(root, run_dates=run_dates, policies=policies)
    )
    if report["root"] != Path(root).resolve():
        raise ValueError("Preflight belongs to a different project root")
    if not allow_inference:
        return report
    from .evaluation import evaluate_experiment

    for row in report["rows"]:
        status = row["status"]
        if status in {"blocked", "cached"}:
            continue
        if status != "missing":
            raise ValueError(f"{row['model']}/{row['input']}: {status}; inspect artifacts")
        model_version, input_version = row["model"], row["input"]
        model = report["contexts"][model_version]
        inputs = report["contexts"][input_version]
        if model_version == input_version:
            result = evaluate_experiment(
                root,
                model_version,
                model["run"].name.removeprefix(f"{model_version}-"),
                allow_inference=True,
            )
        else:
            result = _evaluate_cross(Path(root).resolve(), model, inputs)
        report["results"][cell_name(model_version, input_version)] = result
        row["status"] = "evaluated"
    return report


def result_table(report):
    """Results and statuses, with percentage-unit display fields."""
    rows = []
    for row in report["rows"]:
        item = {**row, "mAP50_pct": np.nan, "mAP50_95_pct": np.nan}
        result = report["results"].get(cell_name(row["model"], row["input"]))
        if result:
            item.update({k + "_pct": result["overall"][k] * 100 for k in ("mAP50", "mAP50_95")})
        rows.append(item)
    return pd.DataFrame(rows)


def compare_policies(report, *, bootstrap_resamples=0, seed=2026):
    """Six contrasts per available 2x2; one paired group draw for all four cells."""
    if bootstrap_resamples < 0 or bootstrap_resamples == 1:
        raise ValueError("Use zero or at least two bootstrap resamples")
    from .evaluation_stats import factorial_bootstrap, load_stats

    available = report["results"]
    contexts = report["contexts"]
    root = report["root"]
    stats_cache = {}
    rows, intervals = [], {}
    for policy in POLICIES:
        keys = [
            cell_name(*pair)
            for pair in (("v0", "v0"), ("v0", policy), (policy, "v0"), (policy, policy))
        ]
        if not all(key in available for key in keys):
            continue
        a, b, c, d = [available[key]["overall"]["mAP50_95"] for key in keys]
        for name, delta in contrasts(a, b, c, d).items():
            rows.append({"policy": policy, "contrast": name, "delta_pp": delta * 100})
        if not bootstrap_resamples:
            continue
        arms = {}
        for key in keys:
            model_version, input_version = key.split("/")
            model, inputs = contexts[model_version], contexts[input_version]
            folder = (
                model["run"] / "eval"
                if model_version == input_version
                else _output(root, model["run"], inputs["run"])
            )
            path = folder / "paired_stats.npz"
            if path not in stats_cache:
                stats_cache[path] = load_stats(path)
            arms[key] = stats_cache[path]
        raw = contexts["v0"]["metadata"]
        groups = raw.source_group.to_dict()
        intervals[policy] = factorial_bootstrap(
            arms,
            groups,
            range(len(FIVE_CLASSES)),
            policy=policy,
            resamples=bootstrap_resamples,
            seed=seed,
        )
    frame = pd.DataFrame(rows)
    if not frame.empty and intervals:
        for i, row in frame.iterrows():
            if row["policy"] not in intervals:
                continue
            info = intervals[row["policy"]]["contrasts"][row["contrast"]]
            frame.loc[i, "ci_low_pp"] = info["ci95"][0] * 100 if info["ci95"] else np.nan
            frame.loc[i, "ci_high_pp"] = info["ci95"][1] * 100 if info["ci95"] else np.nan
            frame.loc[i, "valid_draws"] = info["valid_draws"]
    return frame, intervals


def policy_cohorts(report, *, policy):
    """Identical policy-defined image cohorts in A/B/C/D, regardless of model/input."""
    if policy not in POLICIES:
        raise ValueError("Unknown policy")
    from .evaluation import cohort_metrics
    from .evaluation_stats import load_stats

    contexts = report["contexts"]
    results = report["results"]
    if policy not in contexts or "v0" not in contexts:
        return pd.DataFrame()
    metadata = contexts[policy]["metadata"].reset_index(drop=True)
    rows = []
    for model_version, input_version in (
        ("v0", "v0"),
        ("v0", policy),
        (policy, "v0"),
        (policy, policy),
    ):
        key = cell_name(model_version, input_version)
        if key not in results:
            continue
        model, inputs = contexts[model_version], contexts[input_version]
        folder = (
            (model["run"] / "eval")
            if model_version == input_version
            else _output(report["root"], model["run"], inputs["run"])
        )
        measures = cohort_metrics(load_stats(folder / "paired_stats.npz"), metadata)
        for cohort, item in measures.items():
            rows.append(
                {
                    "model": model_version,
                    "input": input_version,
                    "cohort": cohort,
                    "images": item["images"],
                    "mAP50_95_pct": item["mAP50_95"] * 100
                    if item["mAP50_95"] is not None
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def benchmark_preprocessing(report, *, samples=128, repeats=3, seed=2026):
    """Optional CPU-only policy timing, using saved Raw JPEGs as proxy inputs.

    Excludes image decode, JPEG encoding, disk I/O and the detector. This is not
    an end-to-end deployment benchmark and does not reproduce original-source
    preprocessing bytes. No original/final images are opened here.
    """
    import time

    import cv2

    from .common import read_image
    from .preprocess import apply_recipe

    if samples < 1 or repeats < 1:
        raise ValueError("Require positive samples/repeats")
    raw = report["contexts"].get("v0")
    if raw is None:
        return pd.DataFrame()
    ids = sorted(raw["metadata"].index)
    chosen = np.random.default_rng(seed).choice(ids, min(len(ids), samples), replace=False)
    images = [
        read_image(raw["run"] / "data/images/test" / f"{sample_id}.jpg") for sample_id in chosen
    ]
    original_threads = cv2.getNumThreads()
    records = []
    try:
        cv2.setNumThreads(1)
        for v in VERSIONS:
            ctx = report["contexts"].get(v)
            if ctx is None:
                continue
            if v == "v0":
                records.append(
                    {
                        "policy": v,
                        "samples": len(images),
                        "repeats": repeats,
                        "mean_ms": 0.0,
                        "p50_ms": 0.0,
                        "p95_ms": 0.0,
                        "note": "Raw has no external policy operation",
                    }
                )
                continue
            variant, recipes = ctx["config"]["variant"], ctx["config"]["recipes"]
            for image in images[: min(5, len(images))]:
                apply_recipe(image, variant, recipes)
            measurements = []
            for _ in range(repeats):
                for image in images:
                    begin = time.perf_counter()
                    apply_recipe(image, variant, recipes)
                    measurements.append((time.perf_counter() - begin) * 1000)
            records.append(
                {
                    "policy": v,
                    "samples": len(images),
                    "repeats": repeats,
                    "mean_ms": float(np.mean(measurements)),
                    "p50_ms": float(np.quantile(measurements, 0.5)),
                    "p95_ms": float(np.quantile(measurements, 0.95)),
                    "note": "CPU policy + routing only; Raw JPEG proxy",
                }
            )
    finally:
        cv2.setNumThreads(original_threads)
    return pd.DataFrame(records)


def training_cost(report):
    """Reported native training duration from saved Ultralytics results.csv."""
    rows = []
    for v, ctx in report["contexts"].items():
        path = ctx["run"] / "train/results.csv"
        history = pd.read_csv(path)
        history.columns = history.columns.str.strip()
        rows.append(
            {
                "model": v,
                "epochs": len(history),
                "train_minutes": float(history["time"].iloc[-1]) / 60
                if "time" in history
                else np.nan,
                "source": relative(path, report["root"]),
            }
        )
    return pd.DataFrame(rows)
