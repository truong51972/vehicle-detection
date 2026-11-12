"""Read-only first cross-evaluation plan, then opt-in native YOLO validation.

Only Raw-vs-policy 2x2 comparisons are allowed. Training and data exports are
never started by this module. The final holdout is never accessed.
"""

from importlib.metadata import version as package_version
from pathlib import Path
from zipfile import BadZipFile

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


def _runs(root, run_dates, versions=VERSIONS):
    from .training import completed_training, training_settings

    contexts, blocked = {}, {}
    for v in versions:
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
    comparison_pairs(policies)
    contexts, blocked = _runs(root, run_dates or {}, ("v0", *policies))
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
    if report.get("mode") == "report":
        raise ValueError("Report evidence cannot authorize inference; use inspect_matrix")
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
        metrics = ("mAP50", "mAP50_95", "precision", "recall")
        model = report["contexts"].get(row["model"], {})
        inputs = report["contexts"].get(row["input"], {})
        item = {
            **row,
            "split": "development_validation",
            "model_run": model["run"].name if model else "unavailable",
            "input_run": inputs["run"].name if inputs else "unavailable",
            "profile": model.get("profile", "live preflight"),
            **{k + "_pct": np.nan for k in metrics},
        }
        result = report["results"].get(cell_name(row["model"], row["input"]))
        if result:
            item.update({k + "_pct": result["overall"][k] * 100 for k in metrics})
        rows.append(item)
    return pd.DataFrame(rows)


def compare_policies(report, *, bootstrap_resamples=0, seed=2026):
    """Six contrasts per available 2x2; one paired group draw for all four cells."""
    if bootstrap_resamples < 0 or bootstrap_resamples == 1:
        raise ValueError("Use zero or at least two bootstrap resamples")
    available = report["results"]
    contexts = report["contexts"]
    stats_cache = {}
    rows, intervals = [], {}
    for policy in POLICIES:
        keys = [
            cell_name(*pair)
            for pair in (("v0", "v0"), ("v0", policy), (policy, "v0"), (policy, policy))
        ]
        if not all(key in available for key in keys):
            continue
        if not compatible_runs(contexts["v0"], contexts[policy]):
            continue
        a, b, c, d = [available[key]["overall"]["mAP50_95"] for key in keys]
        for name, delta in contrasts(a, b, c, d).items():
            rows.append({"policy": policy, "contrast": name, "delta_pp": delta * 100})
        if not bootstrap_resamples:
            continue
        paths = [result_folder(report, key) / "paired_stats.npz" for key in keys]
        if not all(path.is_file() for path in paths):
            continue
        from .evaluation_stats import factorial_bootstrap, load_stats

        arms = {}
        for key, path in zip(keys, paths, strict=True):
            if path not in stats_cache:
                stats_cache[path] = load_stats(path)
            arms[key] = stats_cache[path]
        raw = contexts["v0"].get("metadata")
        if raw is None or "source_group" not in raw:
            continue
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
    contexts = report["contexts"]
    results = report["results"]
    if policy not in contexts or "v0" not in contexts:
        return pd.DataFrame()
    if (
        not compatible_runs(contexts["v0"], contexts[policy])
        or "routing_metadata" not in contexts[policy]
        and report.get("mode") == "report"
    ):
        return pd.DataFrame()
    from .evaluation import cohort_metrics
    from .evaluation_stats import load_stats

    metadata = contexts[policy].get("routing_metadata", contexts[policy]["metadata"])
    metadata = metadata.reset_index(drop=True)
    masks = {
        "dark": metadata.brightness < 0.12,
        "bright": metadata.brightness >= 0.12,
        "apply": metadata.applied.astype(str).ne(""),
        "skip": metadata.applied.astype(str).eq(""),
    }
    rows = []
    reference = None
    for model_version, input_version in (
        ("v0", "v0"),
        ("v0", policy),
        (policy, "v0"),
        (policy, policy),
    ):
        key = cell_name(model_version, input_version)
        if key not in results:
            continue
        path = result_folder(report, key) / "paired_stats.npz"
        if not path.is_file():
            return pd.DataFrame()
        stats = load_stats(path)
        if set(stats) != set(metadata.id):
            raise ValueError("Cohorts require identical development IDs in every arm")
        if reference is None:
            reference = stats
        elif any(
            not np.array_equal(reference[sample]["target_cls"], stats[sample]["target_cls"])
            for sample in stats
        ):
            raise ValueError("Cohort ground truth changed across arms")
        measures = cohort_metrics(stats, metadata)
        for cohort, item in measures.items():
            rows.append(
                {
                    "model": model_version,
                    "input": input_version,
                    "cohort": cohort,
                    "images": item["images"],
                    "gt_instances": sum(
                        len(stats[sample]["target_cls"]) for sample in metadata.id[masks[cohort]]
                    ),
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
        if not path.is_file():
            continue
        history = pd.read_csv(path)
        history.columns = history.columns.str.strip()
        if history.empty:
            continue
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


def protocol(config):
    """Policy is the treatment; all training/evaluation settings must otherwise match."""
    return {k: config[k] for k in ("training", "evaluation", "checkpoint_sha256", "seed")}


def compatible_runs(a, b):
    if protocol(a["config"]) != protocol(b["config"]):
        return False
    if "profile" in a or "profile" in b:
        return (
            a.get("verified", False)
            and b.get("verified", False)
            and a["profile"] == b["profile"]
            and a.get("runtime") == b.get("runtime")
            and a.get("ultralytics") == b.get("ultralytics")
        )
    return True


def result_folder(report, key):
    if "paths" in report:
        return report["paths"][key]
    model, inputs = (report["contexts"][v] for v in key.split("/"))
    return (
        model["run"] / "eval"
        if model is inputs
        else _output(report["root"], model["run"], inputs["run"])
    )


def inspect_report(root, *, run_dates=None):
    """Read archived evidence without exports, checkpoints, native runtime or image access.

    The provenance index attests the archived split/GT protocol and runtime; it
    does not revalidate exported pixels/boxes. Unindexed runs are descriptive only.
    """
    import yaml

    from .experiment_data import shared_splits

    root = Path(root).resolve()
    shared, rows = shared_splits(root)
    metadata = pd.DataFrame(r for r in rows if r["role"] == "development_validation")
    metadata = metadata.set_index("id", drop=False)
    index_path = root / "configs/comparison_evidence.json"
    index = read_json(index_path) if index_path.is_file() else {"runs": {}, "artifacts": {}}
    report = {
        "root": root,
        "mode": "report",
        "contexts": {},
        "blocked": {},
        "results": {},
        "paths": {},
        "rows": [],
        "provenance": [],
    }

    def check_index(path):
        expected = index["artifacts"].get(relative(path, root))
        if expected and file_hash(path) != expected:
            raise ValueError(f"Archived artifact hash changed: {relative(path, root)}")
        return expected is not None

    def read_result(folder, expected):
        path = folder / "metrics.json"
        if not path.is_file():
            return "incomplete" if folder.exists() and any(folder.iterdir()) else "missing", None
        check_index(path)
        result = read_json(path)
        if result.get("inputs") != expected:
            raise ValueError("Saved cache identity differs from selected run/config/checkpoint")
        if (
            result.get("role") != "development_validation"
            or result.get("final_test_accessed") is not False
        ):
            raise ValueError("Require development-only evidence; final holdout must stay sealed")
        for metric in ("precision", "recall", "mAP50", "mAP50_95"):
            value = result["overall"][metric]
            if not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid metric proportion: {metric}")
        stats = folder / "paired_stats.npz"
        if stats.is_file():
            check_index(stats)
            if result.get("stats_sha256") and file_hash(stats) != result["stats_sha256"]:
                raise ValueError("Paired statistics hash changed")
            with np.load(stats, allow_pickle=False) as archive:
                ids = archive["ids"].tolist()
            if len(ids) != len(set(ids)) or set(ids) != set(metadata.index):
                raise ValueError("Paired statistics development IDs changed")
        return "cached", result

    selections = {}
    for v in VERSIONS:
        try:
            selections[v] = run_dir(root, v, (run_dates or {}).get(v))
        except FileNotFoundError as exc:
            report["blocked"][v] = ("missing", str(exc))
    # This alias is an extra matched study, never the shared v0/v0 factorial arm.
    archive = root / "docs/v4/evidence/matched_raw"
    if archive.exists():
        selections["matched_raw"] = archive
    for v, run in selections.items():
        try:
            config_path = run / "config.yaml"
            config = yaml.safe_load(config_path.read_text())
            protocol(config)
            marker_path = run / "train/complete.json"
            marker = read_json(marker_path)
            check_index(config_path)
            check_index(marker_path)
            run_id = Path(marker["run"]).name
            if v != "matched_raw" and (run_id != run.name or config["id"] != v):
                raise ValueError("Run/config/completion identity mismatch")
            if v == "v0" and config.get("variant") != "raw":
                raise ValueError("Historical Raw baseline must use Raw inputs")
            if v == "matched_raw" and config.get("variant") != "raw":
                raise ValueError("Archived matched control must use Raw inputs")
            folder = run if v == "matched_raw" else run / "eval"
            saved = read_json(folder / "metrics.json")
            expected = {
                "run": f"runs/{run_id}",
                "files_sha256": saved["inputs"]["files_sha256"],
                "checkpoint_sha256": marker["best_sha256"],
                "settings": config["evaluation"],
                "ultralytics": saved["inputs"]["ultralytics"],
            }
            status, result = read_result(folder, expected)
            if result is None:
                report["blocked"][v] = (status, "Saved metrics unavailable")
                continue
            evidence = index["runs"].get(run_id, {})
            verified = (
                evidence.get("splits_sha256") == shared["splits_sha256"]
                and all(
                    relative(p, root) in index["artifacts"]
                    for p in (config_path, marker_path, folder / "metrics.json")
                )
                and bool(evidence.get("runtime"))
            )
            ctx = {
                "run": run,
                "run_id": run_id,
                "config": config,
                "checkpoint_sha256": marker["best_sha256"],
                "metadata": metadata,
                "profile": evidence.get("profile", f"unverified:{run_id}"),
                "verified": verified,
                "runtime": evidence.get("runtime", "unrecorded"),
                "ultralytics": saved["inputs"]["ultralytics"],
                "limits": evidence.get("limits", "Unverified protocol; descriptive only"),
            }
            routing = run / "data/metadata.csv"
            if routing.is_file() and check_index(routing):
                routed = _metadata(run)
                if set(routed.index) != set(metadata.index) or (
                    routed.source_group.to_dict() != metadata.source_group.to_dict()
                ):
                    raise ValueError("Routing metadata development IDs/source groups changed")
                ctx["routing_metadata"] = routed
            report["contexts"][v] = ctx
            key = f"{v}/{v}"
            report["results"][key] = result
            report["paths"][key] = folder
            report["provenance"].append(
                {
                    "model/input": key,
                    "run_id": run_id,
                    "profile": ctx["profile"],
                    "baseline_eligible": verified,
                    "checkpoint_sha256": marker["best_sha256"],
                    "metrics": relative(folder / "metrics.json", root),
                    "metrics_sha256": file_hash(folder / "metrics.json"),
                    "config_sha256": file_hash(config_path),
                    "runtime": ctx["runtime"],
                    "split": "development_validation",
                    "splits_sha256": shared["splits_sha256"],
                    "protocol_source": evidence.get("source", "unrecorded; descriptive only"),
                    "limits": ctx["limits"],
                }
            )
        except FileNotFoundError as exc:
            report["blocked"][v] = ("incomplete" if run.exists() else "missing", str(exc))
        except (
            ValueError,
            KeyError,
            TypeError,
            OSError,
            EOFError,
            BadZipFile,
            yaml.YAMLError,
        ) as exc:
            report["blocked"][v] = ("stale", str(exc))

    for model, inputs in comparison_pairs():
        key = cell_name(model, inputs)
        detail = ""
        if key in report["results"]:
            status = "cached"
            ctx = report["contexts"][model]
            detail = ctx["profile"] + (
                "; archived protocol" if ctx["verified"] else "; descriptive only"
            )
        elif any(v not in report["contexts"] for v in (model, inputs)):
            reasons = [
                report["blocked"][v]
                for v in dict.fromkeys((model, inputs))
                if v in report["blocked"]
            ]
            status = (
                reasons[0][0]
                if model == inputs
                else "stale"
                if any(s == "stale" for s, _ in reasons)
                else "blocked"
            )
            detail = "; ".join(reason for _, reason in reasons)
        else:
            a, b = (report["contexts"][v] for v in (model, inputs))
            if not compatible_runs(a, b):
                status, detail = (
                    "blocked",
                    "Non-comparable or unverified training/evaluation/runtime profiles",
                )
            else:
                folder = _output(root, a["run"], b["run"])
                diag = report["results"][f"{inputs}/{inputs}"]["inputs"]
                path = folder / "metrics.json"
                try:
                    if path.is_file():
                        saved = read_json(path)
                        expected = {
                            "model_run": a["run_id"],
                            "input_run": b["run_id"],
                            "checkpoint_sha256": a["checkpoint_sha256"],
                            "input_files_sha256": diag["files_sha256"],
                            "input_metadata_sha256": saved["inputs"]["input_metadata_sha256"],
                            "settings": a["config"]["evaluation"],
                            "ultralytics": diag["ultralytics"],
                            "role": "development_validation",
                            "classes": FIVE_CLASSES,
                        }
                        if not check_index(path):
                            raise ValueError(
                                "Unindexed cross-cache; archived metadata identity unavailable"
                            )
                        status, result = read_result(folder, expected)
                        if result is not None:
                            report["results"][key] = result
                            report["paths"][key] = folder
                    else:
                        status, _ = read_result(folder, {})
                    detail = relative(folder, root)
                except (ValueError, KeyError, TypeError, OSError, EOFError, BadZipFile) as exc:
                    status, detail = "stale", str(exc)
        report["rows"].append({"model": model, "input": inputs, "status": status, "detail": detail})
    return report


def matched_studies(report):
    """Only same-profile baselines, with the archived control confined to v4."""
    studies = []
    for baseline, versions in (("v0", ("v1", "v2", "v3")), ("matched_raw", ("v4",))):
        ctx = report["contexts"].get(baseline)
        key = f"{baseline}/{baseline}"
        if ctx is None or key not in report["results"]:
            continue
        keys = [key]
        for v in versions:
            other = report["contexts"].get(v)
            if other is not None and compatible_runs(ctx, other):
                keys.extend(k for k in (f"v0/{v}", f"{v}/v0", f"{v}/{v}") if k in report["results"])
        if len(keys) > 1:
            studies.append({"profile": ctx["profile"], "baseline": key, "keys": keys})
    return studies


def class_deltas(report, study, metric="mAP50-95"):
    """Class pp changes with absent/zero GT support retained as unavailable."""
    results = report["results"]
    baseline = {item["Class"]: item for item in results[study["baseline"]].get("classes", [])}
    rows = []
    for key in study["keys"][1:]:
        classes = {item["Class"]: item for item in results[key].get("classes", [])}
        for name in FIVE_CLASSES:
            a, b = baseline.get(name, {}), classes.get(name, {})
            supported = a.get("Instances", 0) > 0 and b.get("Instances", 0) > 0
            rows.append(
                {
                    "pair": key,
                    "class": name,
                    "baseline": study["baseline"],
                    "delta_pp": (b[metric] - a[metric]) * 100
                    if supported and metric in a and metric in b
                    else np.nan,
                    "gt_instances": b.get("Instances"),
                }
            )
    return pd.DataFrame(rows)


def training_histories(report):
    """Validated archived calibration histories; no checkpoint deserialization."""
    import yaml

    index_path = report["root"] / "configs/comparison_evidence.json"
    index = read_json(index_path).get("artifacts", {}) if index_path.is_file() else {}
    histories, unavailable = {}, {}
    for v, ctx in report["contexts"].items():
        path = ctx["run"] / "train/results.csv"
        try:
            args_path = path.with_name("args.yaml")
            for artifact in (path, args_path):
                expected = index.get(relative(artifact, report["root"]))
                if expected and file_hash(artifact) != expected:
                    raise ValueError("Archived training artifact hash changed")
            args = yaml.safe_load(args_path.read_text())
            settings = {**ctx["config"]["training"], "seed": ctx["config"]["seed"]}
            if any(args.get(k) != value for k, value in settings.items()):
                raise ValueError("Native training recipe differs from config snapshot")
            history = pd.read_csv(path)
            history.columns = history.columns.str.strip()
            marker = read_json(path.with_name("complete.json"))
            if history.empty or "epoch" not in history or len(history) != marker["epochs_run"]:
                raise ValueError("Training history is incomplete")
            if history.epoch.tolist() != list(range(1, len(history) + 1)):
                raise ValueError("Training epoch sequence changed")
            histories[v] = history
        except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
            unavailable[v] = str(exc)
    return histories, unavailable


def evidence_summary(report, effects, *, intervals=None):
    """Conclusions come only from matched stored evidence, never from a shared ranking."""
    ready = sum(row["status"] in {"cached", "evaluated"} for row in report["rows"])
    lines = [f"Development matrix: {ready}/13 saved cells; {13 - ready}/13 pending."]
    for policy in POLICIES:
        subset = effects.loc[effects.policy == policy] if not effects.empty else effects
        if subset.empty:
            lines.append(
                f"{policy}: incomplete or non-comparable 2×2; factorial effects unavailable."
            )
        else:
            values = "; ".join(
                f"{row.contrast}: {row.delta_pp:+.4f} pp" for row in subset.itertuples()
            )
            estimates = (intervals or {}).get(policy, {}).get("contrasts", {})
            ci = (
                "paired 95% CIs available"
                if estimates and all(item["ci95"] is not None for item in estimates.values())
                else "CI unavailable"
            )
            lines.append(f"{policy}: {values}. {ci}.")
    for study in matched_studies(report):
        baseline = report["results"][study["baseline"]]["overall"]
        for key in study["keys"][1:]:
            if key.split("/")[0] != key.split("/")[1]:
                continue
            overall = report["results"][key]["overall"]
            changes = "; ".join(
                f"{metric}: {(overall[metric] - baseline[metric]) * 100:+.4f} pp"
                for metric in ("mAP50", "mAP50_95")
            )
            deltas = [overall[m] - baseline[m] for m in ("mAP50", "mAP50_95")]
            direction = (
                "Both AP measures lower"
                if all(d < 0 for d in deltas)
                else "Both AP measures higher"
                if all(d > 0 for d in deltas)
                else "Mixed or unchanged AP measures"
            )
            lines.append(
                f"{study['profile']}, {key} vs {study['baseline']}: {changes}. "
                f"{direction} in this single-seed study."
            )
            classes = class_deltas(report, {**study, "keys": [study["baseline"], key]}, "mAP50")
            valid = classes.dropna(subset=["delta_pp"])
            if not valid.empty:
                best, worst = valid.loc[valid.delta_pp.idxmax()], valid.loc[valid.delta_pp.idxmin()]
                lines.append(
                    f"AP50 class deltas: highest {best['class']} {best.delta_pp:+.4f} pp; "
                    f"lowest {worst['class']} {worst.delta_pp:+.4f} pp."
                )
    lines.append(
        "Separate batch/runtime profiles; single trained seed. Intervals, when available, "
        "are conditional on fixed checkpoints with no multiple-comparison adjustment. "
        "Point changes alone do not establish significance or mechanisms. Final holdout sealed."
    )
    return lines
