"""Native AP aggregation and paired group bootstrap for saved matching statistics."""

from collections import Counter
from pathlib import Path

import numpy as np
from ultralytics.utils.metrics import ap_per_class

from .data import CLASSES

STAT_KEYS = ("tp", "conf", "pred_cls", "target_cls")


def load_stats(path):
    with np.load(path, allow_pickle=False) as data:
        ids = data["ids"].tolist()
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate prediction statistics IDs")
        return {
            sample_id: {key: data[f"{key}_{i}"] for key in STAT_KEYS}
            for i, sample_id in enumerate(ids)
        }


def class_ap(stats, ids):
    """Native AP by represented class; absent classes have no invented AP value."""
    arrays = {k: np.concatenate([stats[i][k] for i in ids], axis=0) for k in STAT_KEYS}
    if not len(arrays["target_cls"]):
        return {}
    values = ap_per_class(arrays["tp"], arrays["conf"], arrays["pred_cls"], arrays["target_cls"])
    return {int(cls): float(ap.mean()) for cls, ap in zip(values[6], values[5], strict=True)}


def native_ap(stats, ids, *, classes=None):
    """Descriptive observed-class AP, or a fixed-class AP requiring every target class."""
    values = class_ap(stats, ids)
    target = list(values) if classes is None else list(classes)
    if not target or any(cls not in values for cls in target):
        return None
    return float(np.mean([values[cls] for cls in target]))


def bootstrap_support(stats, pilot, *, min_class_groups=5):
    if type(min_class_groups) is not int or min_class_groups < 1:
        raise ValueError("Minimum class groups must be a positive integer")
    if set(stats) != {r["id"] for r in pilot["rows"]}:
        raise ValueError("Statistics must contain identical evaluation IDs")
    support = []
    for cls, name in enumerate(CLASSES):
        counts = {r["id"]: int(np.sum(stats[r["id"]]["target_cls"] == cls)) for r in pilot["rows"]}
        groups = {r["group"] for r in pilot["rows"] if counts[r["id"]]}
        support.append(
            {
                "class_id": cls,
                "class": name,
                "images": sum(v > 0 for v in counts.values()),
                "instances": sum(counts.values()),
                "source_groups": len(groups),
                "included": len(groups) >= min_class_groups,
                "reason": None if len(groups) >= min_class_groups else "insufficient source groups",
            }
        )
    included = [r["class_id"] for r in support if r["included"]]
    if not included:
        raise ValueError("No classes meet the fixed source-group support policy")
    return {
        "version": "fixed-supported-macro-v2",
        "class_ids": included,
        "class_names": [CLASSES[c] for c in included],
        "minimum_source_groups": min_class_groups,
        "support": support,
        "missing_class_draw": "exclude macro draw; no AP imputation or redraw",
        "population": "fixed supported classes; percentile CI conditional on their presence",
        "limit": "Support gate is an engineering minimum, not statistical adequacy",
    }


def paired_interval(before, after, pilot, *, resamples=100, seed=2026, min_class_groups=5):
    if set(before) != set(after) or set(before) != {r["id"] for r in pilot["rows"]}:
        raise ValueError("Paired statistics must contain identical evaluation IDs")
    if type(resamples) is not int or resamples < 2:
        raise ValueError("At least two bootstrap resamples required")
    for sample_id in before:
        if not np.array_equal(before[sample_id]["target_cls"], after[sample_id]["target_cls"]):
            raise ValueError("Paired ground truth changed")
    policy = bootstrap_support(before, pilot, min_class_groups=min_class_groups)
    target = policy["class_ids"]
    groups = {}
    for row in pilot["rows"]:
        groups.setdefault(row["group"], []).append(row["id"])
    keys = sorted(groups)
    rng, deltas, draws = np.random.default_rng(seed), [], []
    per_class_deltas = {c: [] for c in target}
    image_counts = {i: Counter(map(int, row["target_cls"])) for i, row in before.items()}
    for draw in range(resamples):
        sampled = rng.choice(keys, len(keys), replace=True)
        ids = [i for k in sampled for i in groups[k]]
        a, b = class_ap(before, ids), class_ap(after, ids)
        missing = [c for c in target if c not in a]
        changes = {c: b[c] - a[c] for c in target if c in a}
        for c, value in changes.items():
            per_class_deltas[c].append(value)
        recovery = None if missing else float(np.mean([changes[c] for c in target]))
        if recovery is not None:
            deltas.append(recovery)
        support = []
        for c in range(len(CLASSES)):
            support.append(
                {
                    "class_id": c,
                    "images": sum(image_counts[i][c] > 0 for i in ids),
                    "instances": sum(image_counts[i][c] for i in ids),
                    "distinct_source_groups": sum(
                        any(image_counts[i][c] for i in groups[k]) for k in set(sampled)
                    ),
                    "recovery": changes.get(c),
                }
            )
        draws.append(
            {
                "draw": draw,
                "valid": not missing,
                "missing_class_ids": missing,
                "recovery": recovery,
                "class_support": support,
            }
        )
    if len(deltas) < 2:
        raise ValueError("Fewer than two valid fixed-class draws; interval is not estimable")
    original_ids = list(before)
    original_a, original_b = class_ap(before, original_ids), class_ap(after, original_ids)
    per_class = []
    for entry in policy["support"]:
        c = entry["class_id"]
        values = per_class_deltas.get(c, [])
        bounds = np.quantile(values, [0.025, 0.975]) if len(values) >= 2 else (None, None)
        a, b = original_a.get(c), original_b.get(c)
        per_class.append(
            {
                **entry,
                "before_ap": a,
                "after_ap": b,
                "recovery": b - a if a is not None and b is not None else None,
                "ci_low": float(bounds[0]) if bounds[0] is not None else None,
                "ci_high": float(bounds[1]) if bounds[1] is not None else None,
                "valid_draws": len(values),
                "excluded_draws": resamples - len(values),
                "present_draws": sum(d["class_support"][c]["images"] > 0 for d in draws),
            }
        )
    low, high = np.quantile(deltas, [0.025, 0.975])
    return {
        "estimand": policy,
        "before_ap": float(np.mean([original_a[c] for c in target])),
        "after_ap": float(np.mean([original_b[c] for c in target])),
        "recovery": float(np.mean([original_b[c] - original_a[c] for c in target])),
        "recovery_ci_low": float(low),
        "recovery_ci_high": float(high),
        "resamples": resamples,
        "valid_draws": len(deltas),
        "invalid_draws": resamples - len(deltas),
        "draws": draws,
        "per_class": per_class,
        "groups": len(keys),
        "interval": "fixed macro AP; conditional 95% group CI; no multiplicity correction",
    }


class MatchingStats:
    """Save native per-image matches once for downstream analysis without inference."""

    def init_metrics(self, model):
        self.image_ids = []
        super().init_metrics(model)

    def update_metrics(self, preds, batch):
        self.image_ids.extend(Path(p).stem for p in batch["im_file"])
        super().update_metrics(preds, batch)

    def get_stats(self):
        np.savez_compressed(
            self.save_dir / "paired_stats.npz",
            ids=np.array(self.image_ids),
            **{
                f"{key}_{i}": value
                for key in STAT_KEYS
                for i, value in enumerate(self.metrics.stats[key])
            },
        )
        return super().get_stats()
