"""Native AP aggregation and paired source-group bootstrap for saved matching statistics."""

from pathlib import Path

import numpy as np
from ultralytics.utils.metrics import ap_per_class

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


def class_ap(stats, ids, *, detailed=False):
    """Native AP by represented class; optional AP50, precision and recall for reporting."""
    if not ids:
        return {}
    arrays = {k: np.concatenate([stats[i][k] for i in ids], axis=0) for k in STAT_KEYS}
    if not len(arrays["target_cls"]):
        return {}
    values = ap_per_class(arrays["tp"], arrays["conf"], arrays["pred_cls"], arrays["target_cls"])
    if detailed:
        return {
            int(cls): {
                "AP50": float(ap[0]),
                "AP50_95": float(ap.mean()),
                "precision": float(p),
                "recall": float(r),
            }
            for cls, ap, p, r in zip(values[6], values[5], values[2], values[3], strict=True)
        }
    return {int(cls): float(ap.mean()) for cls, ap in zip(values[6], values[5], strict=True)}


def paired_bootstrap(before, after, groups, classes, *, resamples=200, seed=2026):
    """Macro AP50:95 change with a 95% percentile CI, resampling whole source groups.

    Draws missing any fixed class are excluded from the macro interval (never imputed). The
    interval is conditional on the two trained models: it has no seed/selection uncertainty.
    """
    if set(before) != set(after) or set(before) != set(groups):
        raise ValueError("Paired statistics must contain identical evaluation IDs")
    for sample_id in before:
        if not np.array_equal(before[sample_id]["target_cls"], after[sample_id]["target_cls"]):
            raise ValueError("Paired ground truth changed")
    members = {}
    for sample_id, group in groups.items():
        members.setdefault(group, []).append(sample_id)
    keys = sorted(members)
    rng = np.random.default_rng(seed)
    macro, per_class = [], {c: [] for c in classes}
    for _ in range(resamples):
        ids = [i for k in rng.choice(keys, len(keys), replace=True) for i in members[k]]
        a, b = class_ap(before, ids), class_ap(after, ids)
        for c in classes:
            if c in a and c in b:
                per_class[c].append(b[c] - a[c])
        if all(c in a and c in b for c in classes):
            macro.append(float(np.mean([b[c] - a[c] for c in classes])))
    a, b = class_ap(before, list(before)), class_ap(after, list(after))

    def interval(values):
        return [float(v) for v in np.quantile(values, [0.025, 0.975])] if len(values) > 1 else None

    return {
        "delta": float(np.mean([b[c] - a[c] for c in classes])),
        "ci95": interval(macro),
        "valid_draws": len(macro),
        "resamples": resamples,
        "groups": len(keys),
        "classes": {
            c: {"delta": b.get(c, np.nan) - a.get(c, np.nan), "ci95": interval(per_class[c])}
            for c in classes
        },
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


def factorial_bootstrap(arms, groups, classes, *, policy, resamples=500, seed=2026):
    """Paired cluster bootstrap of six 2x2 contrasts, with common group draws.

    A = raw/raw, B = raw/policy, C = policy/raw, D = policy/policy.
    Draw complete source groups with replacement and evaluate *all four arms*
    on the identical resampled image IDs. Intervals are conditional on the
    checkpoints/training seed; no multiple-comparison correction is implied.
    """
    if type(resamples) is not int or resamples < 2:
        raise ValueError("Need at least two bootstrap draws")
    if policy not in ("v1", "v2", "v3"):
        raise ValueError("Unknown policy")
    keys = ("v0/v0", f"v0/{policy}", f"{policy}/v0", f"{policy}/{policy}")
    if set(arms) != set(keys):
        raise ValueError("Bootstrap requires the complete policy 2x2")
    ids = set(groups)
    if not ids or any(set(stats) != ids for stats in arms.values()):
        raise ValueError("Every arm must use the same development IDs/source groups")
    reference = arms[keys[0]]
    for stats in arms.values():
        for sample_id in ids:
            if not np.array_equal(reference[sample_id]["target_cls"], stats[sample_id]["target_cls"]):
                raise ValueError("Paired ground truth changed")

    class_ids = tuple(classes)
    if not class_ids or len(set(class_ids)) != len(class_ids):
        raise ValueError("Require unique, nonempty class IDs")

    def effects(values):
        a, b, c, d = values
        return {
            "inference_on_raw_model": b - a,
            "training_on_processed_input": d - b,
            "training_on_raw_input": c - a,
            "inference_on_policy_model": d - c,
            "joint_pipeline": d - a,
            "interaction": (d - c) - (b - a),
        }

    whole = [class_ap(arms[key], sorted(ids)) for key in keys]
    if any(any(cls not in arm for cls in class_ids) for arm in whole):
        raise ValueError("A complete development evaluation must contain all five classes")
    names = tuple(effects([0.0] * 4))
    samples = {name: [] for name in names}
    class_samples = {cls: {name: [] for name in names} for cls in class_ids}

    members = {}
    for sample_id, group in groups.items():
        members.setdefault(group, []).append(sample_id)
    group_keys = sorted(members)
    rng = np.random.default_rng(seed)
    for _ in range(resamples):
        sampled_ids = [
            item
            for group in rng.choice(group_keys, len(group_keys), replace=True)
            for item in members[group]
        ]
        values = [class_ap(arms[key], sampled_ids) for key in keys]
        for cls in class_ids:
            if all(cls in arm for arm in values):
                for name, delta in effects([arm[cls] for arm in values]).items():
                    class_samples[cls][name].append(float(delta))
        if all(all(cls in arm for cls in class_ids) for arm in values):
            means = [float(np.mean([arm[cls] for cls in class_ids])) for arm in values]
            for name, delta in effects(means).items():
                samples[name].append(float(delta))

    def summarize(delta, draws):
        return {
            "delta": float(delta),
            "ci95": [float(x) for x in np.quantile(draws, [0.025, 0.975])]
            if len(draws) > 1 else None,
            "valid_draws": len(draws),
            "resamples": resamples,
        }

    mean_effects = effects([np.mean([arm[cls] for cls in class_ids]) for arm in whole])
    contrasts = {name: summarize(delta, samples[name]) for name, delta in mean_effects.items()}
    by_class = {
        cls: {
            name: summarize(delta, class_samples[cls][name])
            for name, delta in effects([arm[cls] for arm in whole]).items()
        }
        for cls in class_ids
    }
    return {
        "contrasts": contrasts,
        "per_class": by_class,
        "groups": len(group_keys),
        "seed": seed,
        "method": "paired percentile 95% source-group cluster bootstrap",
        "limitations": "single trained seed; no checkpoint-selection or multiplicity uncertainty",
    }
