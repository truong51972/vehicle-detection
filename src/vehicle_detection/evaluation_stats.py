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
