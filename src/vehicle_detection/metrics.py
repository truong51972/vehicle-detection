"""Classwise confidence-ordered matching and 101-point interpolated AP.

No area/crowd weighting: this is a BDD vehicle metric, not official COCO evaluation.
"""

import numpy as np

from .data import CLASSES

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
METRIC_ID = "bdd-vehicle-confidence-greedy-101point-v1"


def iou(box, targets):
    targets = np.asarray(targets, dtype=float).reshape(-1, 4)
    intersection = np.maximum(
        0, np.minimum(box[2:], targets[:, 2:]) - np.maximum(box[:2], targets[:, :2])
    ).prod(axis=1)
    union = np.prod(box[2:] - box[:2]) + np.prod(targets[:, 2:] - targets[:, :2], axis=1)
    return intersection / np.maximum(union - intersection, 1e-12)


def match(boxes, predictions):
    """Predictions: [class, x1, y1, x2, y2, confidence]; boxes omit confidence."""
    target = np.asarray(boxes, dtype=float).reshape(-1, 5)
    predicted = np.asarray(predictions, dtype=float).reshape(-1, 6)
    predicted = predicted[np.argsort(-predicted[:, 5], kind="stable")]
    correct = np.zeros((len(predicted), len(IOU_THRESHOLDS)), dtype=bool)
    for ti, threshold in enumerate(IOU_THRESHOLDS):
        used = set()
        for pi, pred in enumerate(predicted):
            candidates = np.flatnonzero(target[:, 0] == pred[0])
            candidates = np.array([j for j in candidates if j not in used], dtype=int)
            if not len(candidates):
                continue
            overlaps = iou(pred[1:5], target[candidates, 1:5])
            best = int(np.argmax(overlaps))
            if overlaps[best] >= threshold:
                used.add(int(candidates[best]))
                correct[pi, ti] = True
    return predicted, correct, target


def average_precision(tp, target_count):
    if not len(tp):
        return 0.0
    cumulative = np.cumsum(tp)
    recall = cumulative / target_count
    precision = cumulative / np.arange(1, len(tp) + 1)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    indices = np.searchsorted(recall, np.linspace(0, 1, 101), side="left")
    return float(
        np.mean(
            np.where(indices < len(envelope), envelope[np.minimum(indices, len(envelope) - 1)], 0)
        )
    )


def evaluate(records, precision_confidence=0.25):
    return evaluate_matches(
        [match(r["boxes"], r["predictions"]) for r in records], precision_confidence
    )


def evaluate_matches(matches, precision_confidence=0.25):
    """Aggregate cached per-image matches; AP still ranks predictions across images."""
    predicted = np.concatenate([m[0] for m in matches]) if matches else np.empty((0, 6))
    correct = np.concatenate([m[1] for m in matches]) if matches else np.empty((0, 10), bool)
    target = np.concatenate([m[2] for m in matches]) if matches else np.empty((0, 5))
    order = np.argsort(-predicted[:, 5], kind="stable")
    predicted, correct = predicted[order], correct[order]
    per_class = {}
    for cls, name in enumerate(CLASSES):
        count = int(np.sum(target[:, 0] == cls))
        selection = predicted[:, 0] == cls
        accepted = selection & (predicted[:, 5] >= precision_confidence)
        true_positives = int(np.sum(correct[accepted, 0]))
        detected = int(np.sum(accepted))
        ap = [average_precision(correct[selection, ti], count) for ti in range(10)] if count else []
        per_class[name] = {
            "targets": count,
            "predictions": int(selection.sum()),
            "map50": ap[0] if count else None,
            "map50_95": float(np.mean(ap)) if count else None,
            "precision": true_positives / detected if detected else 0.0,
            "recall": true_positives / count if count else None,
        }
    supported = [row for row in per_class.values() if row["targets"]]
    return {
        "images": len(matches),
        "targets": len(target),
        "per_class": per_class,
        **{
            key: float(np.mean([r[key] for r in supported])) if supported else None
            for key in ("map50", "map50_95", "precision", "recall")
        },
    }
