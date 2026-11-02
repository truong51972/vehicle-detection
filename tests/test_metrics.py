import numpy as np
import pytest

from vehicle_detection.metrics import average_precision, evaluate, evaluate_matches, match


def test_perfect_and_missing_detection():
    perfect = {"boxes": [[0, 0, 0, 10, 10]], "predictions": [[0, 0, 0, 10, 10, 0.9]]}
    result = evaluate([perfect])
    assert result["map50"] == result["map50_95"] == result["precision"] == result["recall"] == 1
    assert result["per_class"]["truck"]["map50_95"] is None
    assert evaluate([{**perfect, "predictions": []}])["map50_95"] == 0
    assert evaluate([{"boxes": [], "predictions": []}])["map50_95"] is None


def test_false_positive_order_and_duplicates():
    record = {
        "boxes": [[0, 0, 0, 10, 10]],
        "predictions": [
            [0, 20, 20, 30, 30, 0.99],
            [0, 0, 0, 10, 10, 0.9],
            [0, 0, 0, 10, 10, 0.8],
        ],
    }
    result = evaluate([record])
    assert result["map50_95"] == pytest.approx(0.5)
    assert result["precision"] == pytest.approx(1 / 3)
    assert result["recall"] == 1
    _, correct, _ = match(record["boxes"], record["predictions"])
    assert correct[:, 0].tolist() == [False, True, False]


def test_iou_sweep_and_class_isolation():
    record = {"boxes": [[0, 0, 0, 10, 10]], "predictions": [[0, 0, 0, 6, 10, 0.9]]}
    result = evaluate([record])
    assert result["map50"] == 1
    assert 0 < result["map50_95"] < 1
    record["predictions"][0][0] = 1
    assert evaluate([record])["map50"] == 0


def test_negative_images_count_false_positives_and_confidence():
    records = [
        {"boxes": [[0, 0, 0, 10, 10]], "predictions": [[0, 0, 0, 10, 10, 0.2]]},
        {"boxes": [], "predictions": [[0, 0, 0, 10, 10, 0.9]]},
    ]
    result = evaluate(records, 0.25)
    assert result["map50_95"] == 0.5
    assert result["precision"] == result["recall"] == 0
    assert average_precision(np.array([True, False, True]), 2) == pytest.approx(
        (51 + 50 * (2 / 3)) / 101
    )


def test_cached_matches_rank_globally_without_mutating_cache():
    records = [
        {"boxes": [[0, 0, 0, 10, 10]], "predictions": [[0, 0, 0, 10, 10, 0.5]]},
        {"boxes": [[0, 0, 0, 10, 10]], "predictions": [[0, 20, 20, 30, 30, 0.9]]},
    ]
    cached = [match(r["boxes"], r["predictions"]) for r in records]
    original = [tuple(array.copy() for array in item) for item in cached]
    result = evaluate_matches(cached)
    assert result == evaluate(records)
    assert result["map50_95"] == pytest.approx(51 * 0.5 / 101)
    assert result["map50_95"] != np.mean([evaluate([r])["map50_95"] for r in records])
    evaluate_matches([cached[0]], precision_confidence=0.6)
    for before, after in zip(original, cached, strict=True):
        for first, second in zip(before, after, strict=True):
            np.testing.assert_array_equal(first, second)
