import copy

import numpy as np
import pytest

from vehicle_detection.config import load_config, validate
from vehicle_detection.preprocess import process, quality, route, stress


@pytest.fixture
def image():
    return np.random.default_rng(3).integers(0, 256, (64, 80, 3), dtype=np.uint8)


@pytest.mark.parametrize("treatment", ["raw", "gamma", "clahe", "denoise", "sharpen", "adaptive"])
def test_operator_invariants(image, treatment, config):
    original = image.copy()
    output, scores, chosen = process(image, treatment, config)
    assert output.shape == image.shape
    assert output.dtype == np.uint8
    np.testing.assert_array_equal(image, original)
    repeated, _, _ = process(image, treatment, config)
    np.testing.assert_array_equal(output, repeated)
    if treatment == "raw":
        np.testing.assert_array_equal(output, image)
        assert chosen == []
    if treatment == "adaptive":
        assert scores == quality(image)
        assert len(chosen) <= 2


def test_quality_and_gamma(config):
    image = np.full((32, 32, 3), 40, np.uint8)
    scores = quality(image)
    assert scores["brightness"] == pytest.approx(40 / 255)
    assert scores["contrast"] == scores["sharpness"] == scores["noise"] == 0
    output, _, _ = process(image, "gamma", config)
    assert output.mean() > image.mean()


def test_routing_order_bounds_disable_and_skip(config):
    good = {"brightness": 0.5, "contrast": 0.3, "noise": 0, "sharpness": 200}
    assert route(good, config["operators"], config["policy"]) == []
    dark = {**good, "brightness": 0.1, "contrast": 0.01}
    chosen = route(dark, config["operators"], config["policy"])
    assert [op["name"] for op in chosen] == ["gamma", "clahe"]
    config["operators"].reverse()
    assert [op["name"] for op in route(dark, config["operators"], config["policy"])] == [
        "clahe",
        "gamma",
    ]
    config["operators"][-1]["enabled"] = False
    assert [op["name"] for op in route(dark, config["operators"], config["policy"])] == ["clahe"]
    config["policy"]["max_filters"] = 0
    assert route(dark, config["operators"], config["policy"]) == []


def test_noise_prevents_sharpening(config):
    scores = {"brightness": 0.5, "contrast": 0.3, "noise": 0.2, "sharpness": 50}
    assert [op["name"] for op in route(scores, config["operators"], config["policy"])] == [
        "denoise"
    ]
    scores["noise"] = 0
    assert [op["name"] for op in route(scores, config["operators"], config["policy"])] == [
        "sharpen"
    ]
    scores["sharpness"] = 0
    assert route(scores, config["operators"], config["policy"]) == []


@pytest.mark.parametrize(
    "bad",
    [None, np.zeros((0, 0, 3), np.uint8), np.zeros((4, 4), np.uint8), np.zeros((4, 4, 3), float)],
)
def test_invalid_image_rejected(bad, config):
    with pytest.raises(ValueError, match="BGR uint8"):
        process(bad, "adaptive", config)


@pytest.mark.parametrize("condition", ["clean", "dark", "blur", "noise", "haze"])
def test_paired_stress_reproducibility(image, condition, config):
    original = image.copy()
    first = stress(image, condition, config["stress"], 42, "image1")
    second = stress(image, condition, config["stress"], 42, "image1")
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(image, original)
    assert first.shape == image.shape and first.dtype == np.uint8
    if condition == "noise":
        assert not np.array_equal(first, stress(image, condition, config["stress"], 43, "image1"))
    if condition == "clean":
        np.testing.assert_array_equal(first, image)


@pytest.mark.parametrize(
    "change",
    [
        {"seed": -1},
        {"policy": {"max_filters": 3}},
        {"benchmark": {"treatments": ["adaptive"]}},
        {"benchmark": {"conditions": ["clean", "clean"]}},
        {"training": {"batch": 1.5}},
        {"training": {"imgsz": 33}},
        {"unknown": 1},
        {"policy": {"noise_above": float("nan")}},
    ],
)
def test_config_rejects_invalid_settings(change):
    with pytest.raises(ValueError):
        load_config(overrides=change)


def test_operator_params_rejected(config):
    invalid = copy.deepcopy(config)
    invalid["operators"][0]["params"]["gamma"] = -1
    with pytest.raises(ValueError, match="gamma.gamma"):
        validate(invalid)
    assert load_config("full")["data"]["train_limit"] is None
