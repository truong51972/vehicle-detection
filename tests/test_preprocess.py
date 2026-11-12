import cv2
import numpy as np
import pytest
import yaml

from vehicle_detection.experiment_data import config_path
from vehicle_detection.preprocess import apply_recipe, local_contrast, quality, validate_recipes


def image(seed=0, scale=1.0, offset=0):
    noise = np.random.default_rng(seed).integers(0, 256, (90, 160, 3), dtype=np.uint8)
    base = cv2.GaussianBlur(noise, (0, 0), 3).astype(float)
    return np.clip(base * scale + offset, 0, 255).astype(np.uint8)


@pytest.mark.parametrize("version", ["v0", "v1", "v2", "v3", "v4"])
def test_every_version_policy_preserves_geometry_and_source(version):
    config = yaml.safe_load(config_path(".", version).read_text())
    validate_recipes(config["recipes"])
    for seed in range(5):
        source = image(seed, scale=0.3 + 0.15 * seed)
        original = source.copy()
        output, applied, scores = apply_recipe(source, config["variant"], config["recipes"])
        assert output.shape == source.shape and output.dtype == np.uint8
        assert np.array_equal(source, original)
        assert set(scores) >= {"brightness", "contrast", "noise", "black_fraction"}
        if not applied:
            assert np.array_equal(output, source)


def test_raw_and_skip_are_identity():
    source = image()
    assert apply_recipe(source, "raw", {"raw": {}})[0] is source
    recipes = yaml.safe_load(config_path(".", "v3").read_text())["recipes"]
    bright = np.full((64, 80, 3), 250, np.uint8)
    output, applied, _ = apply_recipe(bright, "adaptive_local_contrast", recipes)
    assert not applied and output is bright


def test_local_contrast_is_bounded_and_protects_extremes():
    source = image(1, scale=0.4, offset=60)
    output = local_contrast(source, clip_limit=1.5, grid_size=8, gain=0.5, max_offset=12)
    change = output.astype(int) - source.astype(int)
    assert np.abs(change).max() <= 12
    assert (change[:, :, 0] == change[:, :, 1]).mean() > 0.99
    dark = np.zeros((32, 32, 3), np.uint8)
    assert np.array_equal(local_contrast(dark, 1.5, 8, 0.5, 12), dark)


def test_adaptive_local_contrast_routes_low_contrast_images_only():
    recipes = yaml.safe_load(config_path(".", "v3").read_text())["recipes"]
    flat = image(2, scale=0.3, offset=80)
    assert quality(flat)["contrast"] < 0.2
    _, applied, _ = apply_recipe(flat, "adaptive_local_contrast", recipes)
    assert [a["name"] for a in applied] == ["local_contrast"]


@pytest.mark.parametrize(
    "recipes",
    [
        {"unknown": {}},
        {"gamma": {"gamma": 0}},
        {"local_contrast": {"clip_limit": 1.5, "grid_size": 8, "gain": 0.5, "max_offset": 1.5}},
        {
            "adaptive_local_contrast": dict.fromkeys(
                (
                    "contrast_below",
                    "brightness_above",
                    "brightness_below",
                    "noise_max",
                    "black_below",
                    "white_below",
                ),
                0.1,
            )
        },
    ],
)
def test_invalid_recipes_fail(recipes):
    with pytest.raises(ValueError):
        validate_recipes(recipes)
