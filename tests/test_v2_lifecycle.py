"""The experimental v2 arm keeps the exact failed-pilot pixels and shared settings."""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from vehicle_detection.experiment_data import load_experiment
from vehicle_detection.preprocess import apply_recipe, local_shadow, validate_recipes


@pytest.mark.parametrize(
    "shape,expected",
    [
        ((64, 80, 3), "be99e1be15a0595be20394a442949ce9daa9c3b6e06941431369bd69d046d4bf"),
        ((720, 1280, 3), "c1d98bc01b6a3aa31e494ba156a1f5d77f71dba472434d23d44690d6b853c343"),
    ],
)
def test_native_recipe_preserves_original_pilot_pixels(shape, expected):
    image = np.random.default_rng(2026).integers(0, 256, shape, dtype=np.uint8)
    config = load_experiment(Path.cwd(), "v2")
    output, decisions, _ = apply_recipe(image, config["variant"], config["recipes"])
    assert hashlib.sha256(output.tobytes()).hexdigest() == expected
    assert np.array_equal(output, local_shadow(image))
    assert decisions == [{"name": "local_shadow", "params": {}}]
    baseline = load_experiment(Path.cwd(), "v0")
    for key in ("checkpoint", "checkpoint_sha256", "seed", "jpeg_quality", "training"):
        assert config[key] == baseline[key]
    with pytest.raises(ValueError, match="parameters"):
        validate_recipes({"local_shadow": {"sigma": 8}})
