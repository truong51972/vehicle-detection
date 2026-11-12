"""Scientific invariants of the v4 operator and its offline data lifecycle."""

import shutil
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from vehicle_detection.experiment_data import load_experiment, prepare_data
from vehicle_detection.preprocess import (
    apply_recipe,
    canny_detail,
    canny_edges,
    routes_canny_detail,
    validate_recipes,
)


def scene():
    image = np.full((96, 160, 3), (65, 95, 125), np.uint8)
    cv2.rectangle(image, (45, 25), (110, 70), (155, 185, 215), -1)
    return image


def recipe():
    return load_experiment(Path.cwd(), "v4")["recipes"]


def test_canny_detail_is_local_bounded_deterministic_and_keeps_source():
    source = scene()
    original = source.copy()
    recipes = recipe()
    output, applied, scores = apply_recipe(source, "adaptive_canny_detail", recipes)
    assert [action["name"] for action in applied] == ["canny_detail"]
    assert 0 < scores["edge_density"] < 0.2
    assert np.array_equal(source, original)
    assert output.dtype == np.uint8 and output.shape == source.shape
    assert np.array_equal(output, apply_recipe(source, "adaptive_canny_detail", recipes)[0])
    delta = output.astype(int) - source.astype(int)
    assert 0 < np.abs(delta).max() <= 12
    assert np.array_equal(delta[:, :, 0], delta[:, :, 1])
    assert np.array_equal(delta[:, :, 1], delta[:, :, 2])
    assert np.array_equal(output[:15], source[:15])
    edges = canny_edges(source, 50, 150, 1.0)
    assert set(np.unique(edges)) == {0, 255}
    assert np.array_equal(output, canny_detail(source, **recipes["canny_detail"], edges=edges))


@pytest.mark.parametrize("value", [0, 100, 255])
def test_flat_and_extreme_images_skip_exactly(value):
    source = np.full((64, 80, 3), value, np.uint8)
    output, applied, scores = apply_recipe(source, "adaptive_canny_detail", recipe())
    assert output is source and applied == [] and scores["edge_density"] == 0


def test_all_channel_values_remain_bounded_including_saturation():
    source = np.random.default_rng(2026).integers(0, 256, (120, 160, 3), dtype=np.uint8)
    output = canny_detail(source, **recipe()["canny_detail"])
    assert np.abs(output.astype(int) - source.astype(int)).max() <= 12
    for shape in ((2, 2, 3), (2, 30, 3), (30, 2, 3)):
        tiny = np.full(shape, 128, np.uint8)
        assert canny_detail(tiny, **recipe()["canny_detail"]).shape == shape


def test_routes_reject_noisy_or_crowded_edges_and_strict_boundaries():
    settings = recipe()["adaptive_canny_detail"]
    scores = {"brightness": 0.5, "noise": 0.0, "edge_density": 0.05}
    assert routes_canny_detail(scores, settings)
    for feature, value in (("noise", 0.03), ("edge_density", 0.20), ("brightness", 0.04)):
        assert not routes_canny_detail({**scores, feature: value}, settings)


@pytest.mark.parametrize(
    ("name", "key", "value"),
    [
        ("canny_detail", "low_threshold", 151),
        ("canny_detail", "high_threshold", 50),
        ("canny_detail", "mask_radius", 1.5),
        ("canny_detail", "max_offset", True),
        ("canny_detail", "blur_sigma", 0),
        ("canny_detail", "amount", float("nan")),
        ("adaptive_canny_detail", "edge_density_above", 0.3),
        ("adaptive_canny_detail", "brightness_above", 0.99),
    ],
)
def test_invalid_canny_parameters_fail_before_export(name, key, value):
    recipes = deepcopy(recipe())
    recipes[name][key] = value
    with pytest.raises(ValueError):
        validate_recipes(recipes)


def test_adaptive_canny_requires_operator():
    recipes = recipe()
    del recipes["canny_detail"]
    with pytest.raises(ValueError, match="requires"):
        validate_recipes(recipes)


def test_v4_export_preserves_raw_gt_and_skip_bytes(project, monkeypatch):
    from vehicle_detection import experiment_data

    real_read = experiment_data.read_image

    def sealed_read(path):
        assert "/images/100k/val/" not in Path(path).as_posix(), "Final holdout was opened"
        return real_read(path)

    monkeypatch.setattr(experiment_data, "read_image", sealed_read)
    date = "2026-11-10"
    for version in ("v0", "v4"):
        prepare_data(project, version, date, workers=1)
    raw = project / f"runs/v0-{date}/data"
    processed = project / f"runs/v4-{date}/data"
    raw_labels = {
        p.relative_to(raw / "labels"): p.read_bytes() for p in (raw / "labels").rglob("*.txt")
    }
    labels = {
        p.relative_to(processed / "labels"): p.read_bytes()
        for p in (processed / "labels").rglob("*.txt")
    }
    assert labels == raw_labels
    metadata = pd.read_csv(processed / "metadata.csv", keep_default_na=False)
    skipped = metadata.loc[metadata.applied == ""]
    assert not skipped.empty
    for row in skipped.itertuples():
        relative = Path("images") / row.split / f"{row.id}.jpg"
        assert (processed / relative).read_bytes() == (raw / relative).read_bytes()


def test_v4_preview_and_preparation_run_in_fresh_kernels(project):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    source = Path("notebooks/experiments/v4")
    destination = project / source
    shutil.copytree(
        source, destination, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.yaml")
    )

    def run(name, overrides):
        notebook = nbformat.read(destination / name, as_version=4)
        config_cell = next(c for c in notebook.cells if c.cell_type == "code")
        config_cell.source += "\n" + overrides
        guard = """import builtins
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'ultralytics'}:
        raise AssertionError('Preview/export imported a training or inference engine')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
"""
        notebook.cells.insert(1, nbformat.v4.new_code_cell(guard))
        nbclient.NotebookClient(
            notebook,
            timeout=120,
            kernel_name="python3",
            resources={"metadata": {"path": str(destination)}},
        ).execute()

    run("01_experiments.ipynb", "SAMPLES = 64\nSHOW = 1")
    assert not (project / "runs").exists()
    run("02_train_prepare.ipynb", "RUN_DATE = '2026-11-10'\nALLOW_PREPARATION = False")
    assert not (project / "runs").exists()
    run("02_train_prepare.ipynb", "RUN_DATE = '2026-11-10'\nALLOW_PREPARATION = True")
    run("02_train_prepare.ipynb", "RUN_DATE = '2026-11-10'\nALLOW_PREPARATION = False")
    assert (project / "runs/v4-2026-11-10/data/data.yaml").is_file()
    assert not (project / "runs/v4-2026-11-10/train").exists()
