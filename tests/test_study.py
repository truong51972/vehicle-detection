import copy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from vehicle_detection.common import file_hash, write_json
from vehicle_detection.data import CLASSES, choose_splits, convert_labels, scan_bdd
from vehicle_detection.offline import export_variant, verify_alignment
from vehicle_detection.preprocess import apply_recipe, validate_recipes


@pytest.fixture
def recipes():
    return {
        "raw": {},
        "gamma": {"gamma": 0.65},
        "clahe": {"clip_limit": 2.0, "grid_size": 8},
        "denoise": {"strength": 5.0},
        "sharpen": {"amount": 0.5, "sigma": 1.0},
        "adaptive": {
            "brightness_below": 0.28,
            "contrast_below": 0.16,
            "noise_above": 0.035,
            "sharpness_min": 10.0,
            "sharpness_below": 100.0,
            "max_filters": 2,
        },
    }


@pytest.fixture
def bdd(tmp_path):
    root = tmp_path / "bdd"
    for split, n in [("train", 6), ("val", 3)]:
        directory = root / "images/100k" / split
        directory.mkdir(parents=True)
        frames = []
        for i in range(n):
            name = f"{split}_{i}.jpg"
            image = np.random.default_rng(i + (100 if split == "val" else 0)).integers(
                0, 256, (64, 80, 3), dtype=np.uint8
            )
            assert cv2.imwrite(str(directory / name), image)
            labels = [
                {
                    "id": j,
                    "category": c,
                    "box2d": {"x1": 1 + j, "y1": 2 + j, "x2": 30 + j, "y2": 40 + j},
                }
                for j, c in enumerate(["car", "truck", "bus", "train", "motor", "bike"])
            ]
            frames.append({"name": name, "attributes": {"weather": "clear"}, "labels": labels})
        (root / f"bdd100k_labels_images_{split}.json").write_text(json.dumps(frames))
    return root


def test_discovery_and_deterministic_disjoint_splits(bdd):
    records, audit, sources = scan_bdd(bdd)
    assert len(records) == 9 and audit["frames:train"] == 6
    assert sources["train"]["schema"] == "item"
    limits = {"train": 3, "tune": 2, "test": 2}
    first = choose_splits(records, 2026, limits, excluded_test_ids={"val_0"})
    second = choose_splits(list(reversed(records)), 2026, limits, excluded_test_ids={"val_0"})
    assert first == second
    assert all(r["official_split"] == "train" for s in ("train", "tune") for r in first[s])
    assert {r["id"] for r in first["test"]} == {"val_1", "val_2"}
    for split in ("train", "val"):
        path = bdd / f"bdd100k_labels_images_{split}.json"
        path.write_text(json.dumps({"frames": json.loads(path.read_text())}))
    assert scan_bdd(bdd)[2]["val"]["schema"] == "frames.item"


def test_duplicate_content_cannot_cross_splits(bdd):
    source = next((bdd / "images/100k/train").glob("*.jpg"))
    for path in (bdd / "images/100k/val").glob("*.jpg"):
        path.write_bytes(source.read_bytes())
    records, _, _ = scan_bdd(bdd)
    with pytest.raises(ValueError, match="leakage"):
        choose_splits(records, 42, {"train": None, "tune": None, "test": None})


def test_development_split_never_reads_holdout_pixels(bdd, monkeypatch, tmp_path, recipes):
    from vehicle_detection import data

    records, _, _ = scan_bdd(bdd)
    original_read = data.read_image
    original_hash = data.file_hash

    def train_only(path):
        assert "val" not in Path(path).parts, "Holdout pixels must remain untouched"
        return original_read(path)

    def train_hash(path):
        assert "val" not in Path(path).parts, "Holdout pixels must not be hashed"
        return original_hash(path)

    monkeypatch.setattr(data, "read_image", train_only)
    monkeypatch.setattr(data, "file_hash", train_hash)
    partitions = choose_splits(
        records, 2026, dict.fromkeys(["train", "tune", "test"]), prepare_test=False
    )
    assert set(partitions) == {"train", "tune"}
    assert sum(map(len, partitions.values())) == 6
    assert all("sha256" not in r for r in records if r["official_split"] == "val")
    converted = tmp_path / "converted"
    write_json(converted / "conversion.json", {"fixture": True})
    for split, rows in partitions.items():
        for r in rows:
            path = converted / "labels" / split / f"{r['id']}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("0 0.4 0.4 0.2 0.2\n")
    output = tmp_path / "raw_original"
    export_variant(partitions, converted, output, "raw_original", recipes)
    assert verify_alignment([output]) == 6
    assert not list((output / "images/test").iterdir())
    assert not list((output / "labels/test").iterdir())


@pytest.mark.parametrize("variant", ["raw", "gamma", "clahe", "denoise", "sharpen", "adaptive"])
def test_filter_geometry_dtype_and_source_invariance(variant, recipes):
    validate_recipes(recipes)
    image = np.full((32, 40, 3), 40, np.uint8)
    original = image.copy()
    result, decisions, _ = apply_recipe(image, variant, recipes)
    assert result.shape == image.shape and result.dtype == np.uint8
    np.testing.assert_array_equal(image, original)
    np.testing.assert_array_equal(result, apply_recipe(image, variant, recipes)[0])
    if variant == "adaptive":
        assert len(decisions) <= 2
        recipes["adaptive"]["max_filters"] = 0
        skipped, chosen, _ = apply_recipe(image, variant, recipes)
        assert chosen == []
        np.testing.assert_array_equal(skipped, image)


def test_bad_recipe_and_image_rejected(recipes):
    recipes["gamma"]["gamma"] = -1
    with pytest.raises(ValueError):
        validate_recipes(recipes)
    with pytest.raises(ValueError):
        apply_recipe(np.empty((0, 0, 3), np.uint8), "raw", recipes)


def test_native_conversion_six_classes_and_geometry(bdd, tmp_path):
    pytest.importorskip("bdd100k")
    pytest.importorskip("ultralytics")
    records, _, _ = scan_bdd(bdd)
    partitions = choose_splits(records, 42, {"train": 3, "tune": 2, "test": 2})
    output = tmp_path / "converted"
    details = convert_labels(partitions, output)
    assert details["versions"]["bdd100k"] == "1.0.1"
    assert details["yolo_category_ids"] == list(range(6))
    first = partitions["train"][0]
    labels = np.loadtxt(output / "labels/train" / f"{first['id']}.txt", ndmin=2)
    assert labels[:, 0].tolist() == list(range(6))
    np.testing.assert_allclose(labels[:, 3] * 80, 29, atol=0.001)
    np.testing.assert_allclose(labels[:, 4] * 64, 38, atol=0.001)
    # Border ties must not be reordered by the converter's six-significant-digit rounding.
    first["labels"].extend(
        [
            {"id": "left1", "category": "car", "box2d": {"x1": 0, "y1": 20, "x2": 50, "y2": 40}},
            {"id": "left2", "category": "car", "box2d": {"x1": 0, "y1": 3, "x2": 30, "y2": 20}},
            copy.deepcopy(first["labels"][0]),
        ]
    )
    convert_labels(partitions, tmp_path / "border-conversion")
    assert (
        len(np.loadtxt(tmp_path / "border-conversion/labels/train" / f"{first['id']}.txt", ndmin=2))
        == 8
    )


def test_export_restart_integrity_and_recipe_isolation(bdd, tmp_path, recipes):
    records, _, _ = scan_bdd(bdd)
    partitions = choose_splits(records, 42, {"train": 3, "tune": 2, "test": 2})
    source_file = tmp_path / "source_link.jpg"
    source_file.symlink_to(partitions["train"][0]["path"])
    partitions["train"][0]["path"] = str(source_file)
    source_directory = tmp_path / "source_directory"
    source_directory.symlink_to(
        Path(partitions["tune"][0]["path"]).parent, target_is_directory=True
    )
    partitions["tune"][0]["path"] = str(source_directory / Path(partitions["tune"][0]["path"]).name)
    converted = tmp_path / "converted"
    write_json(converted / "conversion.json", {"classes": CLASSES, "test": "fixture labels only"})
    for split, rows in partitions.items():
        for r in rows:
            path = converted / "labels" / split / f"{r['id']}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("0 0.4 0.4 0.2 0.2\n")
    roots = [tmp_path / "export" / v for v in ("raw_original", "gamma")]
    for variant, root in zip(("raw_original", "gamma"), roots, strict=True):
        first = export_variant(partitions, converted, root, variant, recipes)
        assert not first.reused.any()
        second = export_variant(partitions, converted, root, variant, recipes)
        assert second.reused.all()
        assert (
            all(Path(p).is_symlink() for p in second.image) if variant == "raw_original" else True
        )
    assert verify_alignment(roots) == 6
    journal = roots[0] / "metadata.jsonl"
    with journal.open("ab") as stream:
        stream.write(b'{"id":"interrupted')
    assert export_variant(partitions, converted, roots[0], "raw_original", recipes).reused.all()
    assert journal.read_bytes().endswith(b"\n")
    image = Path(first.iloc[0].image)
    image.write_bytes(b"broken generated image")
    with pytest.raises(ValueError, match="changed"):
        verify_alignment(roots)
    repaired = export_variant(partitions, converted, roots[1], "gamma", recipes)
    assert sum(~repaired.reused) == 1
    assert verify_alignment(roots) == 6
    altered = copy.deepcopy(recipes)
    altered["gamma"]["gamma"] = 0.8
    with pytest.raises(ValueError, match="recipe changed"):
        export_variant(partitions, converted, roots[1], "gamma", altered)
    yaml_path = roots[1] / "data.yaml"
    data = yaml.safe_load(yaml_path.read_text())
    data["val"] = "images/test"
    yaml_path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="YAML"):
        verify_alignment(roots)
    source = Path(partitions["train"][0]["path"])
    source.write_bytes(b"changed source")
    with pytest.raises(ValueError, match="Source image changed"):
        export_variant(partitions, converted, roots[0], "raw_original", recipes)


def test_raw_rejects_jpeg_that_native_validation_would_reencode(bdd, tmp_path, recipes):
    records, _, _ = scan_bdd(bdd)
    partitions = choose_splits(records, 42, {"train": 3, "tune": 2, "test": 2})
    record = partitions["train"][0]
    source = Path(record["path"])
    damaged = source.read_bytes()[:-2]
    source.write_bytes(damaged)
    record["sha256"] = file_hash(source)
    output = tmp_path / "raw_original"
    with pytest.raises(ValueError, match="JPEG end marker"):
        export_variant(partitions, tmp_path / "conversion", output, "raw_original", recipes)
    assert source.read_bytes() == damaged
    assert not output.exists()


def test_export_never_writes_through_source_symlink(bdd, tmp_path, recipes):
    records, _, _ = scan_bdd(bdd)
    partitions = choose_splits(records, 42, {"train": 3, "tune": 2, "test": 2})
    converted = tmp_path / "converted"
    write_json(converted / "conversion.json", {})
    for split, rows in partitions.items():
        for r in rows:
            path = converted / "labels" / split / f"{r['id']}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("")
    root = tmp_path / "gamma"
    data = export_variant(partitions, converted, root, "gamma", recipes)
    target = Path(data.iloc[0].image)
    source = Path(data.iloc[0].source)
    original = file_hash(source)
    target.unlink()
    target.symlink_to(source)
    with pytest.raises(ValueError, match="Unsafe export symlink"):
        export_variant(partitions, converted, root, "gamma", recipes)
    assert file_hash(source) == original
    target.unlink()
    temporary = target.with_suffix(".tmp.jpg")
    temporary.symlink_to(source)
    export_variant(partitions, converted, root, "gamma", recipes)
    assert not temporary.exists() and file_hash(source) == original


def test_jpeg_control_format_quality_and_light_verification(bdd, tmp_path, recipes, monkeypatch):
    from vehicle_detection import offline
    from vehicle_detection.common import read_image, read_json

    records, _, _ = scan_bdd(bdd)
    partitions = choose_splits(records, 42, {"train": 3, "tune": 2, "test": 2})
    converted = tmp_path / "converted"
    write_json(converted / "conversion.json", {"classes": CLASSES})
    for split, rows in partitions.items():
        for r in rows:
            label = converted / "labels" / split / f"{r['id']}.txt"
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text("0 .4 .4 .2 .2\n")
    variants = ["raw_original", "raw_reencoded", "gamma"]
    roots = [tmp_path / v for v in variants]
    tables = [
        export_variant(partitions, converted, p, v, recipes)
        for p, v in zip(roots, variants, strict=True)
    ]
    for v, table in zip(variants, tables, strict=True):
        image = Path(table.iloc[0].image)
        source = Path(table.iloc[0].source)
        assert image.suffix == ".jpg"
        assert read_image(image).shape == read_image(source).shape
        if v == "raw_original":
            assert image.is_symlink() and file_hash(image) == file_hash(source)
        if v == "raw_reencoded":
            ok, expected = cv2.imencode(".jpg", read_image(source), [cv2.IMWRITE_JPEG_QUALITY, 95])
            assert ok and image.read_bytes() == expected.tobytes()
            assert table.iloc[0].decisions == []
    assert verify_alignment(roots) == 6
    assert verify_alignment(roots, deep=False) == 6
    for root in roots:
        assert export_variant(partitions, converted, root, root.name, recipes).reused.all()
    assert verify_alignment(roots) == 6
    for settings in [{"jpeg_quality": 80}, {"output_format": "png"}]:
        with pytest.raises(ValueError, match="changed"):
            export_variant(partitions, converted, roots[2], "gamma", recipes, **settings)
    png = export_variant(
        partitions, converted, tmp_path / "png", "gamma", recipes, output_format="png"
    )
    assert Path(png.iloc[0].image).suffix == ".png"
    assert read_json(tmp_path / "png/export.json")["encoding"]["format"] == "png"
    original_hash = file_hash

    def only_manifests(path):
        assert Path(path).suffix in {".json", ".csv"}, "light verification read dataset bytes"
        return original_hash(path)

    monkeypatch.setattr(offline, "file_hash", only_manifests)
    monkeypatch.setattr(
        offline, "read_image", lambda p: pytest.fail("light verification decoded an image")
    )
    assert verify_alignment(roots, deep=False) == 6
    identity = read_json(roots[2] / "export.json")
    altered = copy.deepcopy(identity)
    altered["encoding"]["jpeg_quality"] = 80
    write_json(roots[2] / "export.json", altered)
    with pytest.raises(ValueError, match="identity changed"):
        verify_alignment(roots, deep=False)
    write_json(roots[2] / "export.json", identity)
    Path(tables[2].iloc[0].image).unlink()
    with pytest.raises(ValueError, match="ID mismatch"):
        verify_alignment(roots, deep=False)
    # Original links must fail the cheap check when their source disappears, too.
    Path(tables[0].iloc[0].source).unlink()
    with pytest.raises(ValueError, match="Missing exported image/label"):
        verify_alignment([roots[0]], deep=False)


def test_audit_and_disk_preflight(bdd, recipes):
    from vehicle_detection.data import audit_dataset
    from vehicle_detection.offline import estimate_disk

    records, _, _ = scan_bdd(bdd)
    assert audit_dataset(records)["error_count"] == 0
    bad = copy.deepcopy(records)
    bad[0]["labels"].extend(
        [
            {"box2d": {"x1": 0, "y1": 0, "x2": 100, "y2": 20}},
            {"box2d": {"x1": 5, "y1": 0, "x2": 1, "y2": 20}},
            {"box2d": {"x1": 0, "y1": 0, "x2": float("nan"), "y2": 20}},
            {"box2d": {}},
        ]
    )
    bad[1]["path"] = "missing.jpg"
    result = audit_dataset(bad)
    assert result["error_count"] == 5
    assert result["errors"]["out_of_bounds_box"] == 1
    assert result["examples"]["out_of_bounds_box"] == [bad[0]["id"]]
    assert bad[0]["labels"][-4]["box2d"]["x2"] == 100
    estimates = estimate_disk(records[:2], 100, ["raw_original", "raw_reencoded", "gamma"], recipes)
    assert estimates.loc[estimates.variant == "raw_original", "estimated_GiB"].item() == 0
    assert (estimates.loc[estimates.variant != "raw_original", "estimated_GiB"] > 0).all()
    assert estimates.sample_images.tolist() == [2, 2, 2]
