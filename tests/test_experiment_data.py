"""Tiny source fixtures exercise the active workflow, never scientific measurements."""

import json
import shutil
from pathlib import Path

import cv2
import pandas as pd
import pytest
import yaml
from test_study import bdd as bdd

from vehicle_detection.common import digest, file_hash
from vehicle_detection.data import scan_bdd
from vehicle_detection.experiment_data import (
    audit_sources,
    load_experiment,
    prepare_data,
    shared_splits,
    source_rows,
    verify_data,
)
from vehicle_detection.preprocess import apply_recipe
from vehicle_detection.protocol import group_key


@pytest.fixture
def experiment_project(bdd, tmp_path):
    project = tmp_path / "project"
    (project / "configs").mkdir(parents=True)
    (project / "pyproject.toml").touch()
    annotations = bdd / "bdd100k_labels_images_train.json"
    frames = json.loads(annotations.read_text())
    frames[1]["labels"] = [label for label in frames[1]["labels"] if label["category"] == "train"]
    annotations.write_text(json.dumps(frames))
    for i in (0, 4):
        import numpy as np

        image = np.random.default_rng(i).integers(0, 50, (64, 80, 3), dtype=np.uint8)
        assert cv2.imwrite(str(bdd / f"images/100k/train/train_{i}.jpg"), image)
    records, _, _ = scan_bdd(bdd)
    roles = [
        "training_pool",
        "training_pool",
        "calibration",
        "calibration",
        "development_validation",
        "development_validation",
    ]
    rows = []
    for r in records:
        role = (
            roles[int(r["id"].split("_")[-1])] if r["official_split"] == "train" else "final_test"
        )
        rows.append(
            {
                "id": r["id"],
                "group": group_key(r),
                "source_group": group_key(r),
                "role": role,
                "s10": int(role == "training_pool"),
                "source_path": str(Path(r["path"]).relative_to(bdd)),
                "width": 80,
                "height": 64,
                "sha256": file_hash(r["path"]) if role != "final_test" else "",
                "labels_sha256": digest(r["labels"]),
            }
        )
    pd.DataFrame(rows).to_csv(project / "configs/splits.csv.gz", index=False)
    shared = yaml.safe_load(Path("configs/data.yaml").read_text())
    shared.update(data_root=str(bdd), splits_sha256=file_hash(project / "configs/splits.csv.gz"))
    (project / "configs/data.yaml").write_text(yaml.safe_dump(shared))
    for version in ("v0", "v1"):
        shutil.copyfile(f"configs/{version}.yaml", project / f"configs/{version}.yaml")
    return project


def test_shared_table_matches_preserved_memberships_and_policy():
    shared, rows = shared_splits(Path.cwd())
    assert len(rows) == 79593
    assert sum(r["s10"] for r in rows) == 4890
    assert sum(r["role"] == "development_validation" for r in rows) == 13975
    old = pd.read_csv("docs/issue5/split_membership.csv.gz")
    table = pd.DataFrame(rows).set_index("id")
    for key in ("group", "source_group", "role"):
        assert table[key].to_dict() == old.set_index("id")[key].to_dict()
    policy = json.loads(Path("docs/issue6-v2/policy.json").read_text())
    assert load_experiment(Path.cwd(), "v1")["recipes"] == policy["recipes"]
    assert shared["source_protocol_id"] == policy["protocol_id"]


def test_offline_parity_determinism_reuse_and_sealed_holdout(experiment_project, monkeypatch):
    from vehicle_detection import experiment_data as data

    project = experiment_project
    real_read, real_hash = data.read_image, data.file_hash

    def read(path):
        assert "/val/" not in str(path) or "/runs/" in str(path), "Final pixels opened"
        return real_read(path)

    def hashed(path):
        assert "/val/" not in str(path) or "/runs/" in str(path), "Final pixels hashed"
        return real_hash(path)

    monkeypatch.setattr(data, "read_image", read)
    monkeypatch.setattr(data, "file_hash", hashed)
    assert audit_sources(project).sum() == 9
    _, rows = source_rows(project)
    for version in ("v0", "v1"):
        native = prepare_data(project, version)
        assert "test" not in native
        assert prepare_data(project, version) == native
        assert verify_data(project, version) == native
    raw, adaptive = (project / f"runs/{v}/data" for v in ("v0", "v1"))
    for label in (raw / "labels").rglob("*.txt"):
        assert label.read_bytes() == (adaptive / label.relative_to(raw)).read_bytes()
        assert len(label.read_text().splitlines()) == (0 if label.stem == "train_1" else 5)
    assert (raw / "labels/train/train_0.txt").read_text().splitlines()[0] == (
        "0 0.19375 0.328125 0.3625 0.59375"
    )
    # Same policy runs on original development pixels, then JPEG95 exactly once.
    config = load_experiment(project, "v1")
    shared, _ = shared_splits(project)
    for r in rows:
        if r["role"] != "development_validation":
            continue
        image = real_read(Path(shared["data_root"]) / r["source_path"])
        output, _, _ = apply_recipe(image, config["variant"], config["recipes"])
        ok, encoded = cv2.imencode(".jpg", output, [cv2.IMWRITE_JPEG_QUALITY, 95])
        assert ok and encoded.tobytes() == (adaptive / "images/val" / f"{r['id']}.jpg").read_bytes()
    before = native["files_sha256"]
    shutil.rmtree(adaptive)
    assert prepare_data(project, "v1")["files_sha256"] == before
    assert not list((project / "runs").rglob("images/test"))


def test_preparation_rejects_changed_data_policy_inventory_and_unknown_temp(experiment_project):
    root = experiment_project
    prepare_data(root, "v0")
    directory = root / "runs/v0/data"
    label = directory / "labels/train/train_0.txt"
    saved = label.read_bytes()
    label.write_text("")
    with pytest.raises(ValueError, match="bytes changed"):
        verify_data(root, "v0")
    label.write_bytes(saved)
    image = next((directory / "images/train").glob("*.jpg"))
    image.unlink()
    with pytest.raises(ValueError, match="IDs differ"):
        verify_data(root, "v0")
    (root / "runs/v1/data.tmp").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        prepare_data(root, "v1")
    config_path = root / "configs/v0.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["training"]["epochs"] += 1
    config_path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="inputs/policy changed"):
        prepare_data(root, "v0")


@pytest.mark.parametrize("change", ["group", "id", "s10", "source_path", "sha256"])
def test_invalid_shared_contract_fails_before_any_pixels(experiment_project, change):
    root = experiment_project
    path = root / "configs/splits.csv.gz"
    table = pd.read_csv(path, keep_default_na=False)
    if change in ("group", "id"):
        table.loc[0, change] = table.loc[len(table) - 1, change]
    elif change == "s10":
        table.loc[table.role == "final_test", "s10"] = 1
    elif change == "source_path":
        table.loc[0, change] = "../escape.jpg"
    else:
        table.loc[table.role == "final_test", "sha256"] = "opened"
    table.to_csv(path, index=False)
    shared_path = root / "configs/data.yaml"
    shared = yaml.safe_load(shared_path.read_text())
    shared["splits_sha256"] = file_hash(path)
    shared_path.write_text(yaml.safe_dump(shared))
    with pytest.raises(ValueError):
        shared_splits(root)


def test_source_tamper_and_symlink_fail_without_publishing_export(experiment_project):
    root = experiment_project
    shared, rows = shared_splits(root)
    source = Path(shared["data_root"]) / next(r["source_path"] for r in rows if r["s10"])
    original = source.read_bytes()
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Source image changed"):
        prepare_data(root, "v0")
    assert not (root / "runs/v0/data").exists()
    source.write_bytes(original)
    reserved = Path(shared["data_root"]) / next(
        r["source_path"] for r in rows if r["role"] == "final_test"
    )
    source.unlink()
    source.symlink_to(reserved)
    with pytest.raises(ValueError, match="path/symlink"):
        prepare_data(root, "v0")


def test_prepared_directory_symlinks_fail_before_reading_bytes(experiment_project):
    root = experiment_project
    prepare_data(root, "v0")
    labels = root / "runs/v0/data/labels/train"
    saved = labels.with_name("saved")
    labels.rename(saved)
    labels.symlink_to(saved, target_is_directory=True)
    with pytest.raises(ValueError, match="directory symlink"):
        verify_data(root, "v0")


@pytest.mark.parametrize("experiment", ["v0", "v1"])
def test_source_only_edit_preserves_frozen_export_reuse(
    experiment_project, monkeypatch, tmp_path, experiment
):
    from vehicle_detection import experiment_data as data

    operator = tmp_path / "preprocess.py"
    operator.write_bytes(Path(data.__file__).with_name("preprocess.py").read_bytes())
    monkeypatch.setattr(data, "__file__", str(tmp_path / "experiment_data.py"))
    root = experiment_project
    prepared = prepare_data(root, experiment)
    saved = root / f"runs/{experiment}/data/data.yaml"
    original_yaml = saved.read_bytes()
    operator.write_text(operator.read_text() + "\n# Unrelated future operator documentation.\n")
    assert file_hash(operator) != prepared["operator_sha256"]
    monkeypatch.setattr(data, "source_rows", lambda *args: pytest.fail("Source work on reuse"))
    assert verify_data(root, experiment) == prepared
    assert prepare_data(root, experiment) == prepared
    assert saved.read_bytes() == original_yaml
