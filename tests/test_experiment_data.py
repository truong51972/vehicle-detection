import shutil
from pathlib import Path

import pandas as pd
import pytest
import yaml

from vehicle_detection.common import file_hash
from vehicle_detection.experiment_data import (
    audit_sources,
    config_path,
    experiments,
    load_experiment,
    prepare_data,
    run_dir,
    shared_splits,
    verify_data,
)

DATE = "2026-11-09"


def test_shared_table_matches_frozen_memberships():
    shared, rows = shared_splits(Path.cwd())
    table = pd.DataFrame(rows)
    assert len(rows) == 79593 and shared["training_subset"] == "S20"
    assert table.s10.sum() == 4890 and table.s20.sum() == 9781
    assert table.loc[table.s20 == 1, "group"].nunique() == 5915
    assert table.role.value_counts().to_dict() == {
        "training_pool": 48903,
        "development_validation": 13975,
        "final_test": 9730,
        "calibration": 6985,
    }
    assert (table.loc[table.role == "final_test", "sha256"] == "").all()


@pytest.mark.parametrize("version", ["v0", "v1", "v2", "v3", "v4"])
def test_versions_own_full_native_args_and_relative_checkpoint(version):
    config = load_experiment(Path.cwd(), version)
    training = config["training"]
    assert training["freeze"] == 0 and training["optimizer"] == "SGD"
    assert training["batch"] == (16 if version in {"v4"} else 32)
    assert training["amp"] is True and training["imgsz"] == 640
    assert training["hsv_v"] == 0 and training["mosaic"] == 1.0
    assert not Path(config["checkpoint"]).is_absolute() and config["name"]
    assert config["preparation"]["workers"] >= 1
    base = {k: v for k, v in load_experiment(Path.cwd(), "v0").items() if k in {"training"}}
    expected = dict(base["training"])
    if version in {"v4"}:
        expected.update(batch=16, workers=0)
    assert training == expected


def test_versions_are_discovered_beside_their_notebooks():
    assert experiments(Path.cwd()) == ["v0", "v1", "v2", "v3", "v4"]
    assert config_path(Path.cwd(), "v2") == Path.cwd() / "notebooks/experiments/v2/v2.yaml"
    assert not list(Path("configs").glob("v*.yaml"))
    with pytest.raises(ValueError, match="experiment ID"):
        config_path(Path.cwd(), "../v0")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda c: c["training"].update(data="/abs/data.yaml"), "owned by the lifecycle"),
        (lambda c: c.update(checkpoint_sha256="pinned"), "SHA256"),
        (lambda c: c.update(checkpoint=str(Path("/abs/yolo26n.pt").resolve())), "project-relative"),
        (lambda c: c["preparation"].update(workers=0), "preparation.workers"),
        (lambda c: c.pop("name"), "display name"),
    ],
)
def test_config_rejects_invalid_version_settings(project, change, message):
    path = config_path(project, "v1")
    config = yaml.safe_load(path.read_text())
    change(config)
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match=message):
        load_experiment(project, "v1")


def test_run_dir_dates(project):
    assert run_dir(project, "v0", DATE).name == f"v0-{DATE}"
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        run_dir(project, "v0", "11-09-2026")
    with pytest.raises(FileNotFoundError, match="ALLOW_PREPARATION = True"):
        run_dir(project, "v0")
    for date in ("2026-01-02", "2026-11-09", "2025-12-31"):
        (project / f"runs/v0-{date}").mkdir(parents=True)
    (project / "runs/v0-latest").mkdir()
    assert run_dir(project, "v0").name == "v0-2026-11-09"
    assert run_dir(project, "v0", create=True).name.startswith("v0-")


def test_export_is_relative_deterministic_and_sealed(project, monkeypatch):
    from vehicle_detection import experiment_data

    real_hash = experiment_data.file_hash

    def guarded(path):
        assert "/images/100k/val/" not in str(path), "Final-test pixels opened"
        return real_hash(path)

    monkeypatch.setattr(experiment_data, "file_hash", guarded)
    audit_sources(project)
    prepared = prepare_data(project, "v3", DATE, workers=2)
    run = project / f"runs/v3-{DATE}"
    assert prepared["path"] == f"runs/v3-{DATE}/data"
    saved = yaml.safe_load((run / "data/data.yaml").read_text())
    assert "path" not in saved and saved["test"] == "images/test"
    text = "".join(p.read_text() for p in (run / "data/data.yaml", run / "data/metadata.csv"))
    assert str(project) not in text
    names = {
        s: sorted(p.stem for p in (run / f"data/images/{s}").iterdir())
        for s in ("train", "val", "test")
    }
    assert names["train"] == ["train_0", "train_1", "train_2"]
    assert names["val"] == ["train_3"] and names["test"] == ["train_4", "train_5"]
    assert (run / "data/labels/train/train_1.txt").read_text() == ""
    assert (run / "config.yaml").is_file()
    metadata = pd.read_csv(run / "data/metadata.csv", keep_default_na=False)
    assert set(metadata.split) == {"train", "val", "test"} and len(metadata) == 6

    again = project / "again"
    again.mkdir()
    for name in ("configs", "notebooks", "pyproject.toml"):
        source = project / name
        if source.is_dir():
            shutil.copytree(source, again / name)
        else:
            shutil.copy2(source, again / name)
    assert prepare_data(again, "v3", DATE, workers=1)["files_sha256"] == saved["files_sha256"]
    assert prepare_data(project, "v3", DATE)["files_sha256"] == saved["files_sha256"]


def test_tampering_and_policy_changes_fail(project):
    prepare_data(project, "v1", DATE, workers=2)
    run = project / f"runs/v1-{DATE}/data"
    label = run / "labels/test/train_5.txt"
    original = label.read_text()
    label.write_text(original + "0 0.5 0.5 0.1 0.1\n")
    with pytest.raises(ValueError, match="bytes changed"):
        verify_data(project, "v1")
    label.write_text(original)
    (run / "images/val/extra.jpg").write_bytes(b"x")
    with pytest.raises(ValueError, match="IDs differ"):
        verify_data(project, "v1")
    (run / "images/val/extra.jpg").unlink()
    config = yaml.safe_load(config_path(project, "v1").read_text())
    config["recipes"]["gamma"]["gamma"] = 0.8
    config["name"] = "Renamed"
    config_path(project, "v1").write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="new run date"):
        verify_data(project, "v1")


def test_changed_source_fails(project):
    shared = yaml.safe_load((project / "configs/data.yaml").read_text())
    source = Path(shared["data_root"]) / "images/100k/train/train_3.jpg"
    source.write_bytes(source.read_bytes() + b"\0")
    assert file_hash(source)
    with pytest.raises(ValueError, match="Source image changed"):
        prepare_data(project, "v0", DATE, workers=1)
    assert not (project / f"runs/v0-{DATE}/data").exists()
    assert not (project / f"runs/v0-{DATE}/data.tmp").exists()
