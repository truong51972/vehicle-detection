from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

pytest.importorskip("ultralytics")

from vehicle_detection import training  # noqa: E402
from vehicle_detection.common import file_hash, read_json  # noqa: E402
from vehicle_detection.data import FIVE_CLASSES  # noqa: E402
from vehicle_detection.experiment_data import config_path, prepare_data  # noqa: E402

DATE = "2026-11-09"


@pytest.fixture
def trained(project, monkeypatch):
    checkpoint = project / "weights/yolo26n.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"synthetic initial checkpoint")
    calls = []

    class Native:
        def __init__(self, path):
            self.path = path
            self.names = {i: "other" for i in range(80)} | training.COCO_VEHICLES
            if str(path).endswith("last.pt"):
                self.names = dict(enumerate(FIVE_CLASSES))
                args = yaml.safe_load((Path(path).parents[1] / "args.yaml").read_text())
                self.ckpt = {"train_args": args}

        def train(self, **kwargs):
            calls.append((self.path, kwargs))
            if kwargs.get("resume"):
                directory = Path(self.path).parents[1]
                args = yaml.safe_load((directory / "args.yaml").read_text())
            else:
                directory = Path(kwargs["project"]) / kwargs["name"]
                directory.mkdir(parents=True)
                args = kwargs
                (directory / "args.yaml").write_text(yaml.safe_dump(args))
                (directory / "weights").mkdir()
            for name in ("best.pt", "last.pt"):
                (directory / "weights" / name).write_bytes(directory.parent.name.encode())
            epochs = range(1, args["epochs"] + 1)
            pd.DataFrame({"epoch": epochs, "metrics/mAP50-95(B)": 0.1}).to_csv(
                directory / "results.csv", index=False
            )

    monkeypatch.setattr(training, "YOLO", Native)
    for version in ("v0", "v1"):
        path = config_path(project, version)
        config = yaml.safe_load(path.read_text())
        config["checkpoint_sha256"] = file_hash(checkpoint)
        config["training"]["epochs"] = 2
        path.write_text(yaml.safe_dump(config))
        prepare_data(project, version, DATE, workers=1)
    return project, calls


def test_training_requires_opt_in(trained):
    root, calls = trained
    with pytest.raises(FileNotFoundError, match="ALLOW_TRAINING"):
        training.train_experiment(root, "v0")
    assert not calls


def test_full_finetune_uses_version_args_and_is_reused_read_only(trained):
    root, calls = trained
    for version in ("v0", "v1"):
        assert len(training.train_experiment(root, version, allow_training=True)) == 2
    assert len(calls) == 2
    kwargs = calls[1][1]
    run = root / f"runs/v1-{DATE}"
    assert kwargs["data"] == str(run / "data/data.yaml") and kwargs["project"] == str(run)
    assert kwargs["freeze"] == 0 and kwargs["optimizer"] == "SGD" and kwargs["seed"] == 2026
    marker = read_json(run / "train/complete.json")
    assert marker["run"] == f"runs/v1-{DATE}" and marker["epochs_run"] == 2
    assert len(training.train_experiment(root, "v1")) == 2
    assert len(calls) == 2
    with pytest.raises(ValueError, match="already complete"):
        training.train_experiment(root, "v1", resume=True)


def test_changed_yaml_after_training_fails(trained):
    root, _ = trained
    training.train_experiment(root, "v0", allow_training=True)
    path = config_path(root, "v0")
    config = yaml.safe_load(path.read_text())
    config["training"]["lr0"] = 0.02
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match=r"differ from the version YAML \(lr0\)"):
        training.train_experiment(root, "v0")


def test_interrupted_training_requires_explicit_resume(trained):
    root, calls = trained
    training.train_experiment(root, "v1", allow_training=True)
    (root / f"runs/v1-{DATE}/train/complete.json").unlink()
    with pytest.raises(ValueError, match="RESUME=True"):
        training.train_experiment(root, "v1", allow_training=True)
    training.train_experiment(root, "v1", allow_training=True, resume=True)
    assert calls[-1][1] == {"resume": True}
    assert (root / f"runs/v1-{DATE}/train/complete.json").is_file()


def test_resume_rejects_foreign_checkpoint(trained, monkeypatch):
    root, _ = trained
    training.train_experiment(root, "v1", allow_training=True)
    (root / f"runs/v1-{DATE}/train/complete.json").unlink()
    foreign = SimpleNamespace(
        names=dict(enumerate(FIVE_CLASSES)),
        ckpt={"train_args": {"data": str(root / f"runs/v0-{DATE}/data/data.yaml")}},
    )
    monkeypatch.setattr(training, "YOLO", lambda path: foreign)
    with pytest.raises(ValueError, match="another run"):
        training.train_experiment(root, "v1", allow_training=True, resume=True)


def test_unpinned_checkpoint_is_rejected(trained):
    root, _ = trained
    (root / "weights/yolo26n.pt").write_bytes(b"different initial checkpoint")
    with pytest.raises(ValueError, match="pinned checkpoint_sha256"):
        training.train_experiment(root, "v0", allow_training=True)
