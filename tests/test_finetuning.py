from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml
from test_experiment_data import experiment_project as experiment_project

pytest.importorskip("ultralytics")

from vehicle_detection import finetuning  # noqa: E402
from vehicle_detection.common import file_hash  # noqa: E402
from vehicle_detection.experiment_data import prepare_data  # noqa: E402
from vehicle_detection.five_class import FIVE_CLASSES  # noqa: E402


@pytest.fixture
def trained_project(experiment_project, monkeypatch):
    root = experiment_project
    checkpoint = root / "weights/yolo26n.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"synthetic initial checkpoint")
    monkeypatch.setattr(finetuning, "PRETRAINED_SHA256", file_hash(checkpoint))
    calls = []

    class Native:
        def __init__(self, path):
            self.path = path
            self.ckpt = {"train_args": {"data": str(Path(path).parents[2] / "train.yaml")}}
            self.names = {i: "other" for i in range(80)}
            self.names.update(
                {2: "car", 7: "truck", 5: "bus", 6: "train", 3: "motorcycle", 1: "bicycle"}
            )
            if str(path).endswith("last.pt"):
                self.names = dict(enumerate(FIVE_CLASSES))
            self.model = SimpleNamespace(
                yaml={"yaml_file": "yolo26n.yaml", "end2end": True, "backbone": [None] * 11},
                model=[None] * 24,
            )

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
            pd.DataFrame(
                {
                    "epoch": list(range(1, args["epochs"] + 1)),
                    "metrics/mAP50-95(B)": [0.1] * args["epochs"],
                }
            ).to_csv(directory / "results.csv", index=False)

    monkeypatch.setattr(finetuning, "YOLO", Native)
    for exp in ("v0", "v1"):
        path = root / f"configs/{exp}.yaml"
        config = yaml.safe_load(path.read_text())
        config["checkpoint_sha256"] = file_hash(checkpoint)
        config["training"].update(epochs=2, device="cpu", workers=0)
        path.write_text(yaml.safe_dump(config))
        prepare_data(root, exp)
        finetuning.train_experiment(root, exp, allow_training=True)
    return root, calls


def test_native_training_is_isolated_raw_selection_and_reuse_is_read_only(trained_project):
    root, calls = trained_project
    assert len(calls) == 2 and calls[0][0] == calls[1][0]
    for exp in ("v0", "v1"):
        config = yaml.safe_load((root / f"runs/{exp}/train.yaml").read_text())
        assert config["val"] == str(root / "runs/v0/data/images/val")
        raw = yaml.safe_load((root / "runs/v0/data/data.yaml").read_text())
        assert config["selection_files_sha256"] == raw["files_sha256"]
        assert config["path"] == str(root / f"runs/{exp}/data")
        before = (root / f"runs/{exp}/train/results.csv").stat().st_mtime_ns
        assert len(finetuning.train_experiment(root, exp)) == 2
        assert (root / f"runs/{exp}/train/results.csv").stat().st_mtime_ns == before
    assert len(calls) == 2
    assert calls[0][1]["freeze"] == calls[1][1]["freeze"] == 11


def test_training_requires_opt_in(experiment_project):
    root = experiment_project
    prepare_data(root, "v0")
    with pytest.raises(FileNotFoundError, match="ALLOW_TRAINING"):
        finetuning.train_experiment(root, "v0")


def test_native_resume_is_explicit(trained_project):
    root, calls = trained_project
    (root / "runs/v1/train/results.csv").unlink()
    with pytest.raises(ValueError, match="Interrupted"):
        finetuning.train_experiment(root, "v1", allow_training=True)
    finetuning.train_experiment(root, "v1", allow_training=True, resume=True)
    assert calls[-1][1] == {"resume": True}
    path = root / "runs/v1/train/args.yaml"
    args = yaml.safe_load(path.read_text())
    args["data"] = str(root / "runs/v0/train.yaml")
    path.write_text(yaml.safe_dump(args))
    with pytest.raises(ValueError, match="settings differ"):
        finetuning.train_experiment(root, "v1")


def test_backbone_boundary_checks_the_actual_native_layout():
    from ultralytics.nn.tasks import DetectionModel, yaml_model_load

    model = DetectionModel(yaml_model_load("yolo26n.yaml"), nc=5, verbose=False)
    assert finetuning.backbone_boundary(model) == 11
    model.yaml["yaml_file"] = "other.yaml"
    with pytest.raises(ValueError, match="architecture"):
        finetuning.backbone_boundary(model)


def test_native_resume_rejects_foreign_checkpoint(trained_project, monkeypatch):
    root, _ = trained_project
    (root / "runs/v1/train/results.csv").unlink()
    monkeypatch.setattr(
        finetuning,
        "YOLO",
        lambda path: SimpleNamespace(
            names=dict(enumerate(FIVE_CLASSES)),
            ckpt={"train_args": {"data": str(root / "runs/v0/train.yaml")}},
        ),
    )
    with pytest.raises(ValueError, match="Resume checkpoint belongs"):
        finetuning.train_experiment(root, "v1", allow_training=True, resume=True)
