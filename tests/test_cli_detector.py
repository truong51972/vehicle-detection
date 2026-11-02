from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from vehicle_detection.cli import main
from vehicle_detection.common import read_json
from vehicle_detection.data import CLASSES, prepare
from vehicle_detection.detector import YoloDetector, train


def test_cli_survey_and_prepare(bdd, tmp_path, capsys):
    main(["survey", "--data", str(bdd), "--out", str(tmp_path / "survey")])
    assert '"counts"' in capsys.readouterr().out
    main(["prepare", "--data", str(bdd), "--out", str(tmp_path / "prepared")])
    assert '"identity"' in capsys.readouterr().out


def test_cli_error_is_actionable(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["prepare", "--data", str(tmp_path / "missing"), "--out", str(tmp_path / "out")])
    assert exc.value.code == 2
    assert "Dataset directory does not exist" in capsys.readouterr().err


def test_training_boundary_uses_only_tune_and_records_hashes(bdd, tmp_path, config, monkeypatch):
    prepared, output = tmp_path / "prepared", tmp_path / "train"
    manifest = prepare(bdd, prepared, config)
    calls = []

    class FakeYolo:
        def __init__(self, model):
            assert model == "yolo11n.pt"

        def train(self, **kwargs):
            calls.append(kwargs)
            dataset = yaml.safe_load(Path(kwargs["data"]).read_text())
            assert dataset["val"] == "images/tune"
            assert list(dataset["names"].values()) == CLASSES
            assert kwargs["seed"] == config["seed"]
            assert kwargs["deterministic"] and kwargs["workers"] == 0
            (output / "weights").mkdir(parents=True)
            (output / "weights/best.pt").write_bytes(b"test only best")
            (output / "weights/last.pt").write_bytes(b"test only last")
            (output / "results.csv").write_text("epoch, metrics/mAP50-95(B)\n1, 0.5\n")
            self.trainer = SimpleNamespace(save_dir=output)
            return SimpleNamespace(results_dict={"metrics/mAP50-95(B)": 0.5})

    monkeypatch.setattr("vehicle_detection.detector.require_yolo", lambda: FakeYolo)
    assert train(prepared, output, config) == output / "weights/best.pt"
    assert len(calls) == 1
    record = read_json(output / "training.json")
    assert record["manifest_identity"] == manifest["identity"]
    assert len(record["best"]["sha256"]) == 64
    assert record["last_epoch"]["epoch"] == "1"


def test_training_rejects_changed_split_config(bdd, tmp_path, config):
    prepared = tmp_path / "prepared"
    prepare(bdd, prepared, config)
    dataset = yaml.safe_load((prepared / "dataset.yaml").read_text())
    dataset["val"] = "images/holdout"
    (prepared / "dataset.yaml").write_text(yaml.safe_dump(dataset))
    with pytest.raises(ValueError, match="possible holdout leakage"):
        train(prepared, tmp_path / "train", config)


def test_one_command_pilot_and_resume(bdd, tmp_path, monkeypatch):
    from test_benchmark import FakeDetector

    from vehicle_detection.common import write_json

    calls = []

    def fake_train(prepared, output, config):
        calls.append(config)
        weights = output / "weights/best.pt"
        weights.parent.mkdir(parents=True)
        weights.write_bytes(b"synthetic test boundary only")
        write_json(output / "training.json", {"config": config, "best": {"path": str(weights)}})
        return weights

    monkeypatch.setattr("vehicle_detection.cli.train", fake_train)
    monkeypatch.setattr("vehicle_detection.benchmark.YoloDetector", lambda *args: FakeDetector())
    output = tmp_path / "pilot"
    args = ["run", "--data", str(bdd), "--out", str(output)]
    main(args)
    assert (output / "survey/survey.json").is_file()
    assert (output / "prepared/manifest.json").is_file()
    assert (output / "benchmark/report.md").is_file()
    main([*args, "--resume"])
    assert len(calls) == 1
    with pytest.raises(SystemExit) as exc:
        main([*args, "--resume", "--seed", "123"])
    assert exc.value.code == 2


def test_frozen_yolo_adapter_mapping_and_prediction_settings(tmp_path, config, monkeypatch):
    import numpy as np

    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"synthetic test; never loaded by torch")
    calls = []

    def tensor(array):
        return SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: np.asarray(array)))

    class FakeYolo:
        names = dict(enumerate(CLASSES))

        def __init__(self, weights):
            assert weights == str(checkpoint)

        def predict(self, **kwargs):
            calls.append(kwargs)
            boxes = SimpleNamespace(
                cls=tensor([0]), xyxy=tensor([[10, 12, 50, 40]]), conf=tensor([0.9])
            )
            return [SimpleNamespace(boxes=boxes)]

    monkeypatch.setattr("vehicle_detection.detector.require_yolo", lambda: FakeYolo)
    detector = YoloDetector(checkpoint, config)
    image = np.zeros((64, 80, 3), np.uint8)
    assert detector.predict(image) == [[0, 10, 12, 50, 40, 0.9]]
    assert calls[0]["source"] is image
    assert calls[0]["batch"] == 1
    assert calls[0]["conf"] == config["benchmark"]["confidence"]
    assert calls[0]["iou"] == config["benchmark"]["nms_iou"]
    assert calls[0]["rect"] is calls[0]["half"] is calls[0]["augment"] is False
    assert len(detector.identity["weights_sha256"]) == 64
    FakeYolo.names = {0: "wrong class"}
    with pytest.raises(ValueError, match="do not match"):
        YoloDetector(checkpoint, config)
