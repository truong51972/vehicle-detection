import json
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml
from test_experiment_data import experiment_project as experiment_project
from test_finetuning import trained_project as trained_project

pytest.importorskip("ultralytics")

from vehicle_detection import evaluation  # noqa: E402
from vehicle_detection.five_class import FIVE_CLASSES  # noqa: E402


@pytest.fixture
def evaluation_model(trained_project, monkeypatch):
    root, _ = trained_project
    calls = []

    class Model:
        def __init__(self, path):
            self.names = dict(enumerate(FIVE_CLASSES))
            self.model = SimpleNamespace(yaml={"yaml_file": "yolo26n.yaml"})
            self.ckpt = {"train_args": {"data": str(root / "runs/v1/train.yaml")}}

        def val(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                mean_results=lambda: [0.4, 0.3, 0.2, 0.1],
                to_json=lambda **kw: json.dumps([]),
                speed={"inference": 1.0},
            )

    monkeypatch.setattr(evaluation, "YOLO", Model)
    return root, calls


def test_evaluation_pairs_policy_and_detector_and_reuses_metrics(evaluation_model):
    root, calls = evaluation_model
    with pytest.raises(FileNotFoundError, match="ALLOW_INFERENCE"):
        evaluation.evaluate_experiment(root, "v1")
    result = evaluation.evaluate_experiment(root, "v1", allow_inference=True)
    assert result["overall"]["mAP50_95"] == 0.1
    assert calls[0]["data"] == str(root / "runs/v1/data/data.yaml")
    assert calls[0]["split"] == "val"
    assert result["final_test_accessed"] is False
    assert evaluation.evaluate_experiment(root, "v1") == result
    assert len(calls) == 1
    with pytest.raises(ValueError, match="settings changed"):
        evaluation.evaluate_experiment(root, "v1", settings={"batch": 1})
    checkpoint = root / "runs/v1/train/weights/best.pt"
    checkpoint.write_bytes(b"a different detector")
    with pytest.raises(ValueError, match="inputs/settings changed"):
        evaluation.evaluate_experiment(root, "v1")


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize(
    "failure", ["interrupted", "missing_epoch", "settings", "freeze", "missing_history"]
)
def test_evaluation_requires_completed_matching_training(evaluation_model, cached, failure):
    root, calls = evaluation_model
    saved = root / "runs/v1/eval/metrics.json"
    if cached:
        evaluation.evaluate_experiment(root, "v1", allow_inference=True)
    previous = saved.read_bytes() if cached else None
    native_calls = len(calls)
    run = root / "runs/v1/train"
    history = run / "results.csv"
    error, message = ValueError, "epoch budget"
    if failure in {"interrupted", "missing_epoch"}:
        pd.read_csv(history).iloc[[0 if failure == "interrupted" else 1]].to_csv(
            history, index=False
        )
    elif failure == "missing_history":
        history.unlink()
        error, message = FileNotFoundError, "results.csv"
    else:
        path = run / "args.yaml"
        args = yaml.safe_load(path.read_text())
        key = "optimizer" if failure == "settings" else "freeze"
        args[key] = "SGD" if failure == "settings" else 0
        path.write_text(yaml.safe_dump(args))
        message = "settings differ"
    with pytest.raises(error, match=message):
        evaluation.evaluate_experiment(root, "v1", allow_inference=True)
    assert len(calls) == native_calls
    assert (saved.read_bytes() if saved.exists() else None) == previous


def test_wrong_experiment_and_taxonomy_fail_before_inference(trained_project, monkeypatch):
    root, _ = trained_project
    path = root / "runs/v1/train/args.yaml"
    args = yaml.safe_load(path.read_text())
    args["data"] = str(root / "runs/v0/train.yaml")
    path.write_text(yaml.safe_dump(args))
    with pytest.raises(ValueError, match="another experiment"):
        evaluation.evaluate_experiment(root, "v1", allow_inference=True)
    args["data"] = str(root / "runs/v1/train.yaml")
    path.write_text(yaml.safe_dump(args))
    monkeypatch.setattr(evaluation, "YOLO", lambda path: SimpleNamespace(names={0: "car"}))
    with pytest.raises(ValueError, match="taxonomy"):
        evaluation.evaluate_experiment(root, "v1", allow_inference=True)
    monkeypatch.setattr(
        evaluation,
        "YOLO",
        lambda path: SimpleNamespace(
            names=dict(enumerate(FIVE_CLASSES)),
            model=SimpleNamespace(yaml={"yaml_file": "yolo26n.yaml"}),
            ckpt={"train_args": {"data": str(root / "runs/v0/train.yaml")}},
        ),
    )
    with pytest.raises(ValueError, match="Checkpoint belongs"):
        evaluation.evaluate_experiment(root, "v1", allow_inference=True)


def test_historical_measurements_keep_original_attribution():
    from pathlib import Path

    raw = evaluation.historical_result(Path.cwd(), "raw")
    adaptive = evaluation.historical_result(Path.cwd(), "adaptive")
    assert raw["evaluation"] == "raw-raw"
    assert adaptive["evaluation"] == "adaptive-adaptive"
    assert raw["result"]["overall"]["mAP50_95"] == pytest.approx(0.1907560653)
    assert adaptive["result"]["overall"]["mAP50_95"] == pytest.approx(0.1936308567)
