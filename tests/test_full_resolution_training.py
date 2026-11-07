"""Issue #51 safeguards; real native CPU tensors on synthetic data, never study evidence."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from vehicle_detection.full_resolution_training import (
    MATRIX,
    MODULE,
    NOTEBOOK,
    PLAN,
    RECEIPT,
    STUDY,
    assess,
    check_recovery,
    evaluate,
    install_selection,
    prepare,
    read_report,
    require_new,
    selection_provenance,
    summarize,
    train,
)


@pytest.mark.parametrize("size", [640, 960])
def test_native_fixed_640_selection_and_terminal_validation_keep_training_size(tmp_path, size):
    torch = pytest.importorskip("torch")
    pytest.importorskip("ultralytics")
    import cv2
    from ultralytics import YOLO

    from vehicle_detection.five_class import FIVE_CLASSES
    from vehicle_detection.training_settings import TRAIN_SETTINGS

    for split in ("train", "val"):
        for kind in ("images", "labels"):
            (tmp_path / kind / split).mkdir(parents=True)
        for i in range(2):
            # Non-square inputs expose a stale loader or accidental square-only validation.
            pixels = np.full((64, 96 + 32 * i, 3), 64 + 64 * i, np.uint8)
            assert cv2.imwrite(str(tmp_path / f"images/{split}/{split}{i}.jpg"), pixels)
            (tmp_path / f"labels/{split}/{split}{i}.txt").write_text(f"{i} 0.5 0.5 0.5 0.5\n")
    data = tmp_path / "data.yaml"
    data.write_text(
        yaml.safe_dump(
            {
                "path": str(tmp_path),
                "train": "images/train",
                "val": "images/val",
                "names": dict(enumerate(FIVE_CLASSES)),
            }
        )
    )
    model = YOLO("yolo26n.yaml")
    observed = install_selection(
        model, ["train0", "train1"], ["val0", "val1"], size, tmp_path / "selection"
    )
    model.train(
        data=str(data),
        project=str(tmp_path),
        name="train_run",
        freeze=11,
        **{
            **TRAIN_SETTINGS,
            "imgsz": size,
            "epochs": 2,
            "device": "cpu",
            "workers": 0,
            "plots": False,
        },
    )
    assert observed["train_shapes"] == {(size, size)}
    assert len(observed["passes"]) == 3
    assert [p["terminal"] for p in observed["passes"]] == [False, False, True]
    assert all(p["imgsz"] == 640 and p["tensor_shapes"] == [(448, 640)] for p in observed["passes"])
    history = pd.read_csv(tmp_path / "train_run/results.csv")
    best = torch.load(
        tmp_path / "train_run/weights/best.pt", weights_only=False, map_location="cpu"
    )
    selected = selection_provenance(history, best, observed)
    peak = history["metrics/mAP50-95(B)"].max()
    assert selected["selected_epoch"] == int(
        history.loc[history["metrics/mAP50-95(B)"] == peak, "epoch"].iloc[-1]
    )
    assert best["epoch"] == -1  # The pre-strip receipt, not this sentinel, identifies selection.
    args = yaml.safe_load((tmp_path / "train_run/args.yaml").read_text())
    assert args["imgsz"] == size and args["batch"] == 16 and args["rect"] is False
    assert not any(n.startswith("model.0.") for n in observed["trainable_parameters"])
    damaged = deepcopy(observed)
    damaged["passes"][-1]["terminal"] = False
    with pytest.raises(ValueError, match="terminal validation"):
        selection_provenance(history, best, damaged)
    damaged = deepcopy(best)
    damaged["train_metrics"]["fitness"] = 123
    with pytest.raises(ValueError, match="Stripped best"):
        selection_provenance(history, damaged, observed)
    with pytest.raises(ValueError, match="selection tensors"):
        model.trainer.validator.preprocess(
            {
                "img": torch.zeros((1, 3, 960, 960)),
                "im_file": ["val0.jpg"],
            }
        )

    # Both matrix cells use precisely the same frozen native best checkpoint.
    from vehicle_detection.common import file_hash
    from vehicle_detection.evaluation import ExperimentValidator
    from vehicle_detection.evaluation_stats import class_ap, load_stats

    path = tmp_path / "train_run/weights/best.pt"
    expected = file_hash(path)
    for inference_size in (640, 960):
        inference = YOLO(str(path))
        metrics = inference.val(
            validator=ExperimentValidator,
            data=str(data),
            imgsz=inference_size,
            batch=32,
            workers=0,
            device="cpu",
            half=False,
            rect=True,
            conf=0.001,
            nms=False,
            plots=False,
            project=str(tmp_path),
            name=f"infer{inference_size}",
        )
        stats = load_stats(tmp_path / f"infer{inference_size}/paired_stats.npz")
        assert set(stats) == {"val0", "val1"}
        ap = class_ap(stats, sorted(stats), detailed=True)
        assert np.isclose(metrics.box.map, np.mean([v["AP50_95"] for v in ap.values()]))
        assert file_hash(path) == expected


@pytest.mark.parametrize(
    "change",
    [
        {"batch_size": 8},
        {"nan_recovery_attempts": 1},
        {"_oom_retries": 1},
        {"args": SimpleNamespace(resume=True)},
    ],
)
def test_recovery_and_batch_changes_are_rejected(change):
    trainer = SimpleNamespace(
        batch_size=16,
        args=SimpleNamespace(resume=False),
        nan_recovery_attempts=0,
        _oom_retries=0,
        device=SimpleNamespace(type="cpu"),
        loss=None,
    )
    trainer.__dict__.update(change)
    with pytest.raises(ValueError, match="forbidden"):
        check_recovery(trainer, 16, 0, float("inf"))


def receipt(root):
    from vehicle_detection.common import file_hash, write_json

    for name in (MODULE, NOTEBOOK, PLAN):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic frozen implementation")
    write_json(
        root / RECEIPT,
        {
            "verified": True,
            "inputs": {name: file_hash(root / name) for name in (MODULE, NOTEBOOK, PLAN)},
        },
    )


def test_read_only_default_is_hash_verified_and_acquisition_flags_do_nothing(tmp_path, monkeypatch):
    import sys

    from vehicle_detection.common import file_hash, write_json

    receipt(tmp_path)
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "ultralytics", None)
    assert read_report(tmp_path)["status"] == "not acquired"
    assert prepare(tmp_path) is None and train(tmp_path) is None and evaluate(tmp_path) is None
    assert not (tmp_path / "runs").exists()
    report = tmp_path / f"docs/{STUDY}"
    write_json(report / "result.json", {"status": "synthetic", "issue47_open": True})
    write_json(report / "evidence.json", {"result.json": file_hash(report / "result.json")})
    assert read_report(tmp_path)["issue47_open"]
    write_json(report / "result.json", {})
    with pytest.raises(ValueError, match="evidence changed"):
        read_report(tmp_path)
    (tmp_path / MODULE).write_text("changed code")
    with pytest.raises(ValueError, match="implementation changed"):
        read_report(tmp_path)


def test_attempts_and_output_symlinks_block_overwrite(tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="never overwrite"):
        require_new(existing)
    link = tmp_path / "link"
    link.symlink_to(existing, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        require_new(link / "new-output")
    (tmp_path / f"runs/{STUDY}").mkdir(parents=True)
    with pytest.raises(FileExistsError, match="never overwrite"):
        prepare(tmp_path, allow_preparation=True)


def test_acquisition_requires_committed_verified_implementation(tmp_path, monkeypatch):
    import subprocess

    from vehicle_detection import full_resolution_training as study
    from vehicle_detection.common import read_json, write_json

    receipt(tmp_path)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    with pytest.raises(ValueError, match="Commit protocol"):
        study.frozen_inputs(tmp_path)
    path = tmp_path / "tests/test_full_resolution_training.py"
    path.parent.mkdir()
    path.write_text("synthetic test")
    config = tmp_path / "configs/data.yaml"
    config.parent.mkdir()
    config.write_text("frozen config")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Synthetic Test",
            "-c",
            "user.email=synthetic@example.invalid",
            "commit",
            "-qm",
            "synthetic frozen fixture",
        ],
        cwd=tmp_path,
        check=True,
    )
    monkeypatch.setattr(study.md, "version", lambda name: "8.4.198")
    assert "configs/data.yaml" in study.frozen_inputs(tmp_path)
    config.write_text("uncommitted change")
    with pytest.raises(ValueError, match="Commit protocol"):
        study.frozen_inputs(tmp_path)
    implementation = read_json(tmp_path / RECEIPT)
    implementation["verified"] = False
    write_json(tmp_path / RECEIPT, implementation)
    with pytest.raises(ValueError, match="verification is required"):
        study.frozen_inputs(tmp_path)


def positive_result():
    from vehicle_detection.five_class import FIVE_CLASSES

    def cohort(ap):
        return {
            "mAP50_95": ap,
            "classes": {
                c: {"AP50_95": ap, "source_groups": 10, "instances": 30} for c in FIVE_CLASSES
            },
            "counts": {c: {"TP50": 20, "FN50": 10, "FP50": 2} for c in FIVE_CLASSES},
        }

    return {
        "metrics": {
            key: {c: cohort(ap) for c in ("all", "dark", "bright")}
            for key, ap in (("640/960", 0.2), ("960/960", 0.21))
        },
        "paired_interval": {
            "estimand": {"class_ids": list(range(5))},
            "recovery_ci_low": 0.001,
            "per_class": [{"class": c, "ci_low": 0.001, "ci_high": 0.02} for c in FIVE_CLASSES],
        },
        "training": {str(s): {"wall_seconds": 10, "peak_allocated_gib": 5} for s in (640, 960)},
        "timings": {
            key: {"native_ms_per_image": {"inference": 2}} for key in ("640/960", "960/960")
        },
    }


def test_sparse_contradictory_or_uncertain_rare_class_evidence_cannot_cleanly_pass():
    result = positive_result()
    assert assess(result)["status"] == "follow-up warranted"
    result["metrics"]["960/960"]["dark"]["classes"]["bus"]["source_groups"] = 4
    result["metrics"]["960/960"]["all"]["counts"]["bicycle"]["FN50"] = 11
    result["paired_interval"]["per_class"][-1]["ci_low"] = -0.01
    decision = assess(result)
    assert decision["status"] == "inconclusive" and len(decision["inconclusive"]) == 3
    result["metrics"]["960/960"]["all"]["classes"]["bus"]["AP50_95"] = 0.17
    decision = assess(result)
    assert decision["status"] == "negative" and len(decision["inconclusive"]) == 3
    assert not decision["policy_selected"] and decision["issue47_open"]


def test_summary_uses_1000_paired_original_source_group_draws_and_fixed_contrasts(
    tmp_path, monkeypatch
):
    pytest.importorskip("ultralytics")
    from vehicle_detection import evaluation_stats
    from vehicle_detection.common import write_json

    result = positive_result()
    for key in ("640/640", "960/640"):
        result["metrics"][key] = deepcopy(result["metrics"]["640/960"])
    rows = [{"id": "one", "group": "leakage", "source_group": "original"}]
    calls = []

    def interval(before, after, population, **kwargs):
        calls.append((population, kwargs))
        return deepcopy(result["paired_interval"])

    monkeypatch.setattr(evaluation_stats, "load_stats", lambda path: {"one": {}})
    monkeypatch.setattr(evaluation_stats, "paired_interval", interval)
    monkeypatch.setattr("vehicle_detection.resolution_training.examples", lambda *args: [])
    for size in (640, 960):
        write_json(
            tmp_path / f"docs/{STUDY}/train{size}_measurement.json", result["training"][str(size)]
        )
    summary = summarize(
        tmp_path,
        result["metrics"],
        result["timings"],
        {k: {"path": "synthetic.npz"} for k in ("640/960", "960/960")},
        rows,
        {c: rows for c in ("all", "dark", "bright")},
    )
    assert len(calls) == 3
    assert all(
        kwargs["resamples"] == 1000 and population["rows"][0]["group"] == "original"
        for population, kwargs in calls
    )
    assert summary["contrasts"]["D-B"] == pytest.approx(0.01)
    assert summary["contrasts"]["C-A"] == 0
    assert MATRIX == ((640, 640), (960, 960), (640, 960), (960, 640))


@pytest.mark.parametrize("completed", [False, True])
def test_new_notebook_default_in_fresh_kernel_never_reads_data_or_models(tmp_path, completed):
    import shutil

    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    from vehicle_detection.common import file_hash

    root = tmp_path / "project"
    for name in (PLAN, MODULE, NOTEBOOK, RECEIPT):
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, destination)
    if completed:
        shutil.copytree(f"docs/{STUDY}", root / f"docs/{STUDY}")
    (root / "pyproject.toml").touch()
    before = {str(p.relative_to(root)): file_hash(p) for p in root.rglob("*") if p.is_file()}
    notebook = nbformat.read(root / NOTEBOOK, as_version=4)
    notebook.cells.insert(
        1,
        nbformat.v4.new_code_cell("""import builtins
import io
import sys
from pathlib import Path
from unittest.mock import patch

for module in ("torch", "ultralytics", "vehicle_detection.experiment_data",
               "vehicle_detection.evaluation_stats", "vehicle_detection.finetuning"):
    sys.modules[module] = None
root = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / "pyproject.toml").is_file())
def guarded(original):
    def wrapped(path, mode="r", *args, **kwargs):
        if isinstance(path, (str, Path)):
            path = Path(path).absolute()
            if path.is_relative_to(root):
                relative = path.relative_to(root)
                if (relative.parts[0] in ("data", "runs", "weights", "configs")
                    or any(c in mode for c in "wax+")):
                    raise AssertionError("Default accessed acquisition state or wrote files")
        return original(path, mode, *args, **kwargs)
    return wrapped
patch("builtins.open", guarded(builtins.open)).start()
patch("io.open", guarded(io.open)).start()
for name in ("prepare", "train", "evaluate"):
    patch("vehicle_detection.full_resolution_training." + name,
          side_effect=AssertionError("Acquisition called by default")).start()
"""),
    )
    nbclient.NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str((root / NOTEBOOK).parent)}},
    ).execute()
    assert all(c.execution_count is not None for c in notebook.cells if c.cell_type == "code")
    assert before == {
        str(p.relative_to(root)): file_hash(p) for p in root.rglob("*") if p.is_file()
    }
    assert all(not (root / name).exists() for name in ("runs", "data", "weights", "configs"))
