"""Fresh kernels run all four real notebooks on tiny fixtures, never research evidence."""

import shutil
from pathlib import Path

import pytest
from test_study import bdd as bdd

from vehicle_detection.common import read_json, write_json


def test_four_fresh_kernels_and_cached_evaluation_without_upstream_work(bdd, tmp_path):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    torch = pytest.importorskip("torch")
    pytest.importorskip("ultralytics")
    pytest.importorskip("bdd100k")
    from ultralytics.nn.tasks import DetectionModel, yaml_model_load

    from vehicle_detection.adaptive import calibrate_policy
    from vehicle_detection.baseline import COCO_TO_VEHICLE
    from vehicle_detection.common import digest, file_hash
    from vehicle_detection.data import CLASSES

    repository = Path.cwd()
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").touch()
    shutil.copytree(repository / "notebooks", project / "notebooks")
    shutil.copytree(bdd, project / "data/bdd100k")
    write_json(project / "notebooks/test_exposure.json", {})

    # Every execution starts a new kernel. Paths are the notebook's own declared defaults.
    def run(name, setup="", overrides="", cwd=None):
        notebook = nbformat.read(project / "notebooks" / name, as_version=4)
        if setup:
            notebook.cells.insert(1, nbformat.v4.new_code_cell(setup))
        if overrides:
            index = next(i for i, c in enumerate(notebook.cells) if "require_paths(" in c.source)
            cell = notebook.cells[index]
            position = cell.source.index("require_paths(")
            cell.source = cell.source[:position] + overrides + "\n" + cell.source[position:]
        nbclient.NotebookClient(
            notebook,
            timeout=180,
            kernel_name="python3",
            resources={"metadata": {"path": str(cwd or project)}},
        ).execute()
        assert all(c.execution_count is not None for c in notebook.cells if c.cell_type == "code")

    run("01_data_init.ipynb")
    protocol = read_json(project / "runs/data-v1/protocol.json")
    policy = calibrate_policy(protocol, project / "data/bdd100k", sample_size=2, per_operator=1)
    write_json(project / "docs/issue6-v2/policy.json", policy)
    decision = {"status": "v3_not_deployed", "retained_policy_id": policy["policy_id"]}
    decision["decision_id"] = digest(decision)
    write_json(project / "docs/issue11-v2/v3_decision.json", decision)
    shutil.copy(repository / "docs/issue11-v2/experimental_matrix.csv", project / "docs/issue11-v2")
    shutil.copy(repository / "docs/issue11-v2/supported_recovery.png", project / "docs/issue11-v2")
    decision["reason"] = "Fixture policy, never research evidence"
    decision["decision_id"] = digest({k: v for k, v in decision.items() if k != "decision_id"})
    write_json(project / "docs/issue11-v2/v3_decision.json", decision)
    run(
        "02_data_preprocessing.ipynb",
        overrides="FIVE_CLASS_ONLY = False",
        cwd=project / "notebooks",
    )
    historical_roots = [
        project / "runs/data-v1",
        project / "runs/preprocessing-v1",
        project / "docs/preprocessing-v1",
    ]

    def historical_snapshot():
        return {
            str(path): (file_hash(path), path.stat().st_mtime_ns)
            for root in historical_roots
            for path in root.rglob("*")
            if path.is_file()
        }

    before = historical_snapshot()
    analysis = project / "docs/issue11-v2"
    saved_analysis = project / "docs/issue11-v2-saved"
    analysis.rename(saved_analysis)
    run(
        "02_data_preprocessing.ipynb",
        setup="""from unittest.mock import patch
import cv2
for target in ["vehicle_detection.data.convert_labels",
               "vehicle_detection.subsets.build_subsets",
               "vehicle_detection.subsets.freeze_subsets",
               "vehicle_detection.adaptive.calibrate_policy"]:
    patch(target, side_effect=AssertionError("Historical phase triggered")).start()
original_imread = cv2.imread
def guarded_imread(path, *args, **kwargs):
    assert "/val/" not in str(path), "Reserved pixels accessed"
    return original_imread(path, *args, **kwargs)
patch.object(cv2, "imread", side_effect=guarded_imread).start()
""",
    )
    saved_analysis.rename(analysis)
    assert historical_snapshot() == before
    five = read_json(project / "runs/preprocessing-five-v1/summary.json")
    assert five["role_support_annotation_only"]["final_test"]["images"] == len(
        protocol["splits"]["final_test"]
    )

    config = yaml_model_load("yolo26n.yaml")
    config["end2end"] = True
    model = DetectionModel(config, nc=80, verbose=False)
    model.names = dict.fromkeys(range(80), "other")
    model.names.update({c: CLASSES[b] for c, b in COCO_TO_VEHICLE.items()})
    checkpoint = project / "weights/yolo26n.pt"
    checkpoint.parent.mkdir()
    torch.save({"model": model, "train_args": {"task": "detect"}}, checkpoint)
    setup = f"""import vehicle_detection.baseline as baseline
baseline.PRETRAINED_SHA256 = {file_hash(checkpoint)!r}
"""
    train_setup = (
        setup
        + """import vehicle_detection.finetuning as training
training.TRAIN_SETTINGS.update(epochs=1, batch=2, nbs=2, imgsz=64, workers=0, device="cpu")
"""
    )
    run("03_training.ipynb", setup=train_setup)
    (project / "docs/development-v1").mkdir()
    shutil.copy(repository / "docs/development-v1/PLAN.md", project / "docs/development-v1")
    run(
        "04_eval_and_report.ipynb",
        setup=setup,
        overrides="ALLOW_INFERENCE = True\nBOOTSTRAP_RESAMPLES = 2\n"
        "import vehicle_detection.evaluation as evaluation\n"
        "original = evaluation.evaluate_development\n"
        "def evaluate_development(*args, **kwargs):\n"
        '    return original(*args, settings={"device":"cpu", "batch":2, '
        '"imgsz":64, "workers":0}, **kwargs)',
    )
    # Reuse the exact same saved settings in a fresh kernel. Any upstream work is fatal.
    guard = (
        setup
        + """from unittest.mock import patch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
for target in ["vehicle_detection.data.scan_bdd", "vehicle_detection.protocol.build_protocol",
               "vehicle_detection.adaptive.calibrate_policy",
               "vehicle_detection.offline.export_variant",
               "vehicle_detection.finetuning.prepare_pair",
               "vehicle_detection.finetuning.train_branch"]:
    patch(target, side_effect=AssertionError("Upstream phase triggered")).start()
patch.object(YOLO, "train", side_effect=AssertionError("Training triggered")).start()
patch.object(DetectionTrainer, "__init__", side_effect=AssertionError("Trainer created")).start()
patch("vehicle_detection.evaluation.YOLO",
      side_effect=AssertionError("Inference triggered")).start()
"""
    )
    run(
        "04_eval_and_report.ipynb",
        setup=guard,
        overrides="BOOTSTRAP_RESAMPLES = 2\nimport vehicle_detection.evaluation as evaluation\n"
        "original = evaluation.evaluate_development\n"
        "def evaluate_development(*args, **kwargs):\n"
        '    return original(*args, settings={"device":"cpu", "batch":2, '
        '"imgsz":64, "workers":0}, **kwargs)',
        cwd=project / "notebooks",
    )
