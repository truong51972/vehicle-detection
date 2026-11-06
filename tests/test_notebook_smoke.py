"""Fresh-kernel lifecycle execution on synthetic CPU fixtures; never research evidence."""

import shutil
from pathlib import Path

import pytest
import yaml
from test_experiment_data import experiment_project as experiment_project


def test_versioned_notebooks_run_independently_in_fresh_kernels(experiment_project):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("ultralytics")
    root = experiment_project
    shutil.copytree("notebooks", root / "notebooks")
    shutil.copytree("docs", root / "docs")
    guard = """from unittest.mock import patch
import sys
from ultralytics import YOLO
sys.modules["vehicle_detection.s20_report"] = None
for target in ["vehicle_detection.protocol.build_protocol", "vehicle_detection.data.convert_labels",
               "vehicle_detection.adaptive.calibrate_policy"]:
    patch(target, side_effect=AssertionError("Upstream work started")).start()
patch.object(YOLO, "train", side_effect=AssertionError("Training started")).start()
patch.object(YOLO, "val", side_effect=AssertionError("Inference started")).start()
"""

    def run(path, overrides="", setup=guard, cwd=None):
        notebook = nbformat.read(root / "notebooks" / path, as_version=4)
        cell = next(c for c in notebook.cells if c.cell_type == "code")
        cell.source += "\n" + overrides
        notebook.cells.insert(1, nbformat.v4.new_code_cell(setup))
        nbclient.NotebookClient(
            notebook,
            timeout=180,
            kernel_name="python3",
            resources={"metadata": {"path": str(cwd or root)}},
        ).execute()
        assert all(c.execution_count is not None for c in notebook.cells if c.cell_type == "code")

    run("00_data_init.ipynb", "AUDIT_SOURCES = True")
    run("experiments/v1/01_experiments.ipynb")
    run("experiments/v2/01_experiments.ipynb")
    for version in ("v0", "v1", "v2"):
        directory = root / f"notebooks/experiments/{version}"
        run(
            f"experiments/{version}/02_train_prepare.ipynb",
            "ALLOW_PREPARATION = True",
            cwd=directory,
        )
        run(f"experiments/{version}/03_train.ipynb", cwd=directory)
        run(f"experiments/{version}/04_eval.ipynb", cwd=directory)
        assert not (root / f"runs/{version}/train").exists()
        assert not (root / f"runs/{version}/eval").exists()

    # Exercise the actual native CPU API with a tiny synthetic COCO head/checkpoint.
    torch = pytest.importorskip("torch")
    from ultralytics.nn.tasks import DetectionModel, yaml_model_load

    from vehicle_detection.baseline import COCO_TO_VEHICLE
    from vehicle_detection.common import file_hash
    from vehicle_detection.data import CLASSES

    model = DetectionModel(yaml_model_load("yolo26n.yaml"), nc=80, verbose=False)
    model.names = dict.fromkeys(range(80), "other")
    model.names.update({c: CLASSES[b] for c, b in COCO_TO_VEHICLE.items()})
    checkpoint = root / "weights/yolo26n.pt"
    checkpoint.parent.mkdir()
    torch.save({"model": model, "train_args": {"task": "detect"}}, checkpoint)
    for version in ("v0", "v1", "v2"):
        config_path = root / f"configs/{version}.yaml"
        config = yaml.safe_load(config_path.read_text())
        config["checkpoint_sha256"] = file_hash(checkpoint)
        config["training"].update(epochs=1, batch=2, nbs=2, imgsz=64, workers=0, device="cpu")
        config_path.write_text(yaml.safe_dump(config))
        # Config changed only in this isolated fixture; reprepare before training.
        shutil.rmtree(root / f"runs/{version}/data")
        run(f"experiments/{version}/02_train_prepare.ipynb", "ALLOW_PREPARATION = True")
    for version in ("v0", "v1", "v2"):
        run(
            f"experiments/{version}/03_train.ipynb",
            "READ_HISTORICAL = False\nALLOW_TRAINING = True",
            setup=f"""import vehicle_detection.finetuning as training
training.PRETRAINED_SHA256 = {file_hash(checkpoint)!r}
""",
        )
        eval_override = """READ_HISTORICAL = False
ALLOW_INFERENCE = True
EVAL_SETTINGS = {"device": "cpu", "imgsz": 64, "batch": 2, "workers": 0}
"""
        run(
            f"experiments/{version}/04_eval.ipynb",
            eval_override,
            setup="""from unittest.mock import patch
from ultralytics import YOLO
patch.object(YOLO, "train", side_effect=AssertionError("Training started in evaluation")).start()
""",
        )
        run(f"experiments/{version}/04_eval.ipynb", eval_override, setup=guard)
        assert (root / f"runs/{version}/eval/metrics.json").is_file()


def test_comparison_runs_read_only_without_local_runs_or_study_dependencies(tmp_path):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    from vehicle_detection.common import file_hash

    root = tmp_path / "project"
    directory = root / "notebooks/experiments"
    directory.mkdir(parents=True)
    (root / "pyproject.toml").touch()
    shutil.copyfile("notebooks/experiments/compare.ipynb", directory / "compare.ipynb")
    shutil.copytree("docs/issue45-s20", root / "docs/issue45-s20")
    shutil.copytree("docs/issue47", root / "docs/issue47")
    before = {str(p.relative_to(root)): file_hash(p) for p in root.rglob("*") if p.is_file()}
    notebook = nbformat.read(directory / "compare.ipynb", as_version=4)
    notebook.cells.insert(
        1,
        nbformat.v4.new_code_cell("""import builtins
import io
import sys
from pathlib import Path
from unittest.mock import patch

for module in ("torch", "ultralytics", "vehicle_detection.s20_report",
               "vehicle_detection.evaluation", "vehicle_detection.finetuning"):
    sys.modules[module] = None
_root = next(p for p in [Path.cwd(), *Path.cwd().parents]
             if (p / "pyproject.toml").is_file())
def guarded(original):
    def wrapped(path, mode="r", *args, **kwargs):
        if isinstance(path, (str, Path)):
            path = Path(path).absolute()
            if path.is_relative_to(_root):
                if any(c in mode for c in "wax+") or not path.is_relative_to(_root / "docs"):
                    raise AssertionError("Comparison accessed local execution state or wrote files")
        return original(path, mode, *args, **kwargs)
    return wrapped
patch("builtins.open", guarded(builtins.open)).start()
patch("io.open", guarded(io.open)).start()
"""),
    )
    nbclient.NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(directory)}},
    ).execute()
    assert all(c.execution_count is not None for c in notebook.cells if c.cell_type == "code")
    assert before == {
        str(p.relative_to(root)): file_hash(p) for p in root.rglob("*") if p.is_file()
    }
    assert all(not (root / name).exists() for name in ("runs", "weights", "data"))


@pytest.mark.parametrize(
    "notebook_path, report_name, plan_name",
    [
        ("experiments/v3/01_experiments.ipynb", "issue47", "ISSUE47_PLAN.md"),
        (
            "research/issue47/01_method_discovery.ipynb",
            "issue47-discovery",
            "ISSUE47_DISCOVERY_PLAN.md",
        ),
        (
            "research/issue47/02_resolution_training.ipynb",
            "issue47-resolution-training",
            "ISSUE47_TRAINING_PLAN.md",
        ),
    ],
)
def test_research_defaults_run_with_only_committed_evidence(
    tmp_path, notebook_path, report_name, plan_name
):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    from vehicle_detection.common import file_hash

    root = tmp_path / "project"
    directory = root / "notebooks" / Path(notebook_path).parent
    directory.mkdir(parents=True)
    (root / "pyproject.toml").touch()
    shutil.copytree(f"docs/{report_name}", root / f"docs/{report_name}")
    shutil.copyfile(f"docs/{plan_name}", root / f"docs/{plan_name}")
    path = directory / Path(notebook_path).name
    shutil.copyfile(f"notebooks/{notebook_path}", path)
    before = {str(p.relative_to(root)): file_hash(p) for p in root.rglob("*") if p.is_file()}
    notebook = nbformat.read(path, as_version=4)
    notebook.cells.insert(
        1,
        nbformat.v4.new_code_cell("""import sys
from unittest.mock import patch

for module in ("torch", "ultralytics", "vehicle_detection.evaluation_stats",
               "vehicle_detection.finetuning", "vehicle_detection.stress_eval"):
    sys.modules[module] = None
patch("vehicle_detection.v3_pilot.shared_splits", side_effect=AssertionError("Read splits")).start()
patch("vehicle_detection.v3_pilot.source_rows", side_effect=AssertionError("Read sources")).start()
patch("vehicle_detection.v3_pilot.apply", side_effect=AssertionError("Replayed pixels")).start()
patch("vehicle_detection.method_discovery.source_rows",
      side_effect=AssertionError("Read original metadata")).start()
patch("vehicle_detection.method_discovery.build_diagnosis",
      side_effect=AssertionError("Replayed diagnosis")).start()
"""),
    )
    nbclient.NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(directory)}},
    ).execute()
    assert all(c.execution_count is not None for c in notebook.cells if c.cell_type == "code")
    assert before == {
        str(p.relative_to(root)): file_hash(p) for p in root.rglob("*") if p.is_file()
    }
    assert all(not (root / name).exists() for name in ("runs", "weights", "data", "configs"))
