"""Notebook contracts plus a fresh-kernel lifecycle run on synthetic CPU fixtures."""

import ast
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

VERSIONS = ("v0", "v1", "v2", "v3")
ACTIVE = {
    "00_data_init.ipynb",
    "experiments/compare.ipynb",
    *(f"experiments/{v}/01_experiments.ipynb" for v in VERSIONS[1:]),
    *(
        f"experiments/{v}/{n}.ipynb"
        for v in VERSIONS
        for n in ("02_train_prepare", "03_train", "04_eval")
    ),
}
DATE = "2026-11-09"


def cells(name):
    return json.loads((Path("notebooks") / name).read_text())["cells"]


def test_active_notebooks_are_clean_and_independent():
    assert {
        p.relative_to("notebooks").as_posix() for p in Path("notebooks").rglob("*.ipynb")
    } == ACTIVE
    for name in ACTIVE:
        for cell in cells(name):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            tree = ast.parse(source)
            assert cell["execution_count"] is None and cell["outputs"] == []
            assert not any(token in source for token in ("%run", "runpy", "papermill", "/home/"))
            if name.endswith("04_eval.ipynb"):
                assert "train_experiment" not in source
                assert not any(
                    isinstance(n, ast.Attribute) and n.attr == "train" for n in ast.walk(tree)
                )


def test_active_notebooks_have_numbered_context_and_explicit_prerequisites():
    for name in ACTIVE:
        nb = cells(name)
        overview = "".join(nb[0]["source"])
        assert all(f"**{label}:**" in overview for label in ("Inputs", "Outputs", "Prerequisites"))
        sections = re.findall(
            r"^## (\d+)\. (.+)$", "\n".join("".join(c["source"]) for c in nb), re.M
        )
        assert [int(n) for n, _ in sections] == list(range(1, len(sections) + 1))
        assert sections[0][1] == "Configuration" and "Summary" in sections[-1][1]
        code = ["".join(c["source"]) for c in nb if c["cell_type"] == "code"]
        assert "require_paths(" in code[0]


def test_documentation_links_resolve_and_tracked_text_has_no_absolute_paths():
    for path in [Path("README.md"), Path("data/README.md"), *Path("notebooks").rglob("README.md")]:
        for link in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if "://" not in link and not link.startswith("#"):
                assert (path.parent / link.split("#")[0]).exists(), (path, link)
    tracked = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True)
    pattern = re.compile(r"(?<![\w.~])/(?:home|Users|mnt|root)/\w")
    for name in tracked.stdout.splitlines():
        path = Path(name)
        if path.suffix in {".py", ".ipynb", ".md", ".yaml", ".yml", ".toml", ".qmd", ".json"}:
            if path.exists():
                assert not pattern.search(path.read_text()), f"Absolute path in {name}"


def test_lifecycle_notebooks_run_in_fresh_kernels(project):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    torch = pytest.importorskip("torch")
    pytest.importorskip("ultralytics")
    from ultralytics.nn.tasks import DetectionModel, yaml_model_load

    from vehicle_detection.common import file_hash
    from vehicle_detection.experiment_data import config_path
    from vehicle_detection.training import COCO_VEHICLES

    root = project
    # Keep the fixture's CPU version YAMLs; copy only notebooks/READMEs.
    shutil.copytree(
        "notebooks",
        root / "notebooks",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("v*.yaml"),
    )

    def run(path, overrides="", setup=""):
        notebook = nbformat.read(root / "notebooks" / path, as_version=4)
        cell = next(c for c in notebook.cells if c.cell_type == "code")
        cell.source += "\n" + overrides
        notebook.cells.insert(1, nbformat.v4.new_code_cell(setup or "pass"))
        directory = (root / "notebooks" / path).parent
        nbclient.NotebookClient(
            notebook,
            timeout=600,
            kernel_name="python3",
            resources={"metadata": {"path": str(directory)}},
        ).execute()
        return notebook

    guard = """from unittest.mock import patch
from ultralytics import YOLO
patch.object(YOLO, "train", side_effect=AssertionError("Training started")).start()
patch.object(YOLO, "val", side_effect=AssertionError("Inference started")).start()
"""
    run("00_data_init.ipynb", "AUDIT_SOURCES = True", guard)
    for version in VERSIONS[1:]:
        run(f"experiments/{version}/01_experiments.ipynb", "SAMPLES = 1\nSHOW = 1", guard)
    assert not (root / "runs").exists()
    for version in VERSIONS:
        run(f"experiments/{version}/02_train_prepare.ipynb", f"RUN_DATE = '{DATE}'", guard)
        assert not (root / f"runs/{version}-{DATE}").exists()
        prepare = f"RUN_DATE = '{DATE}'\nALLOW_PREPARATION = True"
        run(f"experiments/{version}/02_train_prepare.ipynb", prepare, guard)
        run(f"experiments/{version}/03_train.ipynb", "", guard)
        run(f"experiments/{version}/04_eval.ipynb", "", guard)
        assert (root / f"runs/{version}-{DATE}/data/data.yaml").is_file()
        assert not (root / f"runs/{version}-{DATE}/train").exists()

    # Exercise the actual native CPU API with a tiny synthetic COCO-80 checkpoint.
    model = DetectionModel(yaml_model_load("yolo26n.yaml"), nc=80, verbose=False)
    model.names = dict.fromkeys(range(80), "other") | COCO_VEHICLES
    checkpoint = root / "weights/yolo26n.pt"
    checkpoint.parent.mkdir()
    torch.save({"model": model, "train_args": {"task": "detect"}}, checkpoint)
    for version in ("v0", "v3"):
        path = config_path(root, version)
        config = yaml.safe_load(path.read_text())
        config["checkpoint_sha256"] = file_hash(checkpoint)
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        run(f"experiments/{version}/03_train.ipynb", "ALLOW_TRAINING = True")
        run(f"experiments/{version}/03_train.ipynb", "", guard)
        run(f"experiments/{version}/04_eval.ipynb", "ALLOW_INFERENCE = True")
        run(f"experiments/{version}/04_eval.ipynb", "", guard)
        metrics = json.loads((root / f"runs/{version}-{DATE}/eval/metrics.json").read_text())
        assert metrics["inputs"]["run"] == f"runs/{version}-{DATE}"
        assert str(root) not in json.dumps(metrics)
    run("experiments/compare.ipynb", "BOOTSTRAP_RESAMPLES = 2", guard)
