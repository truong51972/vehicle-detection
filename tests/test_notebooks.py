"""Notebook contracts plus a fresh-kernel lifecycle run on synthetic CPU fixtures."""

import ast
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

VERSIONS = ("v0", "v1", "v2", "v3", "v4")
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


@pytest.mark.parametrize("name", sorted(ACTIVE))
def test_committed_execution_gates_default_to_false(name):
    gates = set()
    code = [c for c in cells(name) if c["cell_type"] == "code"]
    for index, cell in enumerate(code):
        tree = ast.parse("".join(cell["source"]))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if not isinstance(target, ast.Name):
                    continue
                gate = target.id
                if not (
                    gate.startswith("ALLOW_")
                    or gate in {"AUDIT_SOURCES", "MEASURE_PREPROCESS_COST", "RESUME"}
                ):
                    continue
                assert index == 0 and node in tree.body, (name, gate)
                assert gate not in gates, (name, gate, "gate reassigned")
                assert isinstance(node.value, ast.Constant) and node.value.value is False, (
                    name,
                    gate,
                    "committed default must be literal False",
                )
                gates.add(gate)
    required = {
        "02_train_prepare.ipynb": "ALLOW_PREPARATION",
        "03_train.ipynb": "ALLOW_TRAINING",
        "04_eval.ipynb": "ALLOW_INFERENCE",
        "compare.ipynb": "ALLOW_INFERENCE",
        "00_data_init.ipynb": "AUDIT_SOURCES",
    }
    if Path(name).name in required:
        assert required[Path(name).name] in gates, name


def test_active_notebooks_are_independent():
    found = {
        p.relative_to("notebooks").as_posix()
        for p in Path("notebooks").rglob("*.ipynb")
        if ".ipynb_checkpoints" not in p.parts
    }
    assert found == ACTIVE
    # Saved outputs are historical evidence; Run All safety depends on committed source.
    for name in ACTIVE:
        for cell in cells(name):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            tree = ast.parse(source)
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
        # Ultralytics archives its absolute runtime paths in native args.yaml.
        # Keep those measurements intact while checking portable source/config files.
        if path.match("runs/*/train/args.yaml") or name == (
            "docs/v4/evidence/matched_raw/train/args.yaml"
        ):
            continue
        if path.suffix in {".py", ".ipynb", ".md", ".yaml", ".yml", ".toml", ".qmd", ".json"}:
            if path.exists():
                content = path.read_text()
                if path.suffix == ".ipynb":
                    # Executed training notebooks retain local log paths as evidence;
                    # source cells, not transient outputs, must stay portable.
                    notebook = json.loads(content)
                    content = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
                assert not pattern.search(content), f"Absolute path in {name}"


def test_project_root_ignores_kernel_working_directory(tmp_path, monkeypatch):
    from vehicle_detection.common import project_root

    repo = Path.cwd().resolve()
    assert project_root(repo / "notebooks/experiments/v0") == repo
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "other"\n')
    assert project_root(tmp_path / "x") == repo
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    notebook = nbformat.read("notebooks/00_data_init.ipynb", as_version=4)
    notebook.cells.append(nbformat.v4.new_code_cell("import os\nprint(os.getcwd())"))
    nbclient.NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(tmp_path)}},
    ).execute()
    assert notebook.cells[-1].outputs[0]["text"].strip() == str(repo)


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
        ignore=shutil.ignore_patterns("v*.yaml", ".ipynb_checkpoints"),
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
    read_only_guard = (
        guard
        + """import vehicle_detection.experiment_data as experiment_data
patch.object(experiment_data, "prepare_data", side_effect=AssertionError("Export started")).start()
"""
    )
    run("00_data_init.ipynb", "AUDIT_SOURCES = True", guard)
    for version in VERSIONS[1:]:
        run(f"experiments/{version}/01_experiments.ipynb", "SAMPLES = 1\nSHOW = 1", guard)
    assert not (root / "runs").exists()
    for version in VERSIONS:
        # Use committed defaults: overrides previously hid unsafe True flags.
        for phase in ("02_train_prepare", "03_train", "04_eval"):
            notebook = run(f"experiments/{version}/{phase}.ipynb", setup=read_only_guard)
            output = "".join(
                o.get("text", "")
                for cell in notebook.cells
                if cell.cell_type == "code"
                for o in cell.outputs
            )
            assert "disabled; inspecting saved artifacts only" in output
        assert not (root / f"runs/{version}-{DATE}").exists()
    assert not (root / "runs").exists()
    for version in VERSIONS:
        prepare = f"RUN_DATE = '{DATE}'\nALLOW_PREPARATION = True"
        run(f"experiments/{version}/02_train_prepare.ipynb", prepare, guard)
        before = {p: file_hash(p) for p in (root / "runs").rglob("*") if p.is_file()}
        for phase in ("02_train_prepare", "03_train", "04_eval"):
            run(f"experiments/{version}/{phase}.ipynb", setup=read_only_guard)
        after = {p: file_hash(p) for p in (root / "runs").rglob("*") if p.is_file()}
        assert after == before
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

    # Exercise both off-diagonal YOLO.val paths on the same tiny synthetic
    # development images. This is a CPU contract test, not scientific evidence.
    from vehicle_detection.comparison import (
        compare_policies,
        evaluate_matrix,
        inspect_report,
        result_table,
    )

    matrix = evaluate_matrix(
        root,
        policies=("v3",),
        run_dates={"v0": DATE, "v3": DATE},
        allow_inference=True,
    )
    assert {f"{row['model']}/{row['input']}" for row in matrix["rows"]} == {
        "v0/v0",
        "v0/v3",
        "v3/v0",
        "v3/v3",
    }
    assert set(matrix["results"]) == {"v0/v0", "v0/v3", "v3/v0", "v3/v3"}
    assert {row["status"] for row in matrix["rows"]} <= {"cached", "evaluated"}
    contrasts, ci = compare_policies(matrix, bootstrap_resamples=8)
    assert len(contrasts) == 6 and "v3" in ci
    assert ci["v3"]["contrasts"]["joint_pipeline"]["valid_draws"] > 1
    cached = evaluate_matrix(root, policies=("v3",), allow_inference=False)
    assert {row["status"] for row in cached["rows"]} == {"cached"}

    # Newly evaluated native cells remain visible but cannot enter report effects
    # until an explicit review attests their identities, split and runtime.
    from vehicle_detection.common import write_json

    pending = inspect_report(root)
    overview = result_table(pending).set_index(["model", "input"])
    assert overview.loc[("v0", "v3"), "status"] == "unindexed"
    assert overview.loc[("v0", "v3"), "mAP50_95_pct"] == pytest.approx(
        matrix["results"]["v0/v3"]["overall"]["mAP50_95"] * 100
    )
    assert set(pending["unindexed"]) == {"v0/v3", "v3/v0"}
    assert compare_policies(pending, bootstrap_resamples=2)[0].empty
    index_path = root / "configs/comparison_evidence.json"
    assert not index_path.exists()
    shared = yaml.safe_load((root / "configs/data.yaml").read_text())
    index = {"runs": {}, "artifacts": {}}
    for version in ("v0", "v3"):
        ctx = pending["contexts"][version]
        index["runs"][ctx["run_id"]] = {
            "profile": "synthetic-cpu",
            "runtime": "synthetic native CPU contract test",
            "source": "test fixture; never scientific evidence",
            "splits_sha256": shared["splits_sha256"],
        }
        for name in (
            "config.yaml",
            "train/complete.json",
            "eval/metrics.json",
            "data/metadata.csv",
        ):
            path = ctx["run"] / name
            index["artifacts"][path.relative_to(root).as_posix()] = file_hash(path)
    for key in pending["unindexed"]:
        path = pending["paths"][key] / "metrics.json"
        index["artifacts"][path.relative_to(root).as_posix()] = file_hash(path)
    write_json(index_path, index)
    reviewed = inspect_report(root)
    assert not reviewed["unindexed"]
    assert len(compare_policies(reviewed)[0]) == 6
