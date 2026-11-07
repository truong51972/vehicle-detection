"""Practical active notebook contracts, with history as the historical replay path."""

import ast
import json
import re
from pathlib import Path

ACTIVE = {
    "00_data_init.ipynb",
    "experiments/v0/02_train_prepare.ipynb",
    "experiments/v0/03_train.ipynb",
    "experiments/v0/04_eval.ipynb",
    "experiments/v1/01_experiments.ipynb",
    "experiments/v1/02_train_prepare.ipynb",
    "experiments/v1/03_train.ipynb",
    "experiments/v1/04_eval.ipynb",
    "experiments/v2/01_experiments.ipynb",
    "experiments/v2/02_train_prepare.ipynb",
    "experiments/v2/03_train.ipynb",
    "experiments/v2/04_eval.ipynb",
    "experiments/v3/01_experiments.ipynb",
    "experiments/v3/02_train_prepare.ipynb",
    "experiments/v3/03_train.ipynb",
    "experiments/v3/04_eval.ipynb",
    "experiments/compare.ipynb",
    "research/issue47/01_method_discovery.ipynb",
    "research/issue47/02_resolution_training.ipynb",
    "research/issue47/03_full_resolution_training.ipynb",
    "research/issue47/04_gamma_sharpen_pilot.ipynb",
}


def test_active_entrypoints_are_independent_and_have_no_fabricated_execution():
    assert {
        p.relative_to("notebooks").as_posix() for p in Path("notebooks").rglob("*.ipynb")
    } == ACTIVE
    assert not Path("notebooks/archive").exists()
    for name in ACTIVE:
        cells = json.loads((Path("notebooks") / name).read_text())["cells"]
        for cell in cells:
            source = "".join(cell["source"])
            if cell["cell_type"] != "code":
                continue
            tree = ast.parse(source)
            assert cell["execution_count"] is None and cell["outputs"] == []
            assert not any(token in source for token in ("%run", "runpy", "papermill", "archive/"))
            if name.endswith("04_eval.ipynb"):
                assert "finetuning" not in source
                assert "s20_report" not in source
                assert "ALLOW_COMPARISON" not in source
                assert "compare_s20" not in source
                assert not any(
                    isinstance(n, ast.Attribute) and n.attr == "train" for n in ast.walk(tree)
                )


def test_active_notebooks_have_numbered_context_and_explicit_prerequisites():
    for name in ACTIVE:
        cells = json.loads((Path("notebooks") / name).read_text())["cells"]
        overview = "".join(cells[0]["source"])
        assert all(f"**{label}:**" in overview for label in ("Inputs", "Outputs", "Prerequisites"))
        sections = re.findall(
            r"^## (\d+)\. (.+)$", "\n".join("".join(c["source"]) for c in cells), re.M
        )
        assert [int(n) for n, _ in sections] == list(range(1, len(sections) + 1))
        assert sections[0][1] == "Configuration" and "Summary" in sections[-1][1]
        code = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]
        assert "require_paths(" in code[0]
    assert not Path("notebooks/experiments/v0/01_experiments.ipynb").exists()


def test_documentation_local_links_resolve():
    for path in [Path("README.md"), *Path("notebooks").rglob("README.md")]:
        for link in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if "://" not in link and not link.startswith("#"):
                assert (path.parent / link.split("#")[0]).exists(), (path, link)
