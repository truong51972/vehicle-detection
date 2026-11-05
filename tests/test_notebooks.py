"""Active Run-All contracts and byte-preserved historical execution evidence."""

import ast
import json
from pathlib import Path

from vehicle_detection.common import file_hash

ACTIVE = (
    "01_data_init.ipynb",
    "02_evaluation_data_preparation.ipynb",
    "02_data_preprocessing.ipynb",
    "03_preprocessing_experiments.ipynb",
    "03_training.ipynb",
    "04_eval_and_report.ipynb",
)


def test_active_entrypoints_are_independent_and_archives_are_unchanged():
    assert {p.name for p in Path("notebooks").glob("*.ipynb")} == set(ACTIVE)
    for name in ACTIVE:
        notebook = json.loads((Path("notebooks") / name).read_text())
        sources = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
        assert "require_paths" in sources[0]
        for source in sources:
            tree = ast.parse(source)
            assert not any(token in source for token in ("%run", "runpy", "papermill"))
            if name == ACTIVE[-1]:
                assert "finetuning" not in source
                assert not any(
                    isinstance(n, ast.Attribute) and n.attr == "train" for n in ast.walk(tree)
                )
    manifest = json.loads(Path("notebooks/archive/manifest.json").read_text())
    assert {Path(v["archive"]) for v in manifest.values()} == set(
        Path("notebooks/archive").rglob("*.ipynb")
    )
    for record in manifest.values():
        path = Path(record["archive"])
        assert file_hash(path) == record["sha256"]
        notebook = json.loads(path.read_text())
        code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
        images = 0
        status = notebook["metadata"].get("phase2_status", "pending")
        assert status in {"pending", "preparing", "running", "complete"}
        for i, cell in enumerate(code_cells):
            compile("".join(cell["source"]), f"{path}:cell{i}", "exec")
            if status == "complete":
                assert isinstance(cell["execution_count"], int)
            elif cell["execution_count"] is None:
                assert not cell.get("outputs"), "Unexecuted cells cannot claim results"
            assert not any(o["output_type"] == "error" for o in cell.get("outputs", []))
            images += sum("image/png" in o.get("data", {}) for o in cell.get("outputs", []))
        if status == "complete":
            assert images >= 3


def test_active_notebooks_have_navigable_numbered_sections():
    import re

    for name in ACTIVE:
        phase = int(name[:2])
        cells = json.loads((Path("notebooks") / name).read_text())["cells"]
        overview = "".join(cells[0]["source"])
        title, purpose, *_ = overview.split("\n\n")
        assert re.fullmatch(rf"# {phase:02d} — [^\n]+", title)
        assert purpose.strip() and not purpose.startswith(("#", "- "))
        assert all(f"**{label}:**" in overview for label in ("Inputs", "Outputs", "Prerequisites"))
        assert "**Outline**" in overview
        outline = re.findall(r"\[([^]]+)\]\(#([a-z0-9-]+)\)", overview)
        sections, current, subsections = [], None, {}
        for cell in cells[1:]:
            source = "".join(cell["source"])
            if cell["cell_type"] == "code":
                assert current is not None, "Code needs a numbered section for context"
                current["code_cells"] += 1
                continue
            for number, title in re.findall(r"^## (\d+)\. (.+)$", source, re.MULTILINE):
                anchor = re.search(r'<a id="([a-z0-9-]+)"></a>', source)
                assert anchor, "Main sections need stable outline anchors"
                current = {
                    "number": int(number),
                    "title": title,
                    "anchor": anchor[1],
                    "code_cells": 0,
                }
                sections.append(current)
            for parent, child in re.findall(r"^### (\d+)\.(\d+) ", source, re.MULTILINE):
                assert current and int(parent) == current["number"]
                assert int(child) == subsections.get(parent, 0) + 1
                subsections[parent] = int(child)
        assert [s["number"] for s in sections] == list(range(1, len(sections) + 1))
        assert [(s["title"], s["anchor"]) for s in sections] == outline
        assert len({s["anchor"] for s in sections}) == len(sections)
        assert all(s["code_cells"] for s in sections), "Avoid empty main sections"
        assert "Configuration" in sections[0]["title"]
        assert "Summary" in sections[-1]["title"]


def test_pr37_evidence_is_unchanged_with_original_notebook_archived():
    evidence = json.loads(Path("docs/round2-preprocessing-v2/evidence_manifest.json").read_text())
    archives = json.loads(Path("notebooks/archive/manifest.json").read_text())
    for original, expected in evidence["artifacts"].items():
        path = archives[original]["archive"] if original in archives else original
        assert file_hash(path) == expected, original
