"""Inspect saved notebook evidence without datasets, Jupyter or a GPU in CI."""

import json
from pathlib import Path


def test_notebook_sources_and_committed_execution_outputs():
    paths = sorted(Path("notebooks").glob("0*.ipynb"))
    assert len(paths) == 2
    for path in paths:
        notebook = json.loads(path.read_text())
        code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
        images = 0
        for i, cell in enumerate(code_cells):
            compile("".join(cell["source"]), f"{path}:cell{i}", "exec")
            assert isinstance(cell["execution_count"], int)
            assert not any(o["output_type"] == "error" for o in cell.get("outputs", []))
            images += sum("image/png" in o.get("data", {}) for o in cell.get("outputs", []))
        assert images >= 3
