"""Active Run-All contracts and byte-preserved historical execution evidence."""

import ast
import json
from pathlib import Path

from vehicle_detection.common import file_hash

ACTIVE = (
    "01_data_init.ipynb",
    "02_preprocessing_research.ipynb",
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
        status = notebook["metadata"].get("phase2_status")
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
