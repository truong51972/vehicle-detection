"""Round 2 reuses frozen data and never opens or regenerates final-test inputs."""

import copy
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
from test_study import bdd as bdd

from vehicle_detection.adaptive import calibrate_policy, export_pair
from vehicle_detection.common import digest, file_hash, read_json, write_json
from vehicle_detection.data import CLASSES, scan_bdd
from vehicle_detection.datasets import export_development_pair
from vehicle_detection.evaluation_data import (
    audit_data,
    evaluation_sets,
    scenario_image,
    scenario_source,
    validate_audit,
)
from vehicle_detection.five_class import FIVE_CLASSES, derive_labels, support_audit
from vehicle_detection.offline import export_variant, verify_alignment
from vehicle_detection.preprocess import gamma
from vehicle_detection.protocol import build_protocol, development_records, summarize
from vehicle_detection.stress import corrupt
from vehicle_detection.subsets import build_subsets, training_records


@pytest.fixture
def prepared(bdd, tmp_path):
    source = bdd / "bdd100k_labels_images_train.json"
    frames = read_json(source)
    for i in range(6, 24):
        image = bdd / "images/100k/train" / f"train_{i}.jpg"
        assert cv2.imwrite(
            str(image), np.random.default_rng(i).integers(0, 256, (64, 80, 3), dtype=np.uint8)
        )
        frame = {**frames[0], "name": image.name}
        if i % 3 == 0:
            frame["labels"] = [frames[0]["labels"][3]]  # Railway-only becomes negative.
        frames.append(frame)
    write_json(source, frames)
    records, _, sources = scan_bdd(bdd)
    protocol = build_protocol(records, sources, bdd, [])
    subsets = build_subsets(protocol, attempts=2)
    policy = calibrate_policy(protocol, bdd, sample_size=2, per_operator=1)
    assert len({r["group"] for r in protocol["splits"]["calibration"]}) >= 2
    partitions = {
        "training": training_records(protocol, subsets, "S10", records, bdd),
        "development": development_records(protocol, records, bdd),
    }
    five = tmp_path / "five"
    conversions, identities = {}, {}
    lookup = {r["id"]: r for rows in protocol["splits"].values() for r in rows}
    for name, parts in partitions.items():
        original = tmp_path / "six" / name
        conversions[name] = original
        write_json(
            original / "conversion.json", {"classes": CLASSES, "box_convention": "continuous"}
        )
        frozen = {split: [lookup[r["id"]] for r in rows] for split, rows in parts.items()}
        for split, rows in frozen.items():
            for row in rows:
                label = original / "labels" / split / f"{row['id']}.txt"
                label.parent.mkdir(parents=True, exist_ok=True)
                label.write_text(
                    "".join(
                        f"{c} {(x1 + x2) / 2 / row['width']:.6f} "
                        f"{(y1 + y2) / 2 / row['height']:.6f} "
                        f"{(x2 - x1) / row['width']:.6f} {(y2 - y1) / row['height']:.6f}\n"
                        for c, x1, y1, x2, y2 in row["boxes"]
                    )
                )
        conversion = five / "conversion" / ("S10" if name == "training" else name)
        identities[name] = derive_labels(original, conversion, frozen)
        if name == "training":
            export_pair(parts, conversion, five / "datasets/S10", policy, protocol, subsets, "S10")
        else:
            raw = five / "development/raw"
            export_variant(parts, conversion, raw, "raw_reencoded", policy["recipes"])
            verify_alignment([raw])
            export_development_pair(protocol, policy, raw, conversion, five / "development")
    write_json(
        five / "summary.json",
        {
            "source_protocol_id": protocol["manifest_id"],
            "source_subset_id": subsets["manifest_id"],
            "taxonomy": FIVE_CLASSES,
            "role_support_annotation_only": {
                role: support_audit(rows) for role, rows in protocol["splits"].items()
            },
            "training_conversion": identities["training"],
            "development_conversion": identities["development"],
            "training_pair": read_json(five / "datasets/S10/pair.json"),
            "development_pair": read_json(five / "development/pair.json"),
        },
    )
    return protocol, bdd, five, subsets, conversions, policy


def snapshot(roots):
    return {
        str(p): (file_hash(p), p.stat().st_mtime_ns)
        for root in roots
        for p in Path(root).rglob("*")
        if p.is_file()
    }


def audit_roots(prepared):
    _, root, five, _, conversions, _ = prepared
    return {
        "data": root,
        "five": five,
        "training_conversion": conversions["training"],
        "development_conversion": conversions["development"],
    }


def test_calibration_halves_are_complete_disjoint_groups_and_reproducible(prepared):
    protocol, *_ = prepared
    before = copy.deepcopy(protocol)
    first = evaluation_sets(protocol)
    assert first == evaluation_sets(protocol)
    assert protocol == before
    sets = {k: set(ids) for k, ids in first["sets"].items()}
    exploration, confirmation = sets["calibration_exploration"], sets["calibration_confirmation"]
    assert exploration and confirmation and not exploration & confirmation
    assert exploration | confirmation == {r["id"] for r in protocol["splits"]["calibration"]}
    assert sets["development_validation"] == {
        r["id"] for r in protocol["splits"]["development_validation"]
    }
    assert not set.union(*sets.values()) & {r["id"] for r in protocol["splits"]["final_test"]}
    assert len(first["scenarios"]) == 10
    assert first["scenarios"][0]["synthetic"] is False
    assert all(s["synthetic"] for s in first["scenarios"][1:])
    assert first["manifest_id"] != evaluation_sets(protocol, seed=7)["manifest_id"]
    # Multiple images in one merged group stay together.
    row = protocol["splits"]["calibration"][0]
    protocol["splits"]["calibration"].append({**row, "id": "same-group-negative", "boxes": []})
    protocol["summaries"]["calibration"] = summarize(protocol["splits"]["calibration"])
    protocol["manifest_id"] = digest({k: v for k, v in protocol.items() if k != "manifest_id"})
    sets = evaluation_sets(protocol)["sets"]
    assert any({row["id"], "same-group-negative"} <= set(ids) for ids in sets.values())
    for seed in (True, -1, 1.5):
        with pytest.raises(ValueError, match="Seed"):
            evaluation_sets(protocol, seed=seed)
    for member in protocol["splits"]["calibration"]:
        member["group"] = row["group"]
    protocol["manifest_id"] = digest({k: v for k, v in protocol.items() if k != "manifest_id"})
    with pytest.raises(ValueError, match="at least two calibration groups"):
        evaluation_sets(protocol)


def test_scenarios_use_original_pixels_shared_encoding_and_reject_final(prepared, monkeypatch):
    protocol, root, *_ = prepared
    row = protocol["splits"]["calibration"][0]
    source = root / row["source_path"]
    before = (file_hash(source), source.stat().st_mtime_ns)
    scenarios = evaluation_sets(protocol)["scenarios"]
    outputs = [scenario_image(row, root, scenario) for scenario in scenarios]
    for scenario, output in zip(scenarios, outputs, strict=True):
        np.testing.assert_array_equal(output, scenario_image(row, root, scenario))
        assert output.shape == (row["height"], row["width"], 3)
        assert output.dtype == np.uint8
    clean = cv2.imread(str(source))
    np.testing.assert_array_equal(outputs[0], clean)
    assert (file_hash(source), source.stat().st_mtime_ns) == before
    monkeypatch.setattr(
        "vehicle_detection.evaluation_data.file_hash", lambda p: pytest.fail("Pixel access")
    )
    with pytest.raises(ValueError, match="non-final"):
        scenario_image(protocol["splits"]["final_test"][0], root, scenarios[0])
    with pytest.raises(ValueError, match="parameters"):
        scenario_image(row, root, {**scenarios[1], "params": {"factor": 999}})
    for severity in (False, 0, 4):
        with pytest.raises(ValueError, match="family/severity"):
            scenario_image(row, root, {**scenarios[1], "severity": severity})


def test_raw_and_treated_branches_encode_once_after_corruption(prepared, monkeypatch):
    protocol, root, *_ = prepared
    row = protocol["splits"]["calibration"][0]
    scenarios = evaluation_sets(protocol)["scenarios"]
    imencode, imdecode = cv2.imencode, cv2.imdecode
    for scenario in (scenarios[0], scenarios[5]):
        with monkeypatch.context() as patch:
            patch.setattr(cv2, "imencode", lambda *a, **k: pytest.fail("Premature encoding"))
            patch.setattr(cv2, "imdecode", lambda *a, **k: pytest.fail("Premature decoding"))
            pixels = scenario_image(row, root, scenario)
        expected = corrupt(
            cv2.imread(str(root / row["source_path"])),
            scenario["family"],
            scenario["severity"],
            row["id"],
        )
        np.testing.assert_array_equal(pixels, expected)
        for image, reference in ((pixels, expected), (gamma(pixels, 0.7), gamma(expected, 0.7))):
            ok, actual_jpeg = imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            assert ok
            ok, expected_jpeg = imencode(".jpg", reference, [cv2.IMWRITE_JPEG_QUALITY, 95])
            assert ok and actual_jpeg.tobytes() == expected_jpeg.tobytes()
            assert imdecode(actual_jpeg, cv2.IMREAD_COLOR).shape == expected.shape


def test_checked_source_is_reusable_without_repeated_io(prepared, monkeypatch):
    protocol, root, *_ = prepared
    row = protocol["splits"]["calibration"][0]
    source = scenario_source(row, root)
    before = source.copy()
    scenarios = evaluation_sets(protocol)["scenarios"]
    with monkeypatch.context() as patch:
        patch.setattr(
            "vehicle_detection.evaluation_data.file_hash", lambda p: pytest.fail("Repeated hash")
        )
        patch.setattr(
            "vehicle_detection.evaluation_data.read_image", lambda p: pytest.fail("Repeated decode")
        )
        for scenario in scenarios:
            pixels = corrupt(source, scenario["family"], scenario["severity"], row["id"])
            assert pixels.shape == source.shape
        with pytest.raises(ValueError, match="non-final"):
            scenario_source(protocol["splits"]["final_test"][0], root)
    np.testing.assert_array_equal(source, before)
    (root / row["source_path"]).write_bytes(b"changed after loading")
    with pytest.raises(ValueError, match="Source image changed"):
        scenario_source(row, root)


def test_saved_audit_replays_after_relocation_and_still_rejects_changed_bytes(
    prepared, tmp_path, monkeypatch
):
    protocol, *_ = prepared
    audit = audit_data(*prepared)
    roots = audit_roots(prepared)
    relocated = {}
    for name, root in roots.items():
        target = tmp_path / "relocated" / name
        shutil.copytree(root, target)
        root.rename(root.with_name(root.name + "-unavailable"))
        relocated[name] = target
    assert validate_audit(audit, protocol, roots=relocated) == audit
    original_json = str(audit)
    assert str(tmp_path) not in original_json
    for files in audit["artifacts"].values():
        assert all(not Path(path).is_absolute() for path in files)
    with pytest.raises(ValueError, match="configured audit root"):
        validate_audit(audit, protocol, roots={})
    outside = tmp_path / "outside.json"
    outside.write_text("must not be read")
    (relocated["data"] / "escape.json").symlink_to(outside)
    with monkeypatch.context() as patch:
        patch.setattr(
            "vehicle_detection.evaluation_data.file_hash", lambda p: pytest.fail("Escaped read")
        )
        for relative in ("../outside.json", str(outside), "escape.json"):
            unsafe = {**audit, "artifacts": {"data": {relative: "unused"}}}
            unsafe["audit_id"] = digest({k: v for k, v in unsafe.items() if k != "audit_id"})
            with pytest.raises(ValueError, match="root-relative|escapes"):
                validate_audit(unsafe, protocol, roots=relocated)
    (relocated["five"] / "summary.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="prerequisite changed"):
        validate_audit(audit, protocol, roots=relocated)


def test_audit_is_read_only_and_final_images_remain_unopened(prepared, monkeypatch):
    import io

    protocol, root, five, _, conversions, _ = prepared
    roots = [root, five, *conversions.values()]
    before = snapshot(roots)
    original_open = io.open
    original_read = cv2.imread

    def guarded_open(path, mode="r", *args, **kwargs):
        if isinstance(path, (str, Path)):
            p = Path(path)
            assert not (p.suffix == ".jpg" and "val" in p.parts), "Final-test image opened"
            assert not any(c in mode for c in "wax+"), "Audit wrote a historical file"
        return original_open(path, mode, *args, **kwargs)

    def guarded_read(path, *args, **kwargs):
        assert "val" not in Path(path).parts
        return original_read(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(io, "open", guarded_open)
        patch.setattr(cv2, "imread", guarded_read)
        audit = audit_data(*prepared)
        assert audit == audit_data(*prepared)
        assert validate_audit(audit, protocol, roots=audit_roots(prepared)) == audit
    assert snapshot(roots) == before
    assert audit["role_support_annotation_only"]["final_test"]["images"] == 3
    metadata = five / "development/raw/metadata.csv"
    metadata.write_text(metadata.read_text() + "\n")
    with pytest.raises(ValueError, match="prerequisite changed"):
        validate_audit(audit, protocol, roots=audit_roots(prepared))


@pytest.mark.parametrize(
    "target", ["source", "annotations", "five-label", "six-label", "railway-coordinate", "export"]
)
def test_audit_rejects_stale_inputs_without_repair(prepared, target):
    protocol, root, five, _, conversions, _ = prepared
    row = protocol["splits"]["development_validation"][0]
    paths = {
        "source": root / row["source_path"],
        "annotations": root / protocol["annotation_sources"]["train"]["path"],
        "five-label": five / "conversion/development/labels/tune" / f"{row['id']}.txt",
        "six-label": conversions["development"] / "labels/tune" / f"{row['id']}.txt",
        "railway-coordinate": conversions["development"] / "labels/tune" / f"{row['id']}.txt",
        "export": five / "development/raw/images/tune" / f"{row['id']}.jpg",
    }
    path = paths[target]
    if target == "railway-coordinate":
        lines = path.read_text().splitlines()
        index = next(i for i, line in enumerate(lines) if line.split()[0] == "3")
        boxes = lines[index].split()
        boxes[1] = "0.999"
        lines[index] = " ".join(boxes)
        path.write_text("\n".join(lines) + "\n")
    else:
        path.write_bytes(b"changed")
    changed = path.read_bytes()
    with pytest.raises(ValueError):
        audit_data(*prepared)
    assert path.read_bytes() == changed


def test_two_round2_notebooks_run_independently_and_reuse_saved_inputs(prepared, tmp_path):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    protocol, root, five, subsets, conversions, policy = prepared
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").touch()
    (project / "notebooks").mkdir()
    frozen = project / "runs/frozen"
    write_json(frozen / "protocol.json", protocol)
    write_json(frozen / "subsets.json", subsets)
    write_json(frozen / "policy.json", policy)
    before = snapshot([root, five, *conversions.values()])
    guard = """from unittest.mock import patch
import cv2, io
from pathlib import Path
original_read, original_open = cv2.imread, io.open
def safe_read(path, *args, **kwargs):
    assert "val" not in Path(path).parts, "Reserved image decoded"
    return original_read(path, *args, **kwargs)
def safe_open(path, *args, **kwargs):
    if isinstance(path, (str, Path)):
        assert not (Path(path).suffix == ".jpg" and "val" in Path(path).parts)
    return original_open(path, *args, **kwargs)
patch.object(cv2, "imread", safe_read).start()
patch.object(io, "open", safe_open).start()
for target in ["vehicle_detection.protocol.build_protocol",
               "vehicle_detection.data.convert_labels", "vehicle_detection.subsets.build_subsets",
               "vehicle_detection.five_class.derive_labels",
               "vehicle_detection.adaptive.calibrate_policy",
               "vehicle_detection.offline.export_variant"]:
    patch(target, side_effect=AssertionError("Upstream work triggered")).start()
"""

    def run(name, second=False, cwd=None, settings=""):
        notebook = nbformat.read(Path("notebooks") / name, as_version=4)
        notebook.cells.insert(1, nbformat.v4.new_code_cell(guard))
        config = next(
            c for c in notebook.cells if c.cell_type == "code" and "require_paths(" in c.source
        )
        position = config.source.index("require_paths(")
        values = {
            "DATA_ROOT": root,
            "PROTOCOL_PATH": frozen / "protocol.json",
            "OUTPUT": project / ("evaluation" if second else "audit"),
            "SUBSETS_PATH": frozen / "subsets.json",
            "POLICY_PATH": frozen / "policy.json",
            "FIVE_ROOT": five,
            "TRAIN_CONVERSION": conversions["training"],
            "DEV_CONVERSION": conversions["development"],
        }
        if second:
            values["AUDIT_PATH"] = project / "audit/audit.json"
        overrides = "\n".join(f"{key} = Path({str(value)!r})" for key, value in values.items())
        overrides += "\n" + settings
        config.source = config.source[:position] + overrides + "\n" + config.source[position:]
        if second:
            notebook.cells.insert(
                2,
                nbformat.v4.new_code_cell(
                    'patch("vehicle_detection.data.scan_bdd",\n'
                    '      side_effect=AssertionError("Source rescan")).start()\n'
                    'patch("vehicle_detection.evaluation_data.audit_data",\n'
                    '      side_effect=AssertionError("Audit rerun")).start()'
                ),
            )
        nbclient.NotebookClient(
            notebook,
            timeout=120,
            kernel_name="python3",
            resources={"metadata": {"path": str(cwd or project)}},
        ).execute()
        assert all(c.execution_count is not None for c in notebook.cells if c.cell_type == "code")

    with pytest.raises(nbclient.exceptions.CellExecutionError, match="Missing required artifacts"):
        run("02_evaluation_data_preparation.ipynb", second=True)
    run("01_data_init.ipynb")
    run("02_evaluation_data_preparation.ipynb", second=True, cwd=project / "notebooks")
    outputs = snapshot([project / "audit", project / "evaluation"])
    run("01_data_init.ipynb", cwd=project / "notebooks")
    run("02_evaluation_data_preparation.ipynb", second=True)
    assert snapshot([root, five, *conversions.values()]) == before
    assert {k: v[0] for k, v in snapshot([project / "audit", project / "evaluation"]).items()} == {
        k: v[0] for k, v in outputs.items()
    }
    manifest = read_json(project / "evaluation/evaluation_sets.json")
    assert manifest["data_audit_id"] == read_json(project / "audit/audit.json")["audit_id"]
    outputs = snapshot([project / "audit", project / "evaluation"])
    with pytest.raises(nbclient.exceptions.CellExecutionError, match="new output root"):
        run("01_data_init.ipynb", settings="QUALITY_SAMPLE_PER_ROLE = 2")
    with pytest.raises(nbclient.exceptions.CellExecutionError, match="new version/root"):
        run(
            "02_evaluation_data_preparation.ipynb",
            second=True,
            settings="QUALITY_SAMPLE_PER_SET = 1",
        )
    assert snapshot([project / "audit", project / "evaluation"]) == outputs
    # Relocate the whole fixture and make the original paths unavailable, without rewriting bytes.
    moved = tmp_path / "moved"
    for directory in (project, root, five, tmp_path / "six"):
        shutil.copytree(directory, moved / directory.name)
        directory.rename(directory.with_name(directory.name + "-unavailable"))
    project, root, five = moved / "project", moved / root.name, moved / "five"
    frozen = project / "runs/frozen"
    conversions = {name: moved / "six" / name for name in conversions}
    saved_audit = (project / "audit/audit.json").read_bytes()
    saved_manifest = (project / "evaluation/evaluation_sets.json").read_bytes()
    run("02_evaluation_data_preparation.ipynb", second=True, cwd=project / "notebooks")
    assert (project / "audit/audit.json").read_bytes() == saved_audit
    assert (project / "evaluation/evaluation_sets.json").read_bytes() == saved_manifest
