import copy
from pathlib import Path

import numpy as np
import pytest
from test_study import bdd as bdd
from test_study import recipes as recipes

from vehicle_detection import adaptive, preprocess
from vehicle_detection.adaptive import (
    calibrate_policy,
    export_pair,
    freeze_policy,
    paired_quality,
    validate_policy,
)
from vehicle_detection.common import digest, read_json, write_json
from vehicle_detection.data import scan_bdd
from vehicle_detection.protocol import build_protocol, summarize
from vehicle_detection.subsets import (
    build_subsets,
    freeze_subsets,
    training_records,
    validate_subsets,
)


@pytest.fixture
def frozen(bdd):
    records, _, sources = scan_bdd(bdd)
    return build_protocol(records, sources, bdd, []), records


def test_nested_group_stratification_reproducible_and_immutable(frozen, tmp_path):
    protocol, _ = frozen
    # Unequal group sizes and rare/negative conditions must survive as whole groups.
    template = protocol["splits"]["training_pool"][0]
    pool = []
    for i in range(120):
        row = copy.deepcopy(template)
        row.update(
            id=f"synthetic-{i}",
            group=f"group-{i // 3}",
            source_group=f"source-{i // 3}",
            sha256=digest(i),
            dhash64=digest([i]),
        )
        row["attributes"]["weather"] = "rainy" if i % 7 == 0 else "clear"
        if i % 11 == 0:
            row["boxes"] = []
        pool.append(row)
    protocol["splits"]["training_pool"] = pool
    protocol["summaries"]["training_pool"] = summarize(pool)
    protocol["manifest_id"] = digest({k: v for k, v in protocol.items() if k != "manifest_id"})
    subsets = build_subsets(protocol, attempts=8, include_s40=True)
    reversed_protocol = copy.deepcopy(protocol)
    reversed_protocol["splits"]["training_pool"].reverse()
    reversed_protocol["manifest_id"] = digest(
        {k: v for k, v in reversed_protocol.items() if k != "manifest_id"}
    )
    second = build_subsets(reversed_protocol, attempts=8, include_s40=True)
    assert subsets["subsets"] == second["subsets"]
    assert [len(s["ids"]) for s in subsets["subsets"].values()] == [12, 24, 48]
    assert any(r["feature"].startswith("joint_presence:") for r in subsets["diagnostics"])
    assert any(r["feature"] == "negative" for r in subsets["diagnostics"])
    freeze_subsets(tmp_path / "subsets.json", subsets, protocol)
    freeze_subsets(tmp_path / "subsets.json", subsets, protocol)
    changed = copy.deepcopy(subsets)
    changed["subsets"]["S10"]["ids"].pop()
    with pytest.raises(ValueError, match="checksum"):
        validate_subsets(changed, protocol)
    changed["manifest_id"] = digest({k: v for k, v in changed.items() if k != "manifest_id"})
    with pytest.raises(ValueError, match="whole groups"):
        validate_subsets(changed, protocol)
    with pytest.raises(ValueError, match="attempts"):
        build_subsets(protocol, attempts=0)


def test_calibration_only_and_stale_policy_guards(frozen, bdd, monkeypatch, tmp_path):
    protocol, _ = frozen
    allowed = {str((bdd / r["source_path"]).resolve()) for r in protocol["splits"]["calibration"]}
    original_read = adaptive.read_image

    def calibration_only(path):
        assert str(Path(path).resolve()) in allowed
        return original_read(path)

    monkeypatch.setattr(adaptive, "read_image", calibration_only)
    policy = calibrate_policy(protocol, bdd, sample_size=3, per_operator=1)
    assert policy == calibrate_policy(protocol, bdd, sample_size=3, per_operator=1)
    assert policy["recipes"]["adaptive"]["max_filters"] == 1
    path = tmp_path / "policy.json"
    freeze_policy(path, policy, protocol)
    freeze_policy(path, policy, protocol)
    assert read_json(path) == policy
    changed = copy.deepcopy(policy)
    changed["recipes"]["gamma"]["gamma"] = 0.1
    with pytest.raises(ValueError, match="checksum"):
        validate_policy(changed, protocol)
    changed["policy_id"] = digest({k: v for k, v in changed.items() if k != "policy_id"})
    with pytest.raises(ValueError, match="Immutable"):
        freeze_policy(path, changed, protocol)
    changed = copy.deepcopy(policy)
    changed["calibration_ids"] = [protocol["splits"]["development_validation"][0]["id"]]
    changed["policy_id"] = digest({k: v for k, v in changed.items() if k != "policy_id"})
    with pytest.raises(ValueError, match="only frozen calibration"):
        validate_policy(changed, protocol)
    Path(next(iter(allowed))).write_bytes(b"changed")
    with pytest.raises(ValueError, match="source changed"):
        calibrate_policy(protocol, bdd)


@pytest.mark.parametrize("route", ["gamma", "clahe", "denoise", "sharpen", "skip"])
def test_deterministic_pixel_only_routing_and_noop(route, recipes, monkeypatch):
    recipes["adaptive"]["max_filters"] = 1
    scores = dict(brightness=0.5, contrast=0.3, noise=0.0, sharpness=200.0)
    if route == "gamma":
        scores["brightness"] = 0.1
    elif route == "clahe":
        scores["contrast"] = 0.05
    elif route == "denoise":
        scores["noise"] = 0.1
    elif route == "sharpen":
        scores["sharpness"] = 50.0
    monkeypatch.setattr(preprocess, "quality", lambda image: scores)
    image = np.full((24, 32, 3), 60, dtype=np.uint8)
    before = image.copy()
    result, decisions, _ = preprocess.apply_recipe(image, "adaptive", recipes)
    assert [d["name"] for d in decisions] == ([] if route == "skip" else [route])
    assert result.shape == image.shape and result.dtype == image.dtype
    np.testing.assert_array_equal(image, before)
    np.testing.assert_array_equal(result, preprocess.apply_recipe(image, "adaptive", recipes)[0])
    if route == "skip":
        np.testing.assert_array_equal(result, image)


@pytest.mark.parametrize("skip", [False, True])
def test_pair_annotation_alignment_reuse_and_changed_ids(frozen, bdd, tmp_path, skip):
    protocol, records = frozen
    subsets = build_subsets(protocol, attempts=2)
    policy = calibrate_policy(protocol, bdd, sample_size=2, per_operator=1)
    if skip:
        policy["recipes"]["adaptive"].update(
            brightness_below=0.0,
            contrast_below=0.0,
            noise_above=1.0,
            sharpness_min=0.0,
            sharpness_below=0.0,
        )
        policy["policy_id"] = digest({k: v for k, v in policy.items() if k != "policy_id"})
    partitions = training_records(protocol, subsets, "S10", records, bdd)
    converted = tmp_path / "conversion"
    write_json(converted / "conversion.json", {"fixture": True})
    for row in partitions["train"]:
        path = converted / "labels/train" / f"{row['id']}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("0 0.4 0.4 0.2 0.2\n")
    output = tmp_path / "S10"
    first = export_pair(partitions, converted, output, policy, protocol, subsets, "S10")
    second = export_pair(partitions, converted, output, policy, protocol, subsets, "S10")
    assert all(t.reused.all() for t in second.values())
    assert first["raw"].label_sha256.tolist() == first["adaptive"].label_sha256.tolist()
    assert first["raw"].encoding.tolist() == first["adaptive"].encoding.tolist()
    if skip:
        assert first["raw"].output_sha256.tolist() == first["adaptive"].output_sha256.tolist()
    assert len(paired_quality(output)) == len(partitions["train"])
    assert not list((output / "raw/images/test").iterdir())
    assert not list((output / "adaptive/images/tune").iterdir())
    with pytest.raises(ValueError, match="exactly the selected"):
        export_pair({"train": []}, converted, output, policy, protocol, subsets, "S10")
    changed_policy = copy.deepcopy(policy)
    changed_policy["recipes"]["gamma"]["gamma"] = 0.9
    changed_policy["policy_id"] = digest(
        {k: v for k, v in changed_policy.items() if k != "policy_id"}
    )
    with pytest.raises(ValueError, match="Pair identity changed"):
        export_pair(partitions, converted, output, changed_policy, protocol, subsets, "S10")
    changed = copy.deepcopy(partitions)
    changed["train"][0]["labels"] = []
    with pytest.raises(ValueError, match="annotations/geometry"):
        export_pair(changed, converted, output, policy, protocol, subsets, "S10")
    (output / "adaptive/labels/train" / f"{partitions['train'][0]['id']}.txt").write_text("")
    with pytest.raises(ValueError, match="changed"):
        paired_quality(output)
