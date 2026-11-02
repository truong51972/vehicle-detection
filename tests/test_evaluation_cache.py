"""Cache metadata fixtures only; these values are not detector-quality evidence."""

import copy

import pytest

from vehicle_detection.common import digest, file_hash, reuse_evaluation, write_json


def test_completed_pair_reuse_rejects_changed_inputs_results_and_plots(tmp_path):
    path = tmp_path / "result.json"
    identity = {
        "dataset": {"jpeg_quality": 95},
        "weights": "sha1",
        "settings": {"conf": 0.001},
        "versions": {"ultralytics": "8.4.198"},
    }
    assert reuse_evaluation(path, identity) is None
    plot = tmp_path / "curve.png"
    plot.write_bytes(b"plot fixture")
    result = {"overall": {"map": 0.42}, "classes": [], "artifacts": {plot.name: file_hash(plot)}}
    original = {"identity": identity, "result": result, "checksum": digest(result)}
    write_json(path, original)
    before = path.stat().st_mtime_ns
    assert reuse_evaluation(path, identity) == result
    assert path.stat().st_mtime_ns == before
    for key, value in [
        ("dataset", {"jpeg_quality": 80}),
        ("weights", "sha2"),
        ("settings", {"conf": 0.25}),
        ("versions", {"ultralytics": "different"}),
    ]:
        changed = copy.deepcopy(identity)
        changed[key] = value
        with pytest.raises(ValueError, match="identity/checksum"):
            reuse_evaluation(path, changed)
        assert reuse_evaluation(path, changed, force=True) is None
    corrupted = copy.deepcopy(original)
    corrupted["result"]["overall"]["map"] = 0.9
    write_json(path, corrupted)
    with pytest.raises(ValueError, match="identity/checksum"):
        reuse_evaluation(path, identity)
    write_json(path, original)
    plot.write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        reuse_evaluation(path, identity)
    plot.unlink()
    with pytest.raises(ValueError, match="Missing evaluation artifact"):
        reuse_evaluation(path, identity)
