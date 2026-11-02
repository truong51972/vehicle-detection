import copy

import pytest

from vehicle_detection.benchmark import aligned, benchmark, experiment_lock, validate_predictions
from vehicle_detection.common import read_json, write_json
from vehicle_detection.data import CLASSES, prepare


class FakeDetector:
    """Test-only deterministic detector; these are never real model measurements."""

    identity = {"weights_sha256": "synthetic-fixture-only", "classes": CLASSES}
    training_provenance = None

    def __init__(self, fail_after=None):
        self.calls = 0
        self.fail_after = fail_after

    def predict(self, image):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("simulated interruption")
        return [[0, 10, 12, 50, 40, 0.9]]

    def synchronize(self):
        pass

    def hardware(self):
        return {"device": "synthetic-test-only"}


def test_benchmark_reports_and_safe_resume(bdd, tmp_path, config):
    prepared, output = tmp_path / "prepared", tmp_path / "benchmark"
    prepare(bdd, prepared, config)
    config["benchmark"]["conditions"] = ["clean", "noise"]
    with pytest.raises(RuntimeError, match="simulated"):
        benchmark(prepared, None, output, config, detector=FakeDetector(fail_after=10))
    shards = list((output / "records").rglob("*.json"))
    assert shards
    previous = {p: p.read_bytes() for p in shards}
    result = benchmark(prepared, None, output, config, resume=True, detector=FakeDetector())
    assert len(result) == 12
    assert all(r["map50_95"] == 1 and r["delta_map50_95"] == 0 for r in result)
    assert all(p.read_bytes() == content for p, content in previous.items())
    assert len(list((output / "records").rglob("*.json"))) == 48
    assert len((output / "decisions.jsonl").read_text().splitlines()) == 48
    for name in ("comparison.png", "report.md", "results.csv", "results.json", "experiment.json"):
        assert (output / name).is_file()
    assert list((output / "samples").rglob("*.jpg"))
    assert "synthetic" in (output / "report.md").read_text()
    artifact = read_json(output / "results.json")
    assert artifact["strata"]
    assert all(r["small_sample"] for r in artifact["strata"])
    with pytest.raises(ValueError, match="Experiment exists"):
        benchmark(prepared, None, output, config, detector=FakeDetector())
    changed = copy.deepcopy(config)
    changed["policy"]["brightness_below"] = 0.3
    with pytest.raises(ValueError, match="Resume identity mismatch"):
        benchmark(prepared, None, output, changed, resume=True, detector=FakeDetector())
    checkpoint_changed = FakeDetector()
    checkpoint_changed.identity = {**FakeDetector.identity, "weights_sha256": "different"}
    with pytest.raises(ValueError, match="Resume identity mismatch"):
        benchmark(prepared, None, output, config, resume=True, detector=checkpoint_changed)
    damaged = shards[0]
    row = read_json(damaged)
    row["predictions"] = []
    write_json(damaged, row)
    with pytest.raises(ValueError, match="Corrupt/incompatible"):
        benchmark(prepared, None, output, config, resume=True, detector=FakeDetector())


def test_pair_alignment():
    raw = [{"id": "a", "boxes": [[0, 0, 0, 10, 10]]}]
    with pytest.raises(ValueError, match="Unpaired image IDs"):
        aligned({"raw": raw, "adaptive": [{"id": "b", "boxes": []}]}, ["a"])
    with pytest.raises(ValueError, match="Unpaired ground truth"):
        aligned({"raw": raw, "adaptive": [{"id": "a", "boxes": []}]}, ["a"])


def test_lock_prevents_concurrent_writers(tmp_path):
    with experiment_lock(tmp_path):
        with pytest.raises(ValueError, match="Benchmark locked"):
            with experiment_lock(tmp_path):
                pass
    assert not (tmp_path / ".lock").exists()


@pytest.mark.parametrize(
    "prediction",
    [
        [[9, 0, 0, 10, 10, 0.9]],
        [[0, -1, 0, 10, 10, 0.9]],
        [[0, 0, 0, 10, 10, float("nan")]],
        [[0, 10, 0, 0, 10, 0.9]],
    ],
)
def test_invalid_detector_output(prediction):
    with pytest.raises(ValueError, match="Detector returned invalid"):
        validate_predictions(prediction, 80, 64)
