"""Native fixture loading is throughput plumbing, never a hardware comparison."""

import pytest
from test_adaptive import frozen as frozen
from test_finetuning import pair_inputs as pair_inputs
from test_study import bdd as bdd

from vehicle_detection.finetuning import TRAIN_SETTINGS
from vehicle_detection.loader_benchmark import benchmark_loader


def test_loader_measures_only_cpu_batches_and_rejects_changed_identity(pair_inputs):
    protocol, subsets, _, _, pair, _, _, output = pair_inputs
    ids = set(subsets["subsets"]["S10"]["ids"])
    rows = [r for r in protocol["splits"]["training_pool"] if r["id"] in ids]
    settings = {**TRAIN_SETTINGS, "workers": 0, "batch": 2, "imgsz": 64, "device": "cpu"}
    args = pair / "raw", rows, "raw_reencoded", None, settings, output
    with pytest.raises(FileNotFoundError, match="MEASURE_LOADERS"):
        benchmark_loader(*args)
    result = benchmark_loader(*args, allow_measurement=True)
    assert result["images_per_second"] > 0
    assert [r["images"] for r in result["epochs"]] == [len(rows)] * 4
    assert benchmark_loader(*args) == result
    settings["batch"] = 1
    with pytest.raises(ValueError, match="identity/checksum"):
        benchmark_loader(*args)
