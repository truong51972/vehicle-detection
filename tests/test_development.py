"""Development boundaries and cache failures use fixtures, never claimed GPU results."""

import pytest
from test_adaptive import frozen as frozen
from test_finetuning import pair_inputs as pair_inputs
from test_study import bdd as bdd

from vehicle_detection import evaluation
from vehicle_detection.common import digest, file_hash, read_json, write_json
from vehicle_detection.data import CLASSES
from vehicle_detection.datasets import export_development_pair
from vehicle_detection.evaluation import evaluate_development

# The historical evaluator is migrated separately in issue #22.
pytestmark = pytest.mark.parametrize("pair_inputs", [CLASSES], indirect=True)


def test_explicit_checkpoint_has_no_native_or_legacy_receipt_dependency(pair_inputs):
    protocol, _, policy, _, _, raw, checkpoint, output = pair_inputs
    assert not (output / "raw/complete.json").exists()
    assert not (output / "raw/args.yaml").exists()
    with pytest.raises(FileNotFoundError, match="ALLOW_INFERENCE"):
        evaluate_development(protocol, policy, raw, checkpoint, output / "evaluation")


def test_development_pair_and_cache_reuse_never_repair_missing_inputs(pair_inputs, monkeypatch):
    protocol, _, policy, _, _, raw, checkpoint, output = pair_inputs
    inputs = export_development_pair(
        protocol, policy, raw, raw.parent / "conversion", output / "dev"
    )
    assert set(inputs) == {"raw", "adaptive"}
    monkeypatch.setattr(evaluation, "PRETRAINED_SHA256", file_hash(checkpoint))
    destination = output / "evaluation"
    args = protocol, policy, raw, checkpoint, destination
    with pytest.raises(FileNotFoundError, match="ALLOW_INFERENCE"):
        evaluate_development(*args, pretrained=True)

    def no_model(*args, **kwargs):
        raise RuntimeError("No model in this fixture")

    monkeypatch.setattr(evaluation, "YOLO", no_model)
    with pytest.raises(RuntimeError, match="No model"):
        evaluate_development(*args, pretrained=True, allow_inference=True)
    identity = read_json(destination / "identity.json")
    result = {"artifacts": {}, "fixture": True}
    write_json(
        destination / "native_result.json",
        {
            "identity": identity,
            "result": result,
            "checksum": digest(result),
        },
    )
    assert evaluate_development(*args, pretrained=True) == result
    # Original six-class result files can still be replayed under their pinned identity.
    legacy = {**identity, "version": "development-native-v1"}
    legacy.pop("class_names")
    write_json(
        destination / "native_result.json",
        {
            "identity": legacy,
            "result": result,
            "checksum": digest(result),
        },
    )
    assert evaluate_development(*args, pretrained=True) == result
    with pytest.raises(ValueError, match="Historical evaluation cache"):
        evaluate_development(*args, pretrained=True, settings={"batch": 1})
    checkpoint.unlink()
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        evaluate_development(*args, pretrained=True)


def test_final_role_and_changed_manifest_rejected_before_pixels(pair_inputs, monkeypatch):
    import pandas as pd

    from vehicle_detection import datasets

    protocol, _, policy, _, _, raw, checkpoint, output = pair_inputs

    def no_pixels(*args, **kwargs):
        pytest.fail("Wrong roles must fail before pixel verification")

    monkeypatch.setattr(datasets, "verify_alignment", no_pixels)
    meta_path = raw / "metadata.csv"
    meta = pd.read_csv(meta_path)
    meta["id"] = protocol["splits"]["final_test"][0]["id"]
    meta.to_csv(meta_path, index=False)
    with pytest.raises(ValueError, match="frozen role IDs"):
        evaluate_development(protocol, policy, raw, checkpoint, output)
    protocol["seed"] = -1
    with pytest.raises(ValueError, match="checksum"):
        evaluate_development(protocol, policy, raw, checkpoint, output)
