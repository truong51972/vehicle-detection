"""Development boundaries and cache failures use fixtures, never claimed GPU results."""

import pytest
from test_adaptive import frozen as frozen
from test_finetuning import pair_inputs as pair_inputs
from test_study import bdd as bdd

from vehicle_detection import evaluation
from vehicle_detection.common import digest, file_hash, read_json, write_json
from vehicle_detection.datasets import export_development_pair
from vehicle_detection.evaluation import completed_checkpoints, evaluate_development
from vehicle_detection.finetuning import prepare_pair


def test_completed_training_receipts_retain_original_code_and_reject_stale_weights(pair_inputs):
    pair = prepare_pair(*pair_inputs)
    output = pair_inputs[-1]
    for branch in ("raw", "adaptive"):
        run = output / branch
        identity = {"pair": pair, "branch": branch, "data": str(output / f"{branch}.yaml")}
        identity["run_id"] = digest(identity)
        write_json(run / "identity.json", identity)
        artifacts = {}
        for name in ("weights/best.pt", "weights/last.pt", "results.csv"):
            path = run / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"test receipt fixture")
            artifacts[name] = file_hash(path)
        receipt = {
            "run_id": identity["run_id"],
            "artifacts": artifacts,
            "budget": {"epochs": 1},
            "final_test_accessed": False,
        }
        receipt["checksum"] = digest(receipt)
        write_json(run / "complete.json", receipt)
    paths = completed_checkpoints(output, pair_inputs[0])
    assert set(paths) == {"raw", "adaptive"}
    paths["raw"].write_bytes(b"stale")
    with pytest.raises(ValueError, match="artifact changed"):
        completed_checkpoints(output, pair_inputs[0])


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
    with pytest.raises(ValueError, match="identity/checksum"):
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
