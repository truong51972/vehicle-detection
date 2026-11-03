"""Exercise the actual native metric adapter without a checkpoint download or GPU."""

from types import SimpleNamespace

import numpy as np
import pytest
from test_study import bdd as bdd

torch = pytest.importorskip("torch")
pytest.importorskip("ultralytics")

from vehicle_detection.baseline import (  # noqa: E402
    COCO_TO_VEHICLE,
    VehicleValidator,
    evaluate_pretrained,
    remap_predictions,
    verify_coco_names,
)
from vehicle_detection.data import CLASSES, scan_bdd  # noqa: E402
from vehicle_detection.protocol import build_protocol  # noqa: E402


def coco_names():
    names = dict.fromkeys(range(80), "other")
    names.update({source: CLASSES[target] for source, target in COCO_TO_VEHICLE.items()})
    return names


def test_semantic_mapping_filters_and_keeps_rows_aligned():
    verify_coco_names(coco_names())
    wrong = coco_names()
    wrong[2] = "bicycle"
    with pytest.raises(ValueError, match="semantics"):
        verify_coco_names(wrong)
    pred = {
        "cls": torch.tensor([2.0, 0.0, 7.0, 5.0, 6.0, 3.0, 1.0]),
        "bboxes": torch.arange(28).reshape(7, 4),
        "conf": torch.arange(7),
        "extra": torch.zeros((7, 0)),
    }
    mapped = remap_predictions(pred)
    assert mapped["cls"].tolist() == [0, 1, 2, 3, 4, 5]
    assert mapped["conf"].tolist() == [0, 2, 3, 4, 5, 6]
    assert mapped["bboxes"].tolist() == pred["bboxes"][[0, 2, 3, 4, 5, 6]].tolist()
    empty = remap_predictions({k: v[:0] for k, v in pred.items()})
    assert empty["bboxes"].shape == (0, 4)


def test_native_postprocessing_matching_and_ap_use_vehicle_ids(tmp_path):
    validator = VehicleValidator(
        save_dir=tmp_path, args={"split": "val", "plots": False, "nms": False}
    )
    validator.training = False
    validator.data = {"val": "images/tune", "names": dict(enumerate(CLASSES))}
    validator.dataloader = SimpleNamespace(dataset=SimpleNamespace(labels=[]))
    model = SimpleNamespace(names=coco_names(), end2end=True)
    validator.init_metrics(model)
    assert validator.nc == 6 and validator.metrics.names == dict(enumerate(CLASSES))
    assert validator.confusion_matrix.matrix.shape == (7, 7)
    raw = torch.tensor([[[0.0, 0.0, 10.0, 10.0, 0.99, 2.0], [20.0, 20.0, 30.0, 30.0, 0.9, 0.0]]])
    pred = validator.postprocess(raw)[0]
    assert pred["cls"].tolist() == [0]
    target = {"bboxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "cls": torch.tensor([0.0])}
    matches = validator._process_batch(pred, target)
    assert matches["tp"].shape == (1, 10) and matches["tp"].all()
    validator.metrics.update_stats(
        {
            **matches,
            "target_cls": np.array([0]),
            "target_img": np.array([0]),
            "conf": pred["conf"].numpy(),
            "pred_cls": pred["cls"].numpy(),
            "im_name": "fixture.jpg",
        }
    )
    validator.get_stats()
    assert validator.metrics.box.ap_class_index.tolist() == [0]
    assert validator.metrics.box.map > 0.99
    validator.args.split = "test"
    with pytest.raises(ValueError, match="development validation"):
        validator.init_metrics(model)


def test_final_test_export_rejected_before_any_image_access(bdd, tmp_path, monkeypatch):
    import pandas as pd

    from vehicle_detection import baseline

    records, _, sources = scan_bdd(bdd)
    manifest = build_protocol(records, sources, bdd, [])
    dataset = tmp_path / "wrong_export"
    dataset.mkdir()
    pd.DataFrame([{"id": manifest["splits"]["final_test"][0]["id"], "split": "tune"}]).to_csv(
        dataset / "metadata.csv",
        index=False,
    )

    def no_images(*args, **kwargs):
        pytest.fail("Final-test dataset must fail before deep image verification")

    monkeypatch.setattr(baseline, "verify_alignment", no_images)
    with pytest.raises(ValueError, match="frozen development IDs"):
        evaluate_pretrained(manifest, dataset, tmp_path / "weights.pt", tmp_path / "results")
