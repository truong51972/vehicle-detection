"""Exercise the actual native metric adapter without a checkpoint download or GPU."""

from types import SimpleNamespace

import numpy as np
import pytest
from test_study import bdd as bdd

torch = pytest.importorskip("torch")
pytest.importorskip("ultralytics")

from vehicle_detection.baseline import (  # noqa: E402
    COCO_TO_FIVE,
    COCO_TO_VEHICLE,
    VehicleValidator,
    remap_predictions,
    verify_coco_names,
)
from vehicle_detection.data import CLASSES  # noqa: E402
from vehicle_detection.five_class import FIVE_CLASSES  # noqa: E402


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


@pytest.mark.parametrize(
    "names,mapping", [(CLASSES, COCO_TO_VEHICLE), (FIVE_CLASSES, COCO_TO_FIVE)]
)
def test_native_postprocessing_matching_and_ap_use_vehicle_ids(tmp_path, names, mapping):
    validator = VehicleValidator(
        save_dir=tmp_path, args={"split": "val", "plots": False, "nms": False}
    )
    validator.training = False
    validator.data = {"val": "images/tune", "names": dict(enumerate(names))}
    validator.dataloader = SimpleNamespace(dataset=SimpleNamespace(labels=[]))
    model = SimpleNamespace(names=coco_names(), end2end=True)
    validator.init_metrics(model)
    assert validator.nc == len(names) and validator.metrics.names == dict(enumerate(names))
    assert validator.confusion_matrix.matrix.shape == (len(names) + 1, len(names) + 1)
    sources = list(mapping) + [0] + ([6] if names == FIVE_CLASSES else [])
    raw = torch.tensor(
        [[[i * 20.0, 0.0, i * 20.0 + 10, 10.0, 0.99, float(c)] for i, c in enumerate(sources)]]
    )
    pred = validator.postprocess(raw)[0]
    assert pred["cls"].tolist() == list(range(len(names)))
    target = {"bboxes": raw[0, : len(names), :4], "cls": torch.arange(len(names)).float()}
    matches = validator._process_batch(pred, target)
    assert matches["tp"].shape == (len(names), 10) and matches["tp"].all()
    validator.metrics.update_stats(
        {
            **matches,
            "target_cls": np.arange(len(names)),
            "target_img": np.arange(len(names)),
            "conf": pred["conf"].numpy(),
            "pred_cls": pred["cls"].numpy(),
            "im_name": "fixture.jpg",
        }
    )
    validator.get_stats()
    assert validator.metrics.box.ap_class_index.tolist() == list(range(len(names)))
    assert validator.metrics.box.map > 0.99
    validator.args.split = "test"
    with pytest.raises(ValueError, match="development validation"):
        validator.init_metrics(model)
