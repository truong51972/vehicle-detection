"""Saved-checkpoint selection is authenticated even when native epoch is stripped."""

import pandas as pd
import pytest

from vehicle_detection.s20_report import selected_epoch


def test_selected_epoch_uses_saved_checkpoint_metrics_and_rejects_ambiguity():
    history = pd.DataFrame(
        {
            "epoch": [1, 2, 3],
            "metrics/mAP50-95(B)": [0.1, 0.2, 0.19],
            "val/box_loss": [2.0, 1.9, 1.95],
        }
    )
    metrics = {"metrics/mAP50-95(B)": 0.2, "val/box_loss": 1.9, "fitness": 0.2}
    assert selected_epoch(history, metrics) == 2
    with pytest.raises(ValueError, match="one native history row"):
        selected_epoch(pd.concat([history, history.iloc[[1]]]), metrics)
    with pytest.raises(ValueError, match="Missing checkpoint"):
        selected_epoch(history, {"fitness": 0.2})
