"""Tiny generated BDD fixture: no network, external data, torch or weights."""

import json

import cv2
import numpy as np
import pytest

from vehicle_detection.config import load_config


@pytest.fixture
def config():
    result = load_config()
    result["benchmark"].update(warmup=1, samples=1)
    result["data"]["quality_samples"] = 4
    return result


@pytest.fixture
def bdd(tmp_path):
    root = tmp_path / "nested-kaggle"
    for split, count in (("train", 8), ("val", 4)):
        images = root / "download" / "bdd100k" / "images" / "100k" / split
        images.mkdir(parents=True)
        annotations = []
        for index in range(count):
            name = f"{split}_{index}.png"
            image = np.full((64, 80, 3), 30 + 12 * index + (split == "val"), np.uint8)
            image[12:40, 10:50] = [110, 180, 240]
            assert cv2.imwrite(str(images / name), image)
            annotations.append(
                {
                    "name": name,
                    "attributes": {
                        "weather": "clear" if index % 2 else "rainy",
                        "timeofday": "night" if index % 2 else "daytime",
                        "scene": "city street",
                    },
                    "labels": [
                        {"category": "car", "box2d": {"x1": 10, "y1": 12, "x2": 50, "y2": 40}},
                        {"category": "lane", "poly2d": []},
                    ],
                }
            )
        labels = root / "bdd100k_labels_release" / "bdd100k" / "labels"
        labels.mkdir(parents=True, exist_ok=True)
        (labels / f"bdd100k_labels_images_{split}.json").write_text(json.dumps(annotations))
    return root
