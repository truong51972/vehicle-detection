"""Tiny synthetic BDD sources; they exercise the workflow, never scientific measurements."""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml

from vehicle_detection.common import digest, file_hash
from vehicle_detection.data import group_key, scan_bdd
from vehicle_detection.experiment_data import config_path

REPO = Path(__file__).resolve().parents[1]
ROLES = ["training_pool"] * 3 + ["calibration", "development_validation", "development_validation"]


@pytest.fixture
def bdd(tmp_path):
    root = tmp_path / "bdd"
    for split, n in [("train", 6), ("val", 3)]:
        directory = root / "images/100k" / split
        directory.mkdir(parents=True)
        frames = []
        for i in range(n):
            name = f"{split}_{i}.jpg"
            image = np.random.default_rng(i + (100 if split == "val" else 0)).integers(
                0, 256, (64, 80, 3), dtype=np.uint8
            )
            assert cv2.imwrite(str(directory / name), image)
            labels = [
                {
                    "id": j,
                    "category": c,
                    "box2d": {"x1": 1 + j, "y1": 2 + j, "x2": 30 + j, "y2": 40 + j},
                }
                for j, c in enumerate(["car", "truck", "bus", "train", "motor", "bike"])
            ]
            frames.append({"name": name, "attributes": {"weather": "clear"}, "labels": labels})
        (root / f"bdd100k_labels_images_{split}.json").write_text(json.dumps(frames))
    return root


@pytest.fixture
def project(bdd, tmp_path):
    """Isolated project root: frozen-table copy of the fixture, real version YAMLs, CPU args."""
    root = tmp_path / "project"
    (root / "configs").mkdir(parents=True)
    (root / "pyproject.toml").write_text('[project]\nname = "vehicle-detection"\n')
    annotations = bdd / "bdd100k_labels_images_train.json"
    frames = json.loads(annotations.read_text())
    # A railway-only frame becomes a retained five-class negative.
    frames[1]["labels"] = [label for label in frames[1]["labels"] if label["category"] == "train"]
    annotations.write_text(json.dumps(frames))
    for i in (0, 4):
        image = np.random.default_rng(i).integers(0, 30, (64, 80, 3), dtype=np.uint8)
        assert cv2.imwrite(str(bdd / f"images/100k/train/train_{i}.jpg"), image)
    records, _, _ = scan_bdd(bdd)
    rows = []
    for r in records:
        index = int(r["id"].split("_")[-1])
        role = ROLES[index] if r["official_split"] == "train" else "final_test"
        rows.append(
            {
                "id": r["id"],
                "group": group_key(r),
                "source_group": group_key(r),
                "role": role,
                "s10": int(r["id"] in {"train_0", "train_1"}),
                "s20": int(role == "training_pool"),
                "source_path": str(Path(r["path"]).relative_to(bdd)),
                "width": 80,
                "height": 64,
                "sha256": file_hash(r["path"]) if role != "final_test" else "",
                "labels_sha256": digest(r["labels"]),
            }
        )
    pd.DataFrame(rows).to_csv(root / "configs/splits.csv.gz", index=False)
    shared = yaml.safe_load((REPO / "configs/data.yaml").read_text())
    shared.update(data_root=str(bdd), splits_sha256=file_hash(root / "configs/splits.csv.gz"))
    (root / "configs/data.yaml").write_text(yaml.safe_dump(shared))
    for version in ("v0", "v1", "v2", "v3"):
        config = yaml.safe_load(config_path(REPO, version).read_text())
        config["training"].update(epochs=1, batch=2, nbs=2, imgsz=64, workers=0, device="cpu")
        config["training"].update(amp=False, mosaic=0.0, close_mosaic=0, plots=False)
        config["evaluation"].update(imgsz=64, batch=2, device="cpu", plots=False)
        config["preparation"]["workers"] = 2
        config_path(root, version).parent.mkdir(parents=True)
        config_path(root, version).write_text(yaml.safe_dump(config, sort_keys=False))
    return root
