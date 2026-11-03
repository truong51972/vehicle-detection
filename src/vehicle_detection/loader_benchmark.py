"""Isolated native CPU data-loader measurement, without a model or trainer."""

import importlib.metadata as md
import os
import platform
import time
from pathlib import Path

from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.utils.torch_utils import init_seeds

from .common import digest, file_hash, freeze_json, reuse_evaluation, write_json
from .data import CLASSES
from .datasets import verify_role


def benchmark_loader(dataset, rows, variant, recipes, settings, output, *, allow_measurement=False):
    """One warm-up epoch, three timed epochs; saved measurements are identity-checked."""
    output = Path(output)
    bound = verify_role(dataset, rows, "train", variant, recipes, deep=False)
    identity = {
        "version": "isolated-loader-v1",
        "dataset": bound,
        "settings": settings,
        "warmup_epochs": 1,
        "measured_epochs": 3,
        "code_sha256": file_hash(__file__),
        "hardware": {"platform": platform.platform(), "cpu_count": os.cpu_count()},
        "versions": {p: md.version(p) for p in ("ultralytics", "torch", "numpy", "opencv-python")},
    }
    cached = reuse_evaluation(output / "measurement.json", identity)
    if cached is not None:
        return cached
    if not allow_measurement:
        raise FileNotFoundError(
            "No loader measurement; explicitly enable MEASURE_LOADERS in notebook 04"
        )
    freeze_json(output / "identity.json", identity)
    verify_role(dataset, rows, "train", variant, recipes, deep=True)
    cfg = get_cfg(overrides=settings)
    init_seeds(cfg.seed, deterministic=cfg.deterministic)
    data = build_yolo_dataset(
        cfg,
        str(Path(dataset) / "images/train"),
        cfg.batch,
        {"names": dict(enumerate(CLASSES)), "nc": 6, "channels": 3},
        mode="train",
        stride=32,
    )
    if sorted(Path(p).stem for p in data.im_files) != sorted(r["id"] for r in rows):
        raise ValueError("Native loader changed frozen training membership")
    loader = build_dataloader(
        data, cfg.batch, cfg.workers, shuffle=True, pin_memory=False, device="cpu"
    )
    measurements = []
    for epoch in range(4):
        started = time.perf_counter()
        images = sum(len(batch["img"]) for batch in loader)
        seconds = time.perf_counter() - started
        if images != len(rows):
            raise ValueError("Loader changed epoch image budget")
        measurements.append(
            {
                "epoch": epoch,
                "warmup": epoch == 0,
                "images": images,
                "seconds": seconds,
                "images_per_second": images / seconds,
            }
        )
    measured = measurements[1:]
    result = {
        "epochs": measurements,
        "images_per_second": sum(r["images"] for r in measured)
        / sum(r["seconds"] for r in measured),
        "workers": loader.num_workers,
        "stage": "CPU native loader only; no device transfer, model or optimizer",
        "limitations": (
            "Warm OS cache; startup excluded; sequential branches; no causal speed claim"
        ),
        "artifacts": {},
    }
    write_json(
        output / "measurement.json",
        {"identity": identity, "result": result, "checksum": digest(result)},
    )
    return result
