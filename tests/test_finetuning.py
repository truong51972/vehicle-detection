"""Small fixtures validate training guards; their metrics are never study evidence."""

from types import SimpleNamespace

import pandas as pd
import pytest
import yaml
from test_adaptive import frozen as frozen
from test_study import bdd as bdd

torch = pytest.importorskip("torch")
pytest.importorskip("ultralytics")

from vehicle_detection import finetuning  # noqa: E402
from vehicle_detection.adaptive import calibrate_policy, export_pair  # noqa: E402
from vehicle_detection.baseline import COCO_TO_VEHICLE  # noqa: E402
from vehicle_detection.common import digest, file_hash, write_json  # noqa: E402
from vehicle_detection.data import CLASSES  # noqa: E402
from vehicle_detection.finetuning import (  # noqa: E402
    TRAIN_SETTINGS,
    backbone_boundary,
    prepare_pair,
    train_branch,
)
from vehicle_detection.offline import export_variant  # noqa: E402
from vehicle_detection.protocol import development_records  # noqa: E402
from vehicle_detection.subsets import build_subsets, training_records  # noqa: E402


@pytest.fixture
def pair_inputs(frozen, bdd, tmp_path, monkeypatch):
    protocol, records = frozen
    subsets = build_subsets(protocol, attempts=2)
    policy = calibrate_policy(protocol, bdd, sample_size=2, per_operator=1)
    decision = {"status": "v3_not_deployed", "retained_policy_id": policy["policy_id"]}
    decision["decision_id"] = digest(decision)
    partitions = training_records(protocol, subsets, "S10", records, bdd)
    dev = development_records(protocol, records, bdd)
    conversion = tmp_path / "conversion"
    write_json(conversion / "conversion.json", {"fixture": True})
    by_id = {r["id"]: r for rows in protocol["splits"].values() for r in rows}
    for split, rows in {**partitions, **dev}.items():
        for record in rows:
            row = by_id[record["id"]]
            path = conversion / "labels" / split / f"{row['id']}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                "".join(
                    f"{c} {(x1 + x2) / 2 / row['width']:.6f} {(y1 + y2) / 2 / row['height']:.6f} "
                    f"{(x2 - x1) / row['width']:.6f} {(y2 - y1) / row['height']:.6f}\n"
                    for c, x1, y1, x2, y2 in row["boxes"]
                )
            )
    pair = tmp_path / "S10"
    development = tmp_path / "dev"
    export_pair(partitions, conversion, pair, policy, protocol, subsets, "S10")
    export_variant(dev, conversion, development, "raw_reencoded", policy["recipes"])
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"fixture checkpoint; never detector evidence")
    monkeypatch.setattr(finetuning, "PRETRAINED_SHA256", file_hash(checkpoint))
    names = dict.fromkeys(range(80), "other")
    names.update({c: CLASSES[b] for c, b in COCO_TO_VEHICLE.items()})

    class Detect:
        pass

    model = SimpleNamespace(
        yaml={"yaml_file": "yolo26n.yaml", "backbone": [[]] * 11, "end2end": True},
        end2end=True,
        model=[*[object() for _ in range(11)], Detect()],
    )
    monkeypatch.setattr(finetuning, "YOLO", lambda path: SimpleNamespace(model=model, names=names))
    return (protocol, subsets, policy, decision, pair, development, checkpoint, tmp_path / "run")


def test_pair_binds_roles_and_native_settings_without_training(pair_inputs):
    experiment = prepare_pair(*pair_inputs)
    assert experiment == prepare_pair(*pair_inputs)
    assert experiment["version"] == "paired-native-v1"
    assert experiment["training"]["raw"]["ids"] == experiment["training"]["adaptive"]["ids"]
    assert experiment["freeze"] == 11
    configs = [
        yaml.safe_load((pair_inputs[-1] / f"{b}.yaml").read_text())
        for b in ("raw", "adaptive")
    ]
    assert configs[0]["val"] == configs[1]["val"]
    assert all("test" not in c and c["names"] == dict(enumerate(CLASSES)) for c in configs)
    changed = list(pair_inputs)
    changed[3] = {**changed[3], "retained_policy_id": "different"}
    with pytest.raises(ValueError, match="frozen decision"):
        prepare_pair(*changed)


def test_role_rejected_before_image_pixels(pair_inputs, monkeypatch):
    meta_path = pair_inputs[4] / "raw/metadata.csv"
    meta = pd.read_csv(meta_path)
    meta["id"] = pair_inputs[0]["splits"]["final_test"][0]["id"]
    meta.to_csv(meta_path, index=False)

    def no_decode(*args, **kwargs):
        pytest.fail("Final-test ID must not trigger image reads")

    monkeypatch.setattr(finetuning, "verify_alignment", no_decode)
    with pytest.raises(ValueError, match="frozen role IDs"):
        prepare_pair(*pair_inputs)


def test_deep_label_audit_is_export_only(pair_inputs):
    from vehicle_detection.datasets import verify_role

    protocol, subsets, policy, decision, pair, dev, checkpoint, output = pair_inputs
    experiment = prepare_pair(*pair_inputs)
    records = [
        r for r in protocol["splits"]["training_pool"]
        if r["id"] in set(subsets["subsets"]["S10"]["ids"])
    ]
    label = next((pair / "raw/labels/train").glob("*.txt"))
    label.write_text("0 0.4 0.4 0.2 0.2\n")
    # Cached identity checks avoid per-label reads; explicit deep audit catches in-place edits.
    assert prepare_pair(*pair_inputs) == experiment
    with pytest.raises(ValueError, match="frozen boxes|changed"):
        verify_role(pair / "raw", records, "train", "raw_reencoded", deep=True)


def test_native_training_uses_same_pretrained_and_reuses_outputs(pair_inputs, monkeypatch):
    experiment = prepare_pair(*pair_inputs)
    calls = []

    class Native:
        def __init__(self, checkpoint):
            self.checkpoint = str(checkpoint)

        def train(self, **kwargs):
            calls.append((self.checkpoint, kwargs))
            if kwargs.get("resume"):
                return None
            run = pair_inputs[-1] / kwargs["name"]
            (run / "weights").mkdir(parents=True, exist_ok=True)
            (run / "weights/best.pt").write_bytes(b"fixture best")
            (run / "weights/last.pt").write_bytes(b"fixture last")
            with (run / "args.yaml").open("w") as stream:
                yaml.safe_dump({
                    "data": kwargs["data"], "freeze": kwargs["freeze"],
                    "epochs": kwargs["epochs"], "batch": kwargs["batch"],
                    "imgsz": kwargs["imgsz"],
                }, stream)
            pd.DataFrame({
                "epoch": range(experiment["settings"]["epochs"]),
                "metrics/mAP50-95(B)": [0.1] * experiment["settings"]["epochs"],
            }).to_csv(run / "results.csv", index=False)

    monkeypatch.setattr(finetuning, "YOLO", Native)
    raw = train_branch(experiment, "raw", pair_inputs[-1])
    adaptive = train_branch(experiment, "adaptive", pair_inputs[-1])
    assert raw["budget"] == adaptive["budget"]
    assert [p for p, _ in calls] == [str(pair_inputs[-2])] * 2
    assert all(settings["freeze"] == 11 for _, settings in calls)
    assert train_branch(experiment, "raw", pair_inputs[-1]) == raw
    assert len(calls) == 2
    train_branch(experiment, "raw", pair_inputs[-1], resume=True)
    assert calls[-1][0].endswith("/raw/weights/last.pt")
    assert calls[-1][1] == {"resume": True}


def test_changed_dataset_yaml_native_args_and_symlinks_rejected(pair_inputs, monkeypatch):
    experiment = prepare_pair(*pair_inputs)
    config = pair_inputs[-1] / "raw.yaml"
    original = config.read_text()
    data = yaml.safe_load(original)
    data["val"] = "images/test"
    config.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="YAML changed"):
        train_branch(experiment, "raw", pair_inputs[-1])
    config.write_text(original)
    label = next((pair_inputs[4] / "raw/labels/train").glob("*.txt"))
    moved = label.with_suffix(".saved")
    label.rename(moved)
    label.symlink_to(moved)
    with pytest.raises(ValueError, match="symlink"):
        train_branch(experiment, "raw", pair_inputs[-1])


def test_installed_yolo26_boundary_and_disabled_augmentation():
    from ultralytics.nn.tasks import DetectionModel, yaml_model_load

    config = yaml_model_load("yolo26n.yaml")
    config["end2end"] = True
    model = DetectionModel(config, nc=6, verbose=False)
    assert backbone_boundary(model) == len(model.yaml["backbone"]) == 11
    model.yaml["yaml_file"] = "yolo11n.yaml"
    with pytest.raises(ValueError, match="YOLO26n"):
        backbone_boundary(model)
    assert TRAIN_SETTINGS["augmentations"] == []
    assert TRAIN_SETTINGS["nms"] is False
    assert TRAIN_SETTINGS["patience"] == 0 and TRAIN_SETTINGS["fraction"] == 1
    assert not any(TRAIN_SETTINGS[k] for k in ("mosaic", "hsv_v", "fliplr", "scale"))


def test_real_native_one_epoch_smoke(pair_inputs, monkeypatch):
    """Tiny CPU integration, not a scientific or GPU benchmark."""
    from ultralytics import YOLO as NativeYOLO
    from ultralytics.nn.tasks import DetectionModel, yaml_model_load

    config = yaml_model_load("yolo26n.yaml")
    config["end2end"] = True
    model = DetectionModel(config, nc=80, verbose=False)
    model.names = dict.fromkeys(range(80), "other")
    model.names.update({c: CLASSES[b] for c, b in COCO_TO_VEHICLE.items()})
    checkpoint = pair_inputs[-2]
    torch.save({"model": model, "train_args": {"task": "detect"}}, checkpoint)
    monkeypatch.setattr(finetuning, "PRETRAINED_SHA256", file_hash(checkpoint))
    monkeypatch.setattr(finetuning, "YOLO", NativeYOLO)
    monkeypatch.setattr(finetuning, "TRAIN_SETTINGS", {
        **TRAIN_SETTINGS, "epochs": 1, "batch": 2, "nbs": 2,
        "imgsz": 64, "workers": 0, "device": "cpu", "plots": False,
    })
    experiment = prepare_pair(*pair_inputs)
    runs = [train_branch(experiment, b, pair_inputs[-1]) for b in ("raw", "adaptive")]
    assert runs[0]["budget"] == runs[1]["budget"]
    assert runs[0]["budget"]["epochs"] == 1
    assert all((pair_inputs[-1] / b / "weights/best.pt").is_file() for b in ("raw", "adaptive"))
