"""Small fixtures validate training guards; their metrics are never study evidence."""

import copy
from types import SimpleNamespace

import pytest
import yaml
from test_adaptive import frozen as frozen
from test_study import bdd as bdd

torch = pytest.importorskip("torch")
pytest.importorskip("ultralytics")

from vehicle_detection import finetuning  # noqa: E402
from vehicle_detection.adaptive import calibrate_policy, export_pair  # noqa: E402
from vehicle_detection.baseline import COCO_TO_VEHICLE  # noqa: E402
from vehicle_detection.common import digest, file_hash, read_json, write_json  # noqa: E402
from vehicle_detection.data import CLASSES  # noqa: E402
from vehicle_detection.finetuning import (  # noqa: E402
    TRAIN_SETTINGS,
    FreezeAudit,
    backbone_boundary,
    check_resume,
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
        yaml={"yaml_file": "yolo26n.yaml", "backbone": [[], []], "end2end": True},
        end2end=True,
        model=[object(), object(), Detect()],
    )
    monkeypatch.setattr(finetuning, "YOLO", lambda path: SimpleNamespace(model=model, names=names))
    return (protocol, subsets, policy, decision, pair, development, checkpoint, tmp_path / "run")


def test_pair_binds_exact_roles_policy_boxes_and_shared_validation(pair_inputs):
    identity = prepare_pair(*pair_inputs)
    assert identity == prepare_pair(*pair_inputs)
    assert identity["training"]["raw"]["ids"] == identity["training"]["adaptive"]["ids"]
    assert identity["freeze"] == 2
    configs = [
        yaml.safe_load((pair_inputs[-1] / f"{b}.yaml").read_text()) for b in ("raw", "adaptive")
    ]
    assert configs[0]["val"] == configs[1]["val"]
    assert all("test" not in c and c["names"] == dict(enumerate(CLASSES)) for c in configs)
    changed = list(pair_inputs)
    changed[3] = {**changed[3], "retained_policy_id": "another-policy"}
    with pytest.raises(ValueError, match="frozen decision"):
        prepare_pair(*changed)
    root = pair_inputs[4] / "adaptive"
    label = next((root / "labels/train").glob("*.txt"))
    label.write_text("0 0.4 0.4 0.2 0.2\n")
    with pytest.raises(ValueError, match="frozen boxes"):
        prepare_pair(*pair_inputs)


def test_wrong_role_rejected_before_pixels(pair_inputs, monkeypatch):
    def no_pixels(*args, **kwargs):
        pytest.fail("Unexpected image verification before role checks")

    monkeypatch.setattr(finetuning, "verify_alignment", no_pixels)
    meta = pair_inputs[4] / "raw/metadata.csv"
    import pandas as pd

    table = pd.read_csv(meta)
    table["id"] = pair_inputs[0]["splits"]["final_test"][0]["id"]
    table.to_csv(meta, index=False)
    with pytest.raises(ValueError, match="frozen role IDs"):
        prepare_pair(*pair_inputs)


def test_label_rounding_does_not_reorder_nearly_equal_centers(pair_inputs):
    import pandas as pd

    root = pair_inputs[4] / "raw"
    meta = pd.read_csv(root / "metadata.csv")
    selected = set(meta.id)
    rows = copy.deepcopy(
        [r for r in pair_inputs[0]["splits"]["training_pool"] if r["id"] in selected]
    )
    for row in rows:
        row["boxes"] = [
            [
                0,
                (x - 0.005) * row["width"],
                (y - 0.005) * row["height"],
                (x + 0.005) * row["width"],
                (y + 0.005) * row["height"],
            ]
            for x, y in [(0.482141690, 0.51), (0.482141691, 0.49)]
        ]
        label = root / "labels/train" / f"{row['id']}.txt"
        label.write_text("0 0.482142 0.51 0.01 0.01\n0 0.482142 0.49 0.01 0.01\n")
        meta.loc[meta.id == row["id"], "label_sha256"] = file_hash(label)
    meta.to_csv(root / "metadata.csv", index=False)
    finetuning.verify_role(root, rows, "train", "raw_reencoded")


def test_changed_yaml_and_symlinks_rejected_before_training(pair_inputs):
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
    with pytest.raises(ValueError, match="symlinks"):
        train_branch(experiment, "raw", pair_inputs[-1])


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.BatchNorm1d(3)),
            torch.nn.Linear(3, 3),
            torch.nn.Linear(3, 1),
        )
        for param in self.model[0].parameters():
            param.requires_grad = False
        self.model[0].eval()


def test_frozen_parameters_buffers_gradients_and_actual_updates():
    model = TinyModel()
    audit = FreezeAudit(model, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss = model.model(torch.ones(2, 3)).square().sum()
    loss.backward()
    audit.check_gradients(model)
    optimizer.step()
    report = audit.check_weights(model, updated=True)
    assert report["optimizer_steps_checked"] == 1
    assert report["trainable_count"] > 0 and report["frozen_count"] > 0
    with torch.no_grad():
        model.model[0][1].running_mean.add_(1)
    with pytest.raises(ValueError, match="weight/buffer changed"):
        audit.check_weights(model)
    model = TinyModel()
    audit = FreezeAudit(model, 1)
    with pytest.raises(ValueError, match="gradients"):
        audit.check_gradients(model)
    with pytest.raises(ValueError, match="weight updates"):
        audit.check_weights(model, updated=True)
    model.model[1].weight.grad = torch.full_like(model.model[1].weight, float("nan"))
    with pytest.raises(ValueError, match="Nonfinite"):
        audit.check_gradients(model)


def test_resume_rejects_missing_optimizer_changed_settings_and_checkpoint(pair_inputs, monkeypatch):
    experiment = prepare_pair(*pair_inputs)
    identity = {"run_id": "fixture-run", "pair": experiment, "data": "/fixture/raw.yaml"}
    checkpoint = pair_inputs[-1] / "raw/weights/last.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fixture")
    receipt = checkpoint.parent.parent / "resume.json"
    write_json(
        receipt,
        {
            "run_id": identity["run_id"],
            "checkpoint": checkpoint.name,
            "sha256": file_hash(checkpoint),
        },
    )
    args = {**experiment["settings"], "freeze": 2, "data": identity["data"]}
    saved = {
        "epoch": 1,
        "optimizer": {},
        "train_args": args,
        "version": experiment["versions"]["ultralytics"],
        "ema": object(),
    }
    monkeypatch.setattr(finetuning, "torch_safe_load", lambda path: (saved, path))
    monkeypatch.setattr(finetuning, "backbone_boundary", lambda model: 2)
    assert check_resume(checkpoint, identity) == checkpoint
    for key, bad in [("optimizer", None), ("epoch", -1), ("version", "different")]:
        original = saved[key]
        saved[key] = bad
        with pytest.raises(ValueError, match="native optimizer state"):
            check_resume(checkpoint, identity)
        saved[key] = original
    saved["train_args"] = {**args, "batch": 1}
    with pytest.raises(ValueError, match="native optimizer state"):
        check_resume(checkpoint, identity)
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="identity/hash"):
        check_resume(checkpoint, identity)


def test_installed_yolo26_boundary_and_disabled_augmentation():
    from ultralytics.nn.tasks import DetectionModel, yaml_model_load

    config = yaml_model_load("yolo26n.yaml")
    config["end2end"] = True  # The pinned release checkpoint includes both detection branches.
    model = DetectionModel(config, nc=6, verbose=False)
    assert backbone_boundary(model) == len(model.yaml["backbone"]) == 11
    model.yaml = copy.deepcopy(model.yaml)
    model.yaml["yaml_file"] = "yolo11n.yaml"
    with pytest.raises(ValueError, match="YOLO26n"):
        backbone_boundary(model)
    assert TRAIN_SETTINGS["augmentations"] == []
    assert TRAIN_SETTINGS["nms"] is False
    assert TRAIN_SETTINGS["patience"] == 0 and TRAIN_SETTINGS["fraction"] == 1
    assert not any(TRAIN_SETTINGS[k] for k in ("mosaic", "hsv_v", "fliplr", "scale"))


def test_zero_lr_warmup_defers_weight_check_and_oom_budget_change_fails(tmp_path, monkeypatch):
    from ultralytics.engine.trainer import BaseTrainer

    from vehicle_detection.finetuning import PairedTrainer

    def step(trainer):
        trainer.optimizer.step()
        trainer.optimizer.zero_grad()

    monkeypatch.setattr(BaseTrainer, "optimizer_step", step)
    trainer = object.__new__(PairedTrainer)
    trainer.model = TinyModel()
    trainer.audit = FreezeAudit(trainer.model, 1)
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0)
    trainer.optimizer_steps = 0
    trainer.batch_size = 2
    trainer.save_dir = tmp_path
    trainer.run_identity = {"pair": {"settings": {"batch": 2}}}
    trainer.model.model(torch.ones(2, 3)).square().sum().backward()
    trainer.optimizer_step()
    assert trainer.audit.report["optimizer_steps_checked"] == 0
    trainer.optimizer.param_groups[0]["lr"] = 0.1
    trainer.model.model(torch.ones(2, 3)).square().sum().backward()
    trainer.optimizer_step()
    assert trainer.audit.report["optimizer_steps_checked"] == 1
    assert trainer.optimizer_steps == 2
    trainer.batch_size = 1
    with pytest.raises(ValueError, match="OOM retry"):
        trainer.optimizer_step()


def test_native_fixture_pair_completion_and_cache(pair_inputs, monkeypatch):
    from ultralytics.nn.tasks import DetectionModel, yaml_model_load

    config = yaml_model_load("yolo26n.yaml")
    model = DetectionModel(config, nc=80, verbose=False)
    checkpoint = pair_inputs[-2]
    torch.save({"model": model, "train_args": {"task": "detect"}}, checkpoint)
    monkeypatch.setattr(finetuning, "PRETRAINED_SHA256", file_hash(checkpoint))
    monkeypatch.setattr(
        finetuning,
        "TRAIN_SETTINGS",
        {
            **TRAIN_SETTINGS,
            "epochs": 1,
            "batch": 2,
            "nbs": 2,
            "imgsz": 64,
            "workers": 0,
            "device": "cpu",
            "plots": False,
        },
    )
    experiment = prepare_pair(*pair_inputs)
    # prepare_pair's mocked architecture has two modules; this fixture checkpoint has eleven.
    experiment["freeze"] = 11
    experiment["experiment_id"] = digest(
        {k: v for k, v in experiment.items() if k != "experiment_id"}
    )
    write_json(pair_inputs[-1] / "pair.json", experiment)
    results = [train_branch(experiment, b, pair_inputs[-1]) for b in ("raw", "adaptive")]
    assert results[0]["budget"] == results[1]["budget"]
    assert results[0]["budget"]["epochs"] == 1
    assert results[0]["budget"]["optimizer_steps"] > 0
    assert train_branch(experiment, "raw", pair_inputs[-1]) == results[0]
    (pair_inputs[-1] / "raw/weights/best.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifacts changed"):
        train_branch(experiment, "raw", pair_inputs[-1])


def test_interrupted_checkpoint_publish_preserves_previous_resume(tmp_path, monkeypatch):
    from vehicle_detection.finetuning import PairedTrainer

    trainer = object.__new__(PairedTrainer)
    trainer.save_dir = tmp_path
    trainer.wdir = tmp_path / "weights"
    trainer.wdir.mkdir()
    trainer.last = trainer.wdir / "last.pt"
    trainer.last.write_bytes(b"first complete native checkpoint")
    trainer.run_identity = {"run_id": "fixture"}
    trainer.epoch = 0
    trainer._checkpoint_receipt(trainer)
    receipt = read_json(tmp_path / "resume.json")
    previous = trainer.wdir / receipt["checkpoint"]
    assert file_hash(previous) == receipt["sha256"]
    trainer.epoch = 1
    trainer.last.write_bytes(b"second complete native checkpoint")

    def interrupted(*args, **kwargs):
        raise OSError("simulated interruption before publishing resume receipt")

    monkeypatch.setattr(finetuning, "write_json", interrupted)
    with pytest.raises(OSError, match="simulated interruption"):
        trainer._checkpoint_receipt(trainer)
    assert read_json(tmp_path / "resume.json") == receipt
    assert file_hash(previous) == receipt["sha256"]
    monkeypatch.setattr(finetuning, "write_json", write_json)
    trainer._checkpoint_receipt(trainer)
    assert not previous.exists()
    receipt = read_json(tmp_path / "resume.json")
    assert file_hash(trainer.wdir / receipt["checkpoint"]) == file_hash(trainer.last)
