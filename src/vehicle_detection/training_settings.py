"""Native settings shared by training and read-only lifecycle validation."""

from .baseline import VAL_SETTINGS

# Keep the historical S10 budget and disabled augmentations; both branches share it.
TRAIN_SETTINGS = {
    **VAL_SETTINGS,
    "epochs": 20,
    "patience": 0,
    "batch": 16,
    "amp": False,
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.01,
    "cos_lr": True,
    "warmup_epochs": 3.0,
    "warmup_bias_lr": 0.0,
    "weight_decay": 0.0005,
    "nbs": 64,
    "cls_remap": True,
    "rect": False,
    "cache": False,
    "fraction": 1.0,
    "compile": False,
    "channels_last": False,
    "save": True,
    "val": True,
    "augmentations": [],
    **dict.fromkeys(
        (
            "hsv_h",
            "hsv_s",
            "hsv_v",
            "degrees",
            "translate",
            "scale",
            "shear",
            "perspective",
            "flipud",
            "fliplr",
            "bgr",
            "mosaic",
            "mixup",
            "cutmix",
            "copy_paste",
            "multi_scale",
            "close_mosaic",
        ),
        0,
    ),
}


def validate_training_args(args, data, settings):
    """Check native args without constructing or invoking a trainer."""
    if (
        args.get("data") != str(data)
        or args.get("freeze") != 11
        or any(args.get(k) != value for k, value in settings.items())
    ):
        raise ValueError("Native training settings differ from the experiment")
