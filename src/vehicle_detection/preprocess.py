"""Deterministic BGR uint8 operators; all preserve pixel coordinates."""

import cv2
import numpy as np

from .common import digest
from .config import number


def validate_image(image):
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or min(image.shape[:2]) < 2
    ):
        raise ValueError("Expected a nonempty HxWx3 BGR uint8 image (at least 2x2)")


def quality(image):
    validate_image(image)
    # Threshold units are fixed to a maximum 512-pixel side, independent of source resolution.
    scale = min(1, 512 / max(image.shape[:2]))
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    residual = gray.astype(np.float32) - cv2.medianBlur(gray, 3).astype(np.float32)
    return {
        "brightness": float(gray.mean() / 255),
        "contrast": float(gray.std() / 255),
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "noise": float(np.median(np.abs(residual)) / (0.6745 * 255)),
        "black_fraction": float(np.mean(gray <= 3)),
        "white_fraction": float(np.mean(gray >= 252)),
    }


def gamma(image, gamma):
    lut = np.clip((np.arange(256) / 255) ** gamma * 255, 0, 255).astype(np.uint8)
    return cv2.LUT(image, lut)


def clahe(image, clip_limit, grid_size):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clip_limit, (grid_size, grid_size)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def denoise(image, strength):
    return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)


def sharpen(image, amount, sigma):
    smooth = cv2.GaussianBlur(image, (0, 0), sigma)
    return cv2.addWeighted(image, 1 + amount, smooth, -amount, 0)


REGISTRY = {
    "gamma": (gamma, {"gamma": (0.1, 3.0, False)}),
    "clahe": (clahe, {"clip_limit": (0.1, 8.0, False), "grid_size": (2, 32, True)}),
    "denoise": (denoise, {"strength": (0.1, 30.0, False)}),
    "sharpen": (sharpen, {"amount": (0.0, 2.0, False), "sigma": (0.1, 5.0, False)}),
}


def validate_operators(operators):
    if not isinstance(operators, list):
        raise ValueError("operators must be an ordered list")
    seen = set()
    for op in operators:
        if not isinstance(op, dict) or set(op) != {"name", "enabled", "params"}:
            raise ValueError("Each operator requires name, enabled, params")
        name = op["name"]
        if not isinstance(name, str) or name not in REGISTRY or name in seen:
            raise ValueError(f"Unknown/duplicate operator: {name}")
        seen.add(name)
        if not isinstance(op["enabled"], bool):
            raise ValueError(f"{name}.enabled must be boolean")
        specs = REGISTRY[name][1]
        if not isinstance(op["params"], dict) or set(op["params"]) != set(specs):
            raise ValueError(f"{name} expects parameters: {list(specs)}")
        for key, bounds in specs.items():
            number(op["params"][key], f"{name}.{key}", *bounds)


def route(scores, operators, policy):
    wanted = {
        "gamma": scores["brightness"] < policy["brightness_below"],
        "clahe": scores["contrast"] < policy["contrast_below"],
        "denoise": scores["noise"] > policy["noise_above"],
        "sharpen": (
            policy["sharpness_min"] <= scores["sharpness"] < policy["sharpness_below"]
            and scores["noise"] <= policy["noise_above"]
        ),
    }
    return [op for op in operators if op["enabled"] and wanted[op["name"]]][: policy["max_filters"]]


def process(image, treatment, config):
    validate_image(image)
    scores = quality(image) if treatment == "adaptive" else None
    if treatment == "adaptive":
        chosen = route(scores, config["operators"], config["policy"])
    elif treatment == "raw":
        chosen = []
    else:
        chosen = [op for op in config["operators"] if op["enabled"] and op["name"] == treatment]
        if not chosen:
            raise ValueError(f"Unknown or disabled treatment: {treatment}")
    output = image
    for op in chosen:
        output = REGISTRY[op["name"]][0](output, **op["params"])
    return output, scores, [{"name": op["name"], "params": op["params"]} for op in chosen]


def stress(image, condition, parameters, seed, sample_id):
    validate_image(image)
    if condition == "clean":
        return image.copy()
    if condition == "dark":
        result = image.astype(np.float32) * parameters["dark_factor"]
    elif condition == "blur":
        result = cv2.GaussianBlur(image, (0, 0), parameters["blur_sigma"])
    elif condition == "noise":
        paired_seed = int(digest([seed, sample_id, condition])[:16], 16)
        rng = np.random.default_rng(paired_seed)
        result = image.astype(np.float32) + rng.normal(0, parameters["noise_sigma"], image.shape)
    elif condition == "haze":
        alpha = parameters["haze_alpha"]
        result = image.astype(np.float32) * (1 - alpha) + 255 * alpha
    else:
        raise ValueError(f"Unknown synthetic condition: {condition}")
    return np.clip(result, 0, 255).astype(np.uint8)
