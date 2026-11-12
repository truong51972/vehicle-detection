"""Deterministic BGR uint8 operators; all preserve pixel coordinates."""

import cv2
import numpy as np


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


def local_shadow(image):
    """Bounded smooth illumination gain applied to original detail, from pixels only."""
    validate_image(image)
    scale = min(1, 512 / max(image.shape[:2]))
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    illumination = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (0, 0), 16)
    gain = np.clip(np.sqrt(0.25 / np.maximum(illumination.astype(np.float32) / 255, 0.0625)), 1, 2)
    gain = cv2.resize(gain, image.shape[1::-1], interpolation=cv2.INTER_LINEAR)
    return np.clip(image.astype(np.float32) * gain[:, :, None], 0, 255).astype(np.uint8)


def local_contrast(image, clip_limit, grid_size, gain, max_offset):
    """Bounded CLAHE luminance offset, added equally to BGR and faded near black/white."""
    validate_image(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.createCLAHE(clip_limit, (grid_size, grid_size)).apply(gray)
    luminance = gray.astype(np.float32)
    protection = np.clip(np.minimum(luminance - 8, 247 - luminance) / 32, 0, 1)
    offset = np.rint(np.clip((equalized - luminance) * gain, -max_offset, max_offset) * protection)
    return np.clip(image.astype(np.int16) + offset.astype(np.int16)[:, :, None], 0, 255).astype(
        np.uint8
    )


def routes_local_contrast(scores, settings):
    return (
        scores["contrast"] < settings["contrast_below"]
        and settings["brightness_above"] < scores["brightness"] < settings["brightness_below"]
        and scores["noise"] <= settings["noise_max"]
        and scores["black_fraction"] < settings["black_below"]
        and scores["white_fraction"] < settings["white_below"]
    )


def canny_edges(image, low_threshold, high_threshold, blur_sigma):
    """Full-resolution Canny support; thresholds use Sobel magnitude, not luminance."""
    validate_image(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(gray, (0, 0), blur_sigma)
    return cv2.Canny(smooth, low_threshold, high_threshold, apertureSize=3, L2gradient=True)


def canny_detail(
    image,
    low_threshold,
    high_threshold,
    blur_sigma,
    detail_sigma,
    amount,
    max_offset,
    mask_radius,
    *,
    edges=None,
):
    """Canny-guided, bounded luminance detail; preserve color, geometry and source bytes."""
    validate_image(image)
    if edges is None:
        edges = canny_edges(image, low_threshold, high_threshold, blur_sigma)
    # Finite support keeps pixels outside the edge neighbourhood exactly unchanged.
    kernel = np.ones((2 * mask_radius + 1, 2 * mask_radius + 1), np.uint8)
    mask = cv2.dilate(edges, kernel).astype(np.float32) / 255
    mask = cv2.GaussianBlur(mask, (3, 3), 0.8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    detail = gray - cv2.GaussianBlur(gray, (0, 0), detail_sigma)
    protection = np.clip(np.minimum(gray - 8, 247 - gray) / 32, 0, 1)
    offset = np.rint(np.clip(detail * amount, -max_offset, max_offset) * mask * protection)
    return np.clip(image.astype(np.int16) + offset.astype(np.int16)[:, :, None], 0, 255).astype(
        np.uint8
    )


def routes_canny_detail(scores, settings):
    return (
        settings["brightness_above"] < scores["brightness"] < settings["brightness_below"]
        and scores["noise"] <= settings["noise_max"]
        and settings["edge_density_above"] < scores["edge_density"] < settings["edge_density_below"]
    )


def validate_recipes(recipes):
    bounds = {
        "raw": {},
        "local_shadow": {},
        "gamma": {"gamma": (0.1, 3)},
        "clahe": {"clip_limit": (0.1, 8), "grid_size": (2, 32)},
        "denoise": {"strength": (0.1, 30)},
        "sharpen": {"amount": (0, 2), "sigma": (0.1, 5)},
        "adaptive": {
            "brightness_below": (0, 1),
            "contrast_below": (0, 1),
            "noise_above": (0, 1),
            "sharpness_min": (0, 1e6),
            "sharpness_below": (0, 1e6),
            "max_filters": (0, 2),
        },
        "local_contrast": {
            "clip_limit": (0.1, 8),
            "grid_size": (2, 32),
            "gain": (0, 1),
            "max_offset": (0, 64),
        },
        "adaptive_local_contrast": {
            "contrast_below": (0, 1),
            "brightness_above": (0, 1),
            "brightness_below": (0, 1),
            "noise_max": (0, 1),
            "black_below": (0, 1),
            "white_below": (0, 1),
        },
        "canny_detail": {
            "low_threshold": (0, 1443),
            "high_threshold": (0, 1443),
            "blur_sigma": (0.1, 5),
            "detail_sigma": (0.1, 5),
            "amount": (0, 2),
            "max_offset": (0, 64),
            "mask_radius": (0, 5),
        },
        "adaptive_canny_detail": {
            "brightness_above": (0, 1),
            "brightness_below": (0, 1),
            "noise_max": (0, 1),
            "edge_density_above": (0, 1),
            "edge_density_below": (0, 1),
        },
    }
    for variant, settings in recipes.items():
        if variant not in bounds or set(settings) != set(bounds[variant]):
            raise ValueError(f"Unknown recipe or parameters: {variant}")
        for key, value in settings.items():
            low, high = bounds[variant][key]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or not low <= value <= high
            ):
                raise ValueError(f"Invalid {variant}.{key}")
            if key in {"grid_size", "max_filters", "max_offset", "mask_radius"} and not isinstance(
                value, int
            ):
                raise ValueError(f"{key} must be an integer")
    if "adaptive" in recipes:
        if not {"gamma", "clahe", "denoise", "sharpen"} <= recipes.keys():
            raise ValueError("Adaptive requires the four fixed operator settings")
        if recipes["adaptive"]["sharpness_min"] > recipes["adaptive"]["sharpness_below"]:
            raise ValueError("Adaptive sharpness bounds are reversed")
    if "adaptive_local_contrast" in recipes and "local_contrast" not in recipes:
        raise ValueError("Adaptive local contrast requires the local_contrast operator settings")
    if "canny_detail" in recipes:
        settings = recipes["canny_detail"]
        if settings["low_threshold"] >= settings["high_threshold"]:
            raise ValueError("Canny thresholds must be strictly increasing")
    if "adaptive_canny_detail" in recipes:
        if "canny_detail" not in recipes:
            raise ValueError("Adaptive Canny detail requires the canny_detail operator settings")
        settings = recipes["adaptive_canny_detail"]
        for feature in ("brightness", "edge_density"):
            if settings[f"{feature}_above"] >= settings[f"{feature}_below"]:
                raise ValueError(f"Canny {feature} bounds must be strictly increasing")


def apply_recipe(image, variant, recipes):
    """Offline filtering only; same pixel coordinates, BGR uint8, no source mutation."""
    validate_image(image)
    settings = recipes[variant]
    scores = quality(image)
    if variant == "adaptive_canny_detail":
        operator = recipes["canny_detail"]
        edges = canny_edges(
            image, operator["low_threshold"], operator["high_threshold"], operator["blur_sigma"]
        )
        scores["edge_density"] = float(np.mean(edges != 0))
        if not routes_canny_detail(scores, settings):
            return image, [], scores
        output = canny_detail(image, **operator, edges=edges)
        return output, [{"name": "canny_detail", "params": operator}], scores
    if variant == "adaptive":
        wanted = {
            "gamma": scores["brightness"] < settings["brightness_below"],
            "clahe": scores["contrast"] < settings["contrast_below"],
            "denoise": scores["noise"] > settings["noise_above"],
            "sharpen": (
                settings["sharpness_min"] <= scores["sharpness"] < settings["sharpness_below"]
                and scores["noise"] <= settings["noise_above"]
            ),
        }
        chosen = [name for name, enabled in wanted.items() if enabled][: settings["max_filters"]]
        output = image
        for name in chosen:
            output, _, _ = apply_recipe(output, name, recipes)
        return output, [{"name": n, "params": recipes[n]} for n in chosen], scores
    if variant == "adaptive_local_contrast":
        if not routes_local_contrast(scores, settings):
            return image, [], scores
        output, applied, _ = apply_recipe(image, "local_contrast", recipes)
        return output, applied, scores
    if variant == "raw":
        return image, [], scores
    if variant == "gamma":
        output = gamma(image, **settings)
    elif variant == "clahe":
        output = clahe(image, **settings)
    elif variant == "denoise":
        output = denoise(image, **settings)
    elif variant == "sharpen":
        output = sharpen(image, **settings)
    elif variant == "local_shadow":
        output = local_shadow(image)
    elif variant == "local_contrast":
        output = local_contrast(image, **settings)
    elif variant == "canny_detail":
        output = canny_detail(image, **settings)
    else:
        raise ValueError(f"Unknown recipe: {variant}")
    return output, [{"name": variant, "params": settings}], scores
