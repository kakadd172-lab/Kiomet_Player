"""Dominant visible-colour extraction for cropped Kiomet observations."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class ColourObservation:
    """Dominant chromatic colour measured from an image crop."""
    hue_degrees: float
    saturation: float
    value: float
    hex_rgb: str
    pixel_ratio: float
    sample_count: int


def _load_bgr(image: np.ndarray | str | Path) -> np.ndarray:
    if isinstance(image, (str, Path)):
        loaded = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
        if loaded is None:
            raise ValueError(f"Could not load candidate image: {image}")
        image = loaded

    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Candidate must be a non-empty image array or path")

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim != 3:
        raise ValueError(f"Unsupported candidate image shape: {image.shape!r}")

    if image.shape[2] == 4:
        bgr = image[:, :, :3].astype(np.float32)
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        return np.clip(bgr * alpha, 0, 255).astype(np.uint8)
    if image.shape[2] == 3:
        if image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
            image = image.astype(np.uint8)
        return image

    raise ValueError(f"Unsupported candidate image shape: {image.shape!r}")


def detect_dominant_colour(
    image: np.ndarray | str | Path,
    *,
    min_saturation: int = 55,
    min_value: int = 35,
    hue_tolerance_degrees: float = 12.0,
) -> ColourObservation | None:
    """Measure the dominant saturated colour in an image crop.

    Hue is in degrees [0, 360), saturation/value are in [0, 1], and
    pixel_ratio is the fraction of eligible chromatic pixels within the
    dominant hue window. Returns None when no eligible colour is present.

    This measures colour; it does not classify player or team ownership.
    """
    for name, value in (("min_saturation", min_saturation), ("min_value", min_value)):
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 255:
            raise ValueError(f"{name} must be an integer between 0 and 255")

    if (
        not isinstance(hue_tolerance_degrees, (int, float))
        or isinstance(hue_tolerance_degrees, bool)
        or not 0 < hue_tolerance_degrees <= 90
    ):
        raise ValueError("hue_tolerance_degrees must be greater than 0 and at most 90")

    bgr = _load_bgr(image)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.float32) * 2.0
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    eligible = (saturation >= min_saturation) & (value >= min_value)

    if not np.any(eligible):
        return None

    histogram = np.bincount(hsv[:, :, 0][eligible].astype(np.int32), minlength=180)
    dominant_bin = int(np.argmax(histogram))
    dominant_hue = dominant_bin * 2.0

    hue_distance = np.abs(hue - dominant_hue)
    hue_distance = np.minimum(hue_distance, 360.0 - hue_distance)
    selected = eligible & (hue_distance <= hue_tolerance_degrees)
    sample_count = int(np.count_nonzero(selected))
    eligible_count = int(np.count_nonzero(eligible))

    if sample_count == 0:
        return None

    angles = np.deg2rad(hue[selected])
    mean_hue = float(
        (np.rad2deg(np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))) + 360.0)
        % 360.0
    )
    mean_saturation = float(np.mean(saturation[selected]) / 255.0)
    mean_value = float(np.mean(value[selected]) / 255.0)
    mean_bgr = np.mean(bgr[selected].astype(np.float32), axis=0)
    blue, green, red = [int(round(channel)) for channel in mean_bgr]

    return ColourObservation(
        hue_degrees=round(mean_hue, 2),
        saturation=round(mean_saturation, 4),
        value=round(mean_value, 4),
        hex_rgb=f"#{red:02X}{green:02X}{blue:02X}",
        pixel_ratio=round(sample_count / eligible_count, 4),
        sample_count=sample_count,
    )
