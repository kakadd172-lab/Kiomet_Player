"""Deterministic template matching for cropped Kiomet icon images."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .icon_registry import IconRecord


@dataclass(frozen=True)
class MatchResult:
    """A ranked icon candidate; score is similarity, not probability."""
    record: IconRecord
    score: float


@dataclass(frozen=True)
class _Template:
    record: IconRecord
    grayscale: np.ndarray
    mask: np.ndarray
    shape: np.ndarray


class ImageMatcher:
    """Compare cropped game icons against registered PNG templates.

    The grayscale method is the original pixel-comparison baseline. The
    experimental shape method thresholds the visible foreground, preserves
    its aspect ratio, and compares normalized binary shapes. Neither method
    produces calibrated probabilities.
    """

    SHAPE_SIZE = 96

    def __init__(self, records: Iterable[IconRecord]) -> None:
        self._templates: list[_Template] = []

        for record in records:
            image = cv2.imread(str(record.template_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise FileNotFoundError(
                    f"Could not load icon template: {record.template_path}"
                )

            if image.ndim == 2:
                grayscale = image
                mask = np.full(image.shape, 255, dtype=np.uint8)
            elif image.ndim == 3 and image.shape[2] == 4:
                grayscale = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
                mask = image[:, :, 3]
            elif image.ndim == 3 and image.shape[2] == 3:
                grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                mask = np.full(image.shape[:2], 255, dtype=np.uint8)
            else:
                raise ValueError(f"Unsupported template shape: {image.shape!r}")

            visible = (mask > 16).astype(np.uint8)
            if not np.any(visible):
                raise ValueError(f"Template has no visible pixels: {record.template_path}")

            shape = self._normalize_shape(visible * 255)
            self._templates.append(_Template(record, grayscale, visible, shape))

        if not self._templates:
            raise ValueError("ImageMatcher requires at least one template")

    @staticmethod
    def _to_grayscale(image: np.ndarray | str | Path) -> np.ndarray:
        if isinstance(image, (str, Path)):
            loaded = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
            if loaded is None:
                raise ValueError(f"Could not load candidate image: {image}")
            image = loaded

        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("Candidate must be a non-empty image array or path")

        if image.ndim == 2:
            grayscale = image
        elif image.ndim == 3 and image.shape[2] == 4:
            grayscale = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
        elif image.ndim == 3 and image.shape[2] == 3:
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(f"Unsupported candidate image shape: {image.shape!r}")

        if grayscale.dtype != np.uint8:
            grayscale = cv2.normalize(
                grayscale, None, 0, 255, cv2.NORM_MINMAX
            ).astype(np.uint8)
        return grayscale

    @classmethod
    def _normalize_shape(
        cls,
        mask: np.ndarray,
        *,
        size: int | None = None,
    ) -> np.ndarray:
        """Crop a binary foreground and fit it in a square without distortion."""
        output_size = size or cls.SHAPE_SIZE
        binary = np.where(mask > 0, 255, 0).astype(np.uint8)
        points = cv2.findNonZero(binary)
        if points is None:
            return np.zeros((output_size, output_size), dtype=np.uint8)

        x, y, width, height = cv2.boundingRect(points)
        cropped = binary[y:y + height, x:x + width]
        padding = max(2, int(round(max(width, height) * 0.12)))
        side = max(width, height) + padding * 2
        canvas = np.zeros((side, side), dtype=np.uint8)
        left = (side - width) // 2
        top = (side - height) // 2
        canvas[top:top + height, left:left + width] = cropped
        return cv2.resize(
            canvas,
            (output_size, output_size),
            interpolation=cv2.INTER_NEAREST,
        )

    @classmethod
    def _candidate_shape(
        cls,
        image: np.ndarray | str | Path,
        *,
        filter_saturation: bool = True,
        contrast_fraction: float = 0.45,
    ) -> np.ndarray:
        """Extract bright icon strokes with configurable relative brightness."""
        if not 0.0 <= contrast_fraction <= 1.0:
            raise ValueError("contrast_fraction must be between 0 and 1")
        grayscale = cls._to_grayscale(image)

        if isinstance(image, (str, Path)):
            bgr = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
            if bgr is None:
                raise ValueError(f"Could not load candidate image: {image}")
            if bgr.ndim == 2:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
            elif bgr.ndim == 3 and bgr.shape[2] == 4:
                bgr = bgr[:, :, :3]
            elif bgr.ndim != 3 or bgr.shape[2] != 3:
                raise ValueError(f"Unsupported candidate image shape: {bgr.shape!r}")
        elif image.ndim == 2:
            bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] in (3, 4):
            bgr = image[:, :, :3]
        else:
            raise ValueError(f"Unsupported candidate image shape: {image.shape!r}")

        if bgr.dtype != np.uint8:
            bgr = cv2.normalize(bgr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Kiomet places pale icon strokes over saturated team-colour markers.
        # A grayscale-only threshold can mistake the blue glow for foreground.
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        low_saturation = hsv[:, :, 1] <= 110

        minimum = int(np.min(grayscale))
        maximum = int(np.max(grayscale))
        if maximum <= minimum:
            return np.zeros((cls.SHAPE_SIZE, cls.SHAPE_SIZE), dtype=np.uint8)

        otsu_threshold, _ = cv2.threshold(
            grayscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        contrast_threshold = minimum + (maximum - minimum) * contrast_fraction
        threshold = max(float(otsu_threshold), float(contrast_threshold))
        foreground_mask = grayscale > threshold
        if filter_saturation:
            foreground_mask &= low_saturation
        foreground = foreground_mask.astype(np.uint8) * 255

        # If colour filtering leaves too little signal, fall back to the
        # grayscale foreground rather than silently returning an empty shape.
        if cv2.countNonZero(foreground) < 3:
            _, foreground = cv2.threshold(
                grayscale, threshold, 255, cv2.THRESH_BINARY
            )

        # Discard isolated compression/noise pixels without removing small
        # disconnected parts of a real icon.
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            foreground, connectivity=8
        )
        if component_count > 1:
            largest_area = max(
                int(stats[index, cv2.CC_STAT_AREA])
                for index in range(1, component_count)
            )
            minimum_area = max(2, int(round(largest_area * 0.03)))
            cleaned = np.zeros_like(foreground)
            for index in range(1, component_count):
                area = int(stats[index, cv2.CC_STAT_AREA])
                if area >= minimum_area:
                    cleaned[labels == index] = 255
            if cv2.countNonZero(cleaned) > 0:
                foreground = cleaned

        return cls._normalize_shape(foreground)

    @classmethod
    def _dominant_component_shape(cls, shape: np.ndarray) -> np.ndarray:
        """Keep the largest connected foreground component and normalize it.

        Some game crops include detached troop-count glyphs below the tower
        symbol. Those marks should not distort the tower's silhouette.
        """
        binary = (shape > 0).astype(np.uint8)
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        if component_count <= 1:
            return cls._normalize_shape(binary * 255)

        largest = max(
            range(1, component_count),
            key=lambda index: int(stats[index, cv2.CC_STAT_AREA]),
        )
        dominant = np.where(labels == largest, 255, 0).astype(np.uint8)
        # ``shape`` is already normalized to SHAPE_SIZE by _candidate_shape.
        # Re-normalizing it here adds a second padding/rescale pass and changes
        # the silhouette even when the smaller components were already removed.
        return dominant

    @staticmethod
    def _shape_similarity(candidate: np.ndarray, template: np.ndarray) -> float:
        """Combine binary overlap with symmetric chamfer distance."""
        candidate_mask = candidate > 0
        template_mask = template > 0
        candidate_count = int(np.count_nonzero(candidate_mask))
        template_count = int(np.count_nonzero(template_mask))
        if candidate_count == 0 or template_count == 0:
            return 0.0

        intersection = int(np.count_nonzero(candidate_mask & template_mask))
        union = candidate_count + template_count - intersection
        overlap = intersection / union if union else 0.0

        distance_to_template = cv2.distanceTransform(
            (~template_mask).astype(np.uint8), cv2.DIST_L2, 3
        )
        distance_to_candidate = cv2.distanceTransform(
            (~candidate_mask).astype(np.uint8), cv2.DIST_L2, 3
        )
        candidate_to_template = float(
            np.mean(distance_to_template[candidate_mask])
        )
        template_to_candidate = float(
            np.mean(distance_to_candidate[template_mask])
        )
        mean_distance = (candidate_to_template + template_to_candidate) / 2.0
        distance_score = max(0.0, 1.0 - mean_distance / 16.0)
        return max(0.0, min(1.0, 0.4 * overlap + 0.6 * distance_score))

    def match(
        self,
        image: np.ndarray | str | Path,
        *,
        category: str | None = None,
        top_k: int = 5,
        method: str = "grayscale",
    ) -> list[MatchResult]:
        """Return ranked candidates using grayscale or experimental shape matching."""
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if method not in {"grayscale", "shape"}:
            raise ValueError("method must be 'grayscale' or 'shape'")

        normalized_category = category.strip().lower() if isinstance(category, str) else None
        if category is not None and normalized_category not in {"tower", "unit"}:
            raise ValueError("category must be 'tower', 'unit', or None")

        candidate = self._to_grayscale(image) if method == "grayscale" else None
        candidate_shapes = (
            (
                self._candidate_shape(image, filter_saturation=True),
                self._candidate_shape(image, filter_saturation=False),
                # A high-brightness pass tests whether pale glyph strokes can
                # be separated from the darker saturated territory marker.
                self._candidate_shape(
                    image, filter_saturation=False, contrast_fraction=0.72
                ),
                # Detached count/stack markers below a tower can otherwise
                # stretch the normalized silhouette away from the icon.
                self._dominant_component_shape(
                    self._candidate_shape(image, filter_saturation=True)
                ),
            )
            if method == "shape" else ()
        )
        results: list[MatchResult] = []

        for template in self._templates:
            if normalized_category and template.record.category != normalized_category:
                continue

            if method == "shape":
                # Compare three foreground extraction passes plus a
                # dominant-component silhouette that excludes detached
                # count/stack markers beneath the main tower icon.
                score = max(
                    self._shape_similarity(candidate_shape, template.shape)
                    for candidate_shape in candidate_shapes
                )
            else:
                height, width = template.grayscale.shape
                interpolation = (
                    cv2.INTER_AREA
                    if candidate.shape[0] > height or candidate.shape[1] > width
                    else cv2.INTER_CUBIC
                )
                resized = cv2.resize(
                    candidate, (width, height), interpolation=interpolation
                )
                difference = cv2.absdiff(resized, template.grayscale)
                error = float(np.mean(difference[template.mask.astype(bool)]))
                score = max(0.0, min(1.0, 1.0 - error / 255.0))

            results.append(MatchResult(template.record, score))

        results.sort(key=lambda result: (-result.score, result.record.id))
        return results[:top_k]


def match_icons(
    image: np.ndarray | str | Path,
    records: Iterable[IconRecord],
    *,
    category: str | None = None,
    top_k: int = 5,
    method: str = "grayscale",
) -> list[MatchResult]:
    """One-off matching helper."""
    return ImageMatcher(records).match(
        image, category=category, top_k=top_k, method=method
    )
