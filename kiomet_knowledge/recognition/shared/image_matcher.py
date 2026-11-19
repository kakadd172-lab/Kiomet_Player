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


class ImageMatcher:
    """Compare one cropped icon against registered PNG templates.

    Crops should contain one candidate icon with minimal surrounding UI.
    Grayscale comparison reduces sensitivity to team-colour tinting.
    """

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

            self._templates.append(_Template(record, grayscale, visible))

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

    def match(
        self,
        image: np.ndarray | str | Path,
        *,
        category: str | None = None,
        top_k: int = 5,
    ) -> list[MatchResult]:
        """Return top candidates for one icon crop, optionally category-filtered."""
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer")

        normalized_category = category.strip().lower() if isinstance(category, str) else None
        if category is not None and normalized_category not in {"tower", "unit"}:
            raise ValueError("category must be 'tower', 'unit', or None")

        candidate = self._to_grayscale(image)
        results: list[MatchResult] = []

        for template in self._templates:
            if normalized_category and template.record.category != normalized_category:
                continue

            height, width = template.grayscale.shape
            interpolation = (
                cv2.INTER_AREA
                if candidate.shape[0] > height or candidate.shape[1] > width
                else cv2.INTER_CUBIC
            )
            resized = cv2.resize(candidate, (width, height), interpolation=interpolation)
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
) -> list[MatchResult]:
    """One-off matching helper."""
    return ImageMatcher(records).match(image, category=category, top_k=top_k)
