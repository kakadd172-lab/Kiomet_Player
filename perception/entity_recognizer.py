"""Recognize caller-proposed entity regions inside a full Kiomet screenshot.

This module deliberately does not claim to locate entities. A detector or
vision model must first propose bounding boxes; this layer validates those
boxes, extracts their pixels, and applies the category-specific recognizers.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from kiomet_knowledge.recognition.shared.colour_detector import ColourObservation
from kiomet_knowledge.recognition.towers.tower_recognizer import TowerRecognizer
from kiomet_knowledge.recognition.troops.troop_recognizer import TroopRecognizer


@dataclass(frozen=True)
class RegionProposal:
    """A proposed entity box in normalized screenshot coordinates."""

    region_id: str
    category: str
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class RankedEntityCandidate:
    icon_id: str
    display_name: str
    score: float


@dataclass(frozen=True)
class EntityObservation:
    """Recognition result tied to both normalized and pixel screen bounds."""

    region_id: str
    category: str
    bbox: tuple[float, float, float, float]
    pixel_bbox: tuple[int, int, int, int]
    accepted: bool
    status: str
    icon_id: str | None
    display_name: str | None
    top_score: float
    margin: float | None
    candidates: tuple[RankedEntityCandidate, ...]
    colour: ColourObservation | None


def _load_screenshot(image: np.ndarray | str | Path) -> np.ndarray:
    if isinstance(image, (str, Path)):
        loaded = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
        if loaded is None:
            raise ValueError(f"Could not load screenshot: {image}")
        image = loaded
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Screenshot must be a non-empty image array or path")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return image[:, :, :3]
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    raise ValueError(f"Unsupported screenshot shape: {image.shape!r}")


def _validated_box(proposal: RegionProposal) -> tuple[float, float, float, float]:
    if not isinstance(proposal.region_id, str) or not proposal.region_id.strip():
        raise ValueError("region_id must be a non-empty string")
    if proposal.category not in {"tower", "unit"}:
        raise ValueError("category must be 'tower' or 'unit'")
    values = (proposal.x, proposal.y, proposal.width, proposal.height)
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not isfinite(value) for value in values):
        raise ValueError("Bounding box values must be finite numbers")
    x, y, width, height = (float(value) for value in values)
    if width <= 0 or height <= 0 or x < 0 or y < 0 or x + width > 1 or y + height > 1:
        raise ValueError("Bounding box must have positive size and fit within normalized screenshot bounds")
    return x, y, width, height


class ScreenEntityRecognizer:
    """Classify proposed screenshot regions without performing object detection."""

    def __init__(self, tower_recognizer=None, troop_recognizer=None) -> None:
        self.tower_recognizer = tower_recognizer or TowerRecognizer()
        self.troop_recognizer = troop_recognizer or TroopRecognizer()

    def recognize_regions(
        self,
        screenshot: np.ndarray | str | Path,
        proposals: Iterable[RegionProposal],
        *,
        method: str = "shape",
        top_k: int = 5,
        min_score: float = 0.65,
        min_margin: float = 0.10,
        detect_colour: bool = True,
    ) -> list[EntityObservation]:
        """Crop and classify each supplied region in input order.

        Coordinates use [0, 1] relative to screenshot width/height. Pixel
        bounds are half-open (left, top, right, bottom). Invalid or empty
        regions raise ValueError rather than being silently shifted.
        """
        frame = _load_screenshot(screenshot)
        height, width = frame.shape[:2]
        observations = []

        for proposal in proposals:
            x, y, box_width, box_height = _validated_box(proposal)
            box_left = int(round(x * width))
            box_top = int(round(y * height))
            box_right = min(width, max(box_left + 1, int(round((x + box_width) * width))))
            box_bottom = min(height, max(box_top + 1, int(round((y + box_height) * height))))
            if box_left >= width or box_top >= height or box_right <= box_left or box_bottom <= box_top:
                raise ValueError(f"Region {proposal.region_id!r} produces an empty crop")

            # VLM boxes often hug the bright glyph too tightly. Include a
            # modest context border so antialiasing, glows and detached icon
            # details are not clipped; the matcher still extracts foreground.
            box_width_px = box_right - box_left
            box_height_px = box_bottom - box_top
            pad_x = max(2, int(round(box_width_px * 0.25)))
            pad_y = max(2, int(round(box_height_px * 0.25)))
            left = max(0, box_left - pad_x)
            top = max(0, box_top - pad_y)
            right = min(width, box_right + pad_x)
            bottom = min(height, box_bottom + pad_y)
            crop = frame[top:bottom, left:right]
            recognizer = self.tower_recognizer if proposal.category == "tower" else self.troop_recognizer
            result = recognizer.recognize(
                crop,
                method=method,
                top_k=top_k,
                min_score=min_score,
                min_margin=min_margin,
                detect_colour=detect_colour,
            )
            candidates = tuple(
                RankedEntityCandidate(
                    icon_id=match.record.id,
                    display_name=match.record.display_name,
                    score=float(match.score),
                )
                for match in result.candidates
            )
            observations.append(
                EntityObservation(
                    region_id=proposal.region_id,
                    category=proposal.category,
                    bbox=(x, y, box_width, box_height),
                    pixel_bbox=(left, top, right, bottom),
                    accepted=result.accepted,
                    status=result.status,
                    icon_id=result.icon_id,
                    display_name=result.display_name,
                    top_score=float(result.top_score),
                    margin=result.margin,
                    candidates=candidates,
                    colour=result.colour,
                )
            )
        return observations
