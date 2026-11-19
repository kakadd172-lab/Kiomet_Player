"""Category-aware icon recognition built on the shared matcher and confidence rules."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .colour_detector import ColourObservation, detect_dominant_colour
from .confidence import ConfidenceAssessment, assess_matches
from .icon_registry import IconRecord
from .image_matcher import ImageMatcher, MatchResult


@dataclass(frozen=True)
class IconRecognition:
    """One crop's ranked identity candidates and conservative acceptance result."""

    category: str
    candidates: tuple[MatchResult, ...]
    assessment: ConfidenceAssessment
    colour: ColourObservation | None

    @property
    def best_match(self) -> MatchResult | None:
        return self.assessment.best_match

    @property
    def accepted(self) -> bool:
        return self.assessment.accepted

    @property
    def status(self) -> str:
        return self.assessment.status

    @property
    def top_score(self) -> float:
        return self.assessment.top_score

    @property
    def margin(self) -> float | None:
        return self.assessment.margin

    @property
    def icon_id(self) -> str | None:
        if not self.accepted or self.best_match is None:
            return None
        return self.best_match.record.id

    @property
    def display_name(self) -> str | None:
        if not self.accepted or self.best_match is None:
            return None
        return self.best_match.record.display_name


class IconRecognizer:
    """Recognize cropped tower or unit icons without forcing uncertain labels."""

    def __init__(self, records: Iterable[IconRecord], category: str) -> None:
        normalized = category.strip().lower() if isinstance(category, str) else ""
        if normalized not in {"tower", "unit"}:
            raise ValueError("category must be 'tower' or 'unit'")

        selected = tuple(records)
        if not selected:
            raise ValueError("IconRecognizer requires at least one icon record")
        if any(record.category != normalized for record in selected):
            raise ValueError(f"All records must belong to category {normalized!r}")

        self.category = normalized
        self.matcher = ImageMatcher(selected)

    def recognize(
        self,
        image: np.ndarray | str | Path,
        *,
        method: str = "shape",
        top_k: int = 5,
        min_score: float = 0.65,
        min_margin: float = 0.10,
        detect_colour: bool = True,
    ) -> IconRecognition:
        """Recognize an already-cropped icon; this does not locate icons in a full screen.

        Similarity scores are not probabilities. An identity is exposed only
        when both the score and runner-up margin pass the configured heuristics.
        """
        matches = tuple(
            self.matcher.match(
                image,
                category=self.category,
                top_k=top_k,
                method=method,
            )
        )
        assessment = assess_matches(
            matches,
            min_score=min_score,
            min_margin=min_margin,
        )
        colour = detect_dominant_colour(image) if detect_colour else None
        return IconRecognition(
            category=self.category,
            candidates=matches,
            assessment=assessment,
            colour=colour,
        )
