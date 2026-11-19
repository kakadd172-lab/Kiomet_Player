"""Conservative confidence assessment for ranked icon matches.

Scores are similarity measurements, not probabilities. This module evaluates
the best score and its separation from the runner-up; it never claims that a
score is a calibrated probability of correctness.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .image_matcher import MatchResult


@dataclass(frozen=True)
class ConfidenceAssessment:
    """Decision-support summary for a ranked set of icon matches."""

    best_match: MatchResult | None
    top_score: float
    margin: float | None
    status: str
    accepted: bool


def assess_matches(
    matches: Sequence[MatchResult],
    *,
    min_score: float = 0.65,
    min_margin: float = 0.10,
) -> ConfidenceAssessment:
    """Assess whether the best template match is sufficiently distinct.

    A strong result requires a high top similarity and a clear lead over the
    runner-up. A single candidate is ambiguous because it has no competitor.

    Thresholds are heuristics and must be calibrated against labelled Kiomet
    screenshots before being used for consequential decisions.
    """
    for name, value in (("min_score", min_score), ("min_margin", min_margin)):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{name} must be a number between 0 and 1")
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")

    if not matches:
        return ConfidenceAssessment(None, 0.0, None, "no_match", False)

    ranked = sorted(matches, key=lambda item: item.score, reverse=True)
    best = ranked[0]
    if not 0.0 <= best.score <= 1.0:
        raise ValueError("Match scores must be between 0 and 1")

    if len(ranked) == 1:
        return ConfidenceAssessment(best, best.score, None, "ambiguous", False)

    second_score = ranked[1].score
    if not 0.0 <= second_score <= 1.0:
        raise ValueError("Match scores must be between 0 and 1")

    margin = best.score - second_score
    if best.score < min_score:
        status = "low_score"
    elif margin < min_margin:
        status = "ambiguous"
    else:
        status = "strong"

    return ConfidenceAssessment(
        best, best.score, margin, status, status == "strong"
    )
