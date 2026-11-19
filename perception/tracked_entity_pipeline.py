"""Connect verified screenshot detections to stable temporal entity tracks.

This adapter is intentionally thin: the vision detector proposes and verifies
regions, while EntityTracker owns cross-frame identity and disappearance logic.
Unaccepted icon matches remain unidentified observations and are still eligible
for spatial tracking; they are never promoted to a guessed game identity.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .entity_tracker import EntityObservation, EntityTracker, TrackedEntity
from .vision_entity_detector import VisionEntityDetector, VerifiedEntity


class TrackedEntityPipeline:
    """Run perception and update persistent tracks for each screenshot frame."""

    def __init__(
        self,
        detector: VisionEntityDetector | None = None,
        tracker: EntityTracker | None = None,
    ) -> None:
        self.detector = detector or VisionEntityDetector()
        self.tracker = tracker or EntityTracker()
        self.last_detections: tuple[VerifiedEntity, ...] = ()
        self.last_snapshot: tuple[TrackedEntity, ...] = ()

    @staticmethod
    def _to_observation(detection: VerifiedEntity) -> EntityObservation:
        recognition = detection.recognition
        identity = None
        confidence = None
        if recognition.accepted:
            # Prefer the recognizer's stable catalogue ID over its display label.
            identity = recognition.icon_id or recognition.display_name
            confidence = recognition.top_score
        return EntityObservation(
            category=detection.proposal.category,
            bbox=(
                detection.proposal.x,
                detection.proposal.y,
                detection.proposal.width,
                detection.proposal.height,
            ),
            identity=identity,
            confidence=confidence,
        )

    def process(
        self,
        screenshot: np.ndarray | str | Path,
        **detect_options: Any,
    ) -> tuple[TrackedEntity, ...]:
        """Detect one frame and return the track snapshot after that frame.

        A detector exception leaves tracker state untouched. Unidentified
        detections still participate in spatial matching but retain identity=None.
        """
        detections = tuple(self.detector.detect(screenshot, **detect_options))
        observations = [self._to_observation(item) for item in detections]
        snapshot = self.tracker.update(observations)
        self.last_detections = detections
        self.last_snapshot = snapshot
        return snapshot

    def snapshot(self, *, include_missing: bool = True) -> tuple[TrackedEntity, ...]:
        return self.tracker.snapshot(include_missing=include_missing)

    def history(self, track_id: str):
        """Return retained frame history for one stable entity ID."""
        return self.tracker.history(track_id)

    def reset(self) -> None:
        self.tracker.reset()
        self.last_detections = ()
        self.last_snapshot = ()
