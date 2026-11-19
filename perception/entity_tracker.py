"""Deterministic temporal tracking for recognized Kiomet entities.

The tracker consumes normalized observations from perception and assigns stable
IDs across frames. It deliberately does not perform visual recognition itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class EntityObservation:
    category: str
    bbox: BBox
    identity: str | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class TrackedEntity:
    track_id: str
    category: str
    bbox: BBox
    identity: str | None
    confidence: float | None
    first_seen_frame: int
    last_seen_frame: int
    missed_frames: int


@dataclass(frozen=True)
class EntityHistoryEntry:
    """One frame's observed or temporarily missing state for a track."""

    frame: int
    bbox: BBox
    identity: str | None
    confidence: float | None
    visible: bool
    missed_frames: int


def _validate_bbox(bbox: BBox) -> None:
    if len(bbox) != 4:
        raise ValueError("bbox must contain x, y, width and height")
    x, y, width, height = bbox
    if not all(isinstance(value, (int, float)) for value in bbox):
        raise ValueError("bbox values must be numeric")
    if not all(0.0 <= value <= 1.0 for value in bbox):
        raise ValueError("bbox values must be normalized between 0 and 1")
    if width <= 0 or height <= 0 or x + width > 1.0 or y + height > 1.0:
        raise ValueError("bbox must have positive size and fit inside the frame")


def _iou(first: BBox, second: BBox) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def _center_distance(first: BBox, second: BBox) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    dx = (ax + aw / 2) - (bx + bw / 2)
    dy = (ay + ah / 2) - (by + bh / 2)
    return (dx * dx + dy * dy) ** 0.5


class EntityTracker:
    """Track entities between observations using category and spatial overlap."""

    def __init__(
        self,
        *,
        max_missed_frames: int = 2,
        min_iou: float = 0.05,
        max_center_distance: float = 0.08,
        max_history_per_track: int = 120,
    ) -> None:
        if not isinstance(max_missed_frames, int) or isinstance(max_missed_frames, bool) or max_missed_frames < 0:
            raise ValueError("max_missed_frames must be a non-negative integer")
        if not 0 <= min_iou <= 1:
            raise ValueError("min_iou must be between 0 and 1")
        if max_center_distance < 0:
            raise ValueError("max_center_distance must be non-negative")
        if not isinstance(max_history_per_track, int) or isinstance(max_history_per_track, bool) or max_history_per_track < 1:
            raise ValueError("max_history_per_track must be a positive integer")
        self.max_missed_frames = max_missed_frames
        self.min_iou = float(min_iou)
        self.max_center_distance = float(max_center_distance)
        self.max_history_per_track = max_history_per_track
        self._history: dict[str, list[EntityHistoryEntry]] = {}
        self._frame = 0
        self._next_id = 1
        self._tracks: dict[str, TrackedEntity] = {}

    @property
    def frame(self) -> int:
        return self._frame

    def reset(self) -> None:
        self._frame = 0
        self._next_id = 1
        self._tracks.clear()
        self._history.clear()

    def history(self, track_id: str) -> tuple[EntityHistoryEntry, ...]:
        """Return retained frame history, including the final expiry frame."""
        return tuple(self._history.get(track_id, ()))

    def snapshot(self, *, include_missing: bool = True) -> tuple[TrackedEntity, ...]:
        tracks = self._tracks.values()
        if not include_missing:
            tracks = (track for track in tracks if track.missed_frames == 0)
        return tuple(sorted(tracks, key=lambda track: track.track_id))

    def update(self, observations: Iterable[EntityObservation]) -> tuple[TrackedEntity, ...]:
        current = list(observations)
        for observation in current:
            if observation.category not in {"tower", "unit"}:
                raise ValueError("category must be 'tower' or 'unit'")
            _validate_bbox(observation.bbox)
            if observation.confidence is not None and not 0 <= observation.confidence <= 1:
                raise ValueError("confidence must be between 0 and 1")

        self._frame += 1
        previous = list(self._tracks.values())
        candidates: dict[int, list[tuple[float, float, str]]] = {}
        for index, observation in enumerate(current):
            scored = []
            for track in previous:
                if track.category != observation.category or track.missed_frames > self.max_missed_frames:
                    continue
                if track.identity and observation.identity and track.identity != observation.identity:
                    continue
                overlap = _iou(track.bbox, observation.bbox)
                distance = _center_distance(track.bbox, observation.bbox)
                if overlap >= self.min_iou or distance <= self.max_center_distance:
                    scored.append((overlap, -distance, track.track_id))
            scored.sort(reverse=True)
            candidates[index] = scored

        # Augmenting paths maximize the number of matched observations instead
        # of allowing an early greedy match to strand another valid observation.
        assigned_track_to_observation: dict[str, int] = {}

        def assign(observation_index: int, visited: set[str]) -> bool:
            for _, _, track_id in candidates[observation_index]:
                if track_id in visited:
                    continue
                visited.add(track_id)
                previous_index = assigned_track_to_observation.get(track_id)
                if previous_index is None or assign(previous_index, visited):
                    assigned_track_to_observation[track_id] = observation_index
                    return True
            return False

        for index in range(len(current)):
            assign(index, set())

        observation_to_track = {
            observation_index: track_id
            for track_id, observation_index in assigned_track_to_observation.items()
        }
        updated: dict[str, TrackedEntity] = {}
        for track in previous:
            updated[track.track_id] = TrackedEntity(
                track_id=track.track_id,
                category=track.category,
                bbox=track.bbox,
                identity=track.identity,
                confidence=track.confidence,
                first_seen_frame=track.first_seen_frame,
                last_seen_frame=track.last_seen_frame,
                missed_frames=track.missed_frames + 1,
            )

        for index, observation in enumerate(current):
            track_id = observation_to_track.get(index)
            old = self._tracks.get(track_id) if track_id else None
            if old is None:
                track_id = f"entity-{self._next_id:06d}"
                self._next_id += 1
                first_seen = self._frame
            else:
                first_seen = old.first_seen_frame
            updated[track_id] = TrackedEntity(
                track_id=track_id,
                category=observation.category,
                bbox=observation.bbox,
                identity=observation.identity if observation.identity is not None else (old.identity if old else None),
                confidence=observation.confidence,
                first_seen_frame=first_seen,
                last_seen_frame=self._frame,
                missed_frames=0,
            )

        # Record every track's state before pruning expired tracks. Missing
        # frames retain the last known box but are explicitly marked invisible.
        for track_id, track in updated.items():
            entries = self._history.setdefault(track_id, [])
            entries.append(EntityHistoryEntry(
                frame=self._frame,
                bbox=track.bbox,
                identity=track.identity,
                confidence=track.confidence,
                visible=track.missed_frames == 0,
                missed_frames=track.missed_frames,
            ))
            if len(entries) > self.max_history_per_track:
                del entries[:-self.max_history_per_track]

        self._tracks = {
            track_id: track
            for track_id, track in updated.items()
            if track.missed_frames <= self.max_missed_frames
        }
        return self.snapshot()
