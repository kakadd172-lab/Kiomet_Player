"""Tower-specific facade for the shared Kiomet icon recognizer."""
from __future__ import annotations

from typing import Iterable

from ..shared.icon_registry import IconRecord
from ..shared.recognizer import IconRecognition, IconRecognizer
from .tower_registry import all_towers


class TowerRecognizer(IconRecognizer):
    """Recognize a cropped tower icon using the checked-in tower catalogue."""

    def __init__(self, records: Iterable[IconRecord] | None = None) -> None:
        super().__init__(all_towers() if records is None else records, "tower")


def recognize_tower(image, **options) -> IconRecognition:
    """Convenience one-shot tower recognition helper."""
    return TowerRecognizer().recognize(image, **options)
