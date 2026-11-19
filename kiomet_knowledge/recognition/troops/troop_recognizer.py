"""Unit-specific facade for the shared Kiomet icon recognizer."""
from __future__ import annotations

from typing import Iterable

from ..shared.icon_registry import IconRecord
from ..shared.recognizer import IconRecognition, IconRecognizer
from .troop_registry import all_troops


class TroopRecognizer(IconRecognizer):
    """Recognize a cropped unit icon using the checked-in unit catalogue."""

    def __init__(self, records: Iterable[IconRecord] | None = None) -> None:
        super().__init__(all_troops() if records is None else records, "unit")


def recognize_troop(image, **options) -> IconRecognition:
    """Convenience one-shot unit recognition helper."""
    return TroopRecognizer().recognize(image, **options)
