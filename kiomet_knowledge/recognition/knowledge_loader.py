"""Unified, validated access to Kiomet's tower and unit reference data."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from .shared.icon_registry import IconRecord
from .towers.tower_registry import TOWER_REGISTRY, all_towers
from .troops.troop_registry import TROOP_REGISTRY, all_troops


@dataclass(frozen=True)
class KnowledgeBase:
    """Read-only lookup tables for the project's known game icons."""

    towers: tuple[IconRecord, ...]
    troops: tuple[IconRecord, ...]
    _tower_aliases: Mapping[str, IconRecord]
    _troop_aliases: Mapping[str, IconRecord]

    @property
    def all_icons(self) -> tuple[IconRecord, ...]:
        return self.towers + self.troops

    def get_tower(self, name: str) -> IconRecord | None:
        """Find a tower by ID, internal name, or display name."""
        if not isinstance(name, str):
            return None
        return self._tower_aliases.get(name.strip().lower())

    def get_troop(self, name: str) -> IconRecord | None:
        """Find a unit by ID, internal name, or display name."""
        if not isinstance(name, str):
            return None
        return self._troop_aliases.get(name.strip().lower())

    def get_icon(self, category: str, name: str) -> IconRecord | None:
        """Look up an icon using the canonical category 'tower' or 'unit'."""
        normalized = category.strip().lower() if isinstance(category, str) else ""
        if normalized == "tower":
            return self.get_tower(name)
        if normalized == "unit":
            return self.get_troop(name)
        return None

    def summary(self) -> dict[str, int]:
        """Return simple counts suitable for diagnostics and tests."""
        return {
            "towers": len(self.towers),
            "troops": len(self.troops),
            "total": len(self.all_icons),
        }


def _validate_unique_ids(records: tuple[IconRecord, ...]) -> None:
    seen: dict[str, IconRecord] = {}
    for record in records:
        previous = seen.get(record.id)
        if previous is not None:
            raise ValueError(
                f"Duplicate knowledge ID {record.id!r}: "
                f"{previous.display_name!r} and {record.display_name!r}"
            )
        seen[record.id] = record


def load_knowledge() -> KnowledgeBase:
    """Load the checked-in tower and unit registries.

    The underlying registries validate their metadata schema, declared entry
    counts, categories, and the existence of each SVG and PNG template.
    """
    towers = tuple(all_towers())
    troops = tuple(all_troops())
    _validate_unique_ids(towers + troops)

    return KnowledgeBase(
        towers=towers,
        troops=troops,
        _tower_aliases=MappingProxyType(dict(TOWER_REGISTRY)),
        _troop_aliases=MappingProxyType(dict(TROOP_REGISTRY)),
    )


# Convenient module-level instance for callers that want a shared registry.
KNOWLEDGE = load_knowledge()
