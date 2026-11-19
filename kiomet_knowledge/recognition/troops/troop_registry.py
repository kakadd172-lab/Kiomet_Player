from pathlib import Path

from ..shared.icon_registry import (
    get_icon,
    load_icon_registry,
    unique_icons,
)


METADATA_PATH = Path(__file__).with_name("troop_metadata.json")

TROOP_REGISTRY = load_icon_registry(
    METADATA_PATH,
    expected_category="unit",
)


def get_troop(name: str):
    return get_icon(TROOP_REGISTRY, name)


def all_troops():
    return unique_icons(TROOP_REGISTRY)
