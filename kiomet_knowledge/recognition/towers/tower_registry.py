from pathlib import Path

from ..shared.icon_registry import (
    get_icon,
    load_icon_registry,
    unique_icons,
)


METADATA_PATH = Path(__file__).with_name("tower_metadata.json")

TOWER_REGISTRY = load_icon_registry(
    METADATA_PATH,
    expected_category="tower",
)


def get_tower(name: str):
    return get_icon(TOWER_REGISTRY, name)


def all_towers():
    return unique_icons(TOWER_REGISTRY)
