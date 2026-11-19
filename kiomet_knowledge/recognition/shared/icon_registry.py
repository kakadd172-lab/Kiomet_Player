import json
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class IconRecord:
    id: str
    category: str
    display_name: str
    internal_name: str
    asset_category: str
    filename: str
    svg_path: Path
    template_path: Path
    source: str
    license: str


def load_icon_registry(
    metadata_path: Path,
    expected_category: str,
) -> dict[str, IconRecord]:

    metadata_path = Path(metadata_path)

    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)

    document = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    if document.get("schema_version") != 1:
        raise ValueError("Unsupported metadata schema version")

    if document.get("category") != expected_category:
        raise ValueError(
            f"Expected category {expected_category!r}, "
            f"received {document.get('category')!r}"
        )

    entries = document.get("entries")

    if not isinstance(entries, list):
        raise ValueError("Metadata entries must be a list")

    if len(entries) != document.get("count"):
        raise ValueError("Metadata count does not match entry count")

    registry = {}

    for item in entries:
        record = IconRecord(
            id=item["id"],
            category=item["category"],
            display_name=item["display_name"],
            internal_name=item["internal_name"],
            asset_category=item["asset_category"],
            filename=item["filename"],
            svg_path=PROJECT_ROOT / item["svg_path"],
            template_path=PROJECT_ROOT / item["template_path"],
            source=item["source"],
            license=item["license"],
        )

        if record.category != expected_category:
            raise ValueError(
                f"Invalid category for {record.display_name}"
            )

        if not record.svg_path.is_file():
            raise FileNotFoundError(record.svg_path)

        if not record.template_path.is_file():
            raise FileNotFoundError(record.template_path)

        aliases = {
            record.id,
            record.internal_name,
            record.display_name,
            record.internal_name.lower(),
            record.display_name.lower(),
        }

        for alias in aliases:
            key = alias.strip().lower()

            existing = registry.get(key)

            if existing is not None and existing.id != record.id:
                raise ValueError(f"Registry alias collision: {key}")

            registry[key] = record

    return registry


def get_icon(
    registry: dict[str, IconRecord],
    name: str,
) -> IconRecord | None:

    return registry.get(name.strip().lower())


def unique_icons(
    registry: dict[str, IconRecord],
) -> list[IconRecord]:

    records = {
        record.id: record
        for record in registry.values()
    }

    return sorted(
        records.values(),
        key=lambda record: record.display_name.lower(),
    )
