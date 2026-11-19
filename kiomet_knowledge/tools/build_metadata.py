import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
RECOGNITION = PROJECT / "kiomet_knowledge" / "recognition"
MANIFEST = RECOGNITION / "official_icon_manifest.json"

INTERNAL_NAME_OVERRIDES = {
    ("towers", "EWS"): "Ews",
    ("units", "EMP"): "Emp",
    ("units", "King"): "Ruler",
}

EXPECTED_COUNTS = {
    "towers": 27,
    "units": 10,
}


def main():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    entries = {
        "towers": [],
        "units": [],
    }

    for icon in manifest["icons"]:
        source_category = icon["category"]

        if source_category not in entries:
            raise ValueError(f"Unexpected category: {source_category}")

        display_name = icon["name"]
        internal_name = INTERNAL_NAME_OVERRIDES.get(
            (source_category, display_name),
            display_name,
        )

        asset_category = (
            "towers" if source_category == "towers" else "troops"
        )

        stem = Path(icon["filename"]).stem

        svg_relative = (
            Path("kiomet_knowledge")
            / "assets"
            / asset_category
            / "official_svg"
            / icon["filename"]
        )

        template_relative = (
            Path("kiomet_knowledge")
            / "assets"
            / "derived"
            / "official_icon_renders"
            / asset_category
            / f"{stem}.png"
        )

        svg_absolute = PROJECT / svg_relative
        template_absolute = PROJECT / template_relative

        if not svg_absolute.is_file():
            raise FileNotFoundError(svg_absolute)

        if not template_absolute.is_file():
            raise FileNotFoundError(template_absolute)

        entry = {
            "id": f"{'tower' if source_category == 'towers' else 'unit'}.{stem}",
            "category": "tower" if source_category == "towers" else "unit",
            "display_name": display_name,
            "internal_name": internal_name,
            "asset_category": asset_category,
            "filename": icon["filename"],
            "svg_path": svg_relative.as_posix(),
            "template_path": template_relative.as_posix(),
            "source": icon["source"],
            "license": icon["license"],
        }

        entries[source_category].append(entry)

    for category, expected in EXPECTED_COUNTS.items():
        actual = len(entries[category])
        if actual != expected:
            raise ValueError(
                f"{category}: expected {expected} entries, found {actual}"
            )

    for category, filename in (
        ("towers", "tower_metadata.json"),
        ("units", "troop_metadata.json"),
    ):
        document = {
            "schema_version": 1,
            "category": "tower" if category == "towers" else "unit",
            "source": manifest["source"],
            "license": manifest["license"],
            "count": len(entries[category]),
            "entries": entries[category],
        }

        output = RECOGNITION / category
        if category == "units":
            output = RECOGNITION / "troops"

        output.mkdir(parents=True, exist_ok=True)

        destination = output / filename
        destination.write_text(
            json.dumps(document, indent=4, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        print(f"Created: {destination}")
        print(f"Entries: {len(entries[category])}")

    print("\nMetadata generation successful.")


if __name__ == "__main__":
    main()
