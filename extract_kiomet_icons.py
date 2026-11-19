
import base64
import json
import re
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT = Path(__file__).resolve().parent

TOWER_DIR = (
    PROJECT / "kiomet_knowledge" / "assets"
    / "towers" / "official_svg"
)

UNIT_DIR = (
    PROJECT / "kiomet_knowledge" / "assets"
    / "troops" / "official_svg"
)

MANIFEST = (
    PROJECT / "kiomet_knowledge" / "recognition"
    / "official_icon_manifest.json"
)

GAME_URL = "https://kiomet.com/"

SOURCE_URL = "https://github.com/SoftbearStudios/kiomet"
SOURCE_LICENSE = "AGPL-3.0-or-later"


# ============================================================
# HELPERS
# ============================================================

def slugify(name):
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def extract_gallery(page, category, output_dir):

    print("\n" + "=" * 55)
    print(f"EXTRACTING {category.upper()}")
    print("=" * 55)

    input(
        f"\nOpen the {category.upper()} gallery in Chromium. "
        "\nWhen it is visible, press ENTER here..."
    )

    # Wait for gallery SVG elements to appear.
    try:
        page.locator("svg image").first.wait_for(
            state="attached",
            timeout=15000
        )
    except PlaywrightTimeoutError:
        print(
            "\nERROR: No gallery SVG images were detected."
            "\nMake sure you opened the correct gallery."
        )
        return []

    icons = page.locator("svg image").evaluate_all("""
        elements => elements.map(element => {
            const title = element.querySelector("title");

            return {
                name: title ? title.textContent.trim() : null,
                href:
                    element.getAttribute("href") ||
                    element.getAttributeNS(
                        "http://www.w3.org/1999/xlink",
                        "href"
                    )
            };
        })
    """)

    output_dir.mkdir(parents=True, exist_ok=True)

    extracted = []
    seen = set()

    for icon in icons:

        name = icon.get("name")
        href = icon.get("href")

        if not name or not href:
            continue

        if not href.startswith("data:image/svg+xml;base64,"):
            continue

        key = name.lower()

        if key in seen:
            continue

        try:
            encoded = href.split(",", 1)[1]
            svg_bytes = base64.b64decode(encoded)

            # Basic SVG validation.
            if b"<svg" not in svg_bytes:
                print(f"Skipped invalid SVG: {name}")
                continue

            filename = slugify(name) + ".svg"
            output_path = output_dir / filename

            output_path.write_bytes(svg_bytes)

            extracted.append({
                "name": name,
                "category": category,
                "filename": filename,
                "relative_path": str(
                    output_path.relative_to(PROJECT)
                ),
                "source": SOURCE_URL,
                "license": SOURCE_LICENSE
            })

            seen.add(key)

            print(f"[SAVED] {name}")

        except Exception as error:
            print(f"[ERROR] {name}: {error}")

    print(f"\nUnique icons extracted: {len(extracted)}")

    return extracted


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 55)
    print("KIOMET OFFICIAL ICON EXTRACTOR")
    print("=" * 55)

    TOWER_DIR.mkdir(parents=True, exist_ok=True)
    UNIT_DIR.mkdir(parents=True, exist_ok=True)

    all_icons = []

    browser = None

    try:
        with sync_playwright() as playwright:

            print("\nStarting Chromium...")

            browser = playwright.chromium.launch(
                headless=False,
                args=[
                    "--start-maximized",
                    "--disable-infobars"
                ]
            )

            # Use the actual browser window size.
            context = browser.new_context(
                no_viewport=True
            )

            page = context.new_page()

            print("Opening Kiomet...")

            try:
                page.goto(
                    GAME_URL,
                    wait_until="domcontentloaded",
                    timeout=120000
                )
            except PlaywrightTimeoutError:
                print(
                    "\nWARNING: Page loading timed out."
                    "\nChecking whether Kiomet is accessible..."
                )

            page.wait_for_timeout(10000)

            print("\nChromium is ready.")
            print("Check that the game fits in the window.")

            input(
                "\nOnce the game has loaded, press ENTER "
                "to begin extraction..."
            )

            # ---------------- TOWERS ----------------

            towers = extract_gallery(
                page,
                "towers",
                TOWER_DIR
            )

            all_icons.extend(towers)

            # ---------------- UNITS ----------------

            units = extract_gallery(
                page,
                "units",
                UNIT_DIR
            )

            all_icons.extend(units)

            # ---------------- MANIFEST ----------------

            manifest = {
                "source": SOURCE_URL,
                "license": SOURCE_LICENSE,
                "description": (
                    "Official Kiomet tower and unit SVG icons "
                    "extracted from the game's rendered galleries."
                ),
                "icons": all_icons
            }

            MANIFEST.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            MANIFEST.write_text(
                json.dumps(manifest, indent=4),
                encoding="utf-8"
            )

            print("\n" + "=" * 55)
            print("EXTRACTION COMPLETE")
            print("=" * 55)

            print(f"Towers: {len(towers)}")
            print(f"Units: {len(units)}")
            print(f"Total: {len(all_icons)}")

            print(f"\nManifest:\n{MANIFEST}")

            input(
                "\nPress ENTER to close Chromium..."
            )

    except Exception as error:

        print("\nFATAL ERROR:")
        print(repr(error))

        input("\nPress ENTER to exit...")

        sys.exit(1)

    finally:

        if browser:
            try:
                browser.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
