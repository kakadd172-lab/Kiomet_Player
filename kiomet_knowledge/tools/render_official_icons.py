from pathlib import Path
from base64 import b64encode

from PIL import Image, ImageDraw
from playwright.sync_api import sync_playwright


PROJECT = Path(__file__).resolve().parents[2]

ASSET_ROOT = PROJECT / "kiomet_knowledge" / "assets"

OUTPUT = (
    ASSET_ROOT
    / "derived"
    / "official_icon_renders"
)

ICON_SIZE = 128
COLUMNS = 4
CELL_WIDTH = 160
CELL_HEIGHT = 170


def render_svg(page, svg_path, output_path):
    svg_data = svg_path.read_bytes()
    encoded = b64encode(svg_data).decode("ascii")

    html = f"""
    <html>
      <body style="
        margin: 0;
        width: {ICON_SIZE}px;
        height: {ICON_SIZE}px;
        background: transparent;
      ">
        <img
          id="icon"
          src="data:image/svg+xml;base64,{encoded}"
          style="width: 100%; height: 100%;"
        />
      </body>
    </html>
    """

    page.set_content(html)
    page.locator("#icon").wait_for(state="visible")
    page.locator("#icon").screenshot(path=str(output_path))


def create_contact_sheet(category, files):
    rows = (len(files) + COLUMNS - 1) // COLUMNS

    sheet = Image.new(
        "RGBA",
        (COLUMNS * CELL_WIDTH, rows * CELL_HEIGHT),
        (35, 38, 45, 255),
    )

    draw = ImageDraw.Draw(sheet)

    for index, file in enumerate(files):
        image = Image.open(file).convert("RGBA")

        x = (index % COLUMNS) * CELL_WIDTH
        y = (index // COLUMNS) * CELL_HEIGHT

        icon_x = x + (CELL_WIDTH - ICON_SIZE) // 2
        icon_y = y + 5

        sheet.alpha_composite(
            image,
            (icon_x, icon_y),
        )

        draw.text(
            (x + 8, y + 140),
            file.stem,
            fill=(235, 235, 235, 255),
        )

    sheet.save(
        OUTPUT / (
            "tower_contact_sheet.png"
            if category == "towers"
            else "troop_contact_sheet.png"
        )
    )


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)

        page = browser.new_page(
            viewport={
                "width": ICON_SIZE,
                "height": ICON_SIZE,
            },
            device_scale_factor=1,
        )

        for category in ("towers", "troops"):
            source = ASSET_ROOT / category / "official_svg"
            destination = OUTPUT / category

            destination.mkdir(parents=True, exist_ok=True)

            svg_files = sorted(source.glob("*.svg"))
            rendered_files = []

            print(f"\nRendering {category}: {len(svg_files)} icons")

            for svg in svg_files:
                png = destination / f"{svg.stem}.png"

                render_svg(page, svg, png)
                rendered_files.append(png)

                print(f"  Rendered: {svg.stem}")

            create_contact_sheet(category, rendered_files)

        browser.close()

    print("\nRendering complete.")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()
