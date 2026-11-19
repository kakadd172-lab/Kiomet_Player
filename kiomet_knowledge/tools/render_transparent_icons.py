from pathlib import Path
from base64 import b64encode

from PIL import Image, ImageDraw
from playwright.sync_api import sync_playwright


PROJECT = Path(__file__).resolve().parents[2]
ASSET_ROOT = PROJECT / "kiomet_knowledge" / "assets"
OUTPUT = ASSET_ROOT / "derived" / "official_icon_renders"

ICON_SIZE = 128
COLUMNS = 4
CELL_WIDTH = 160
CELL_HEIGHT = 180

SOURCE_FOLDERS = {
    "towers": ASSET_ROOT / "towers" / "official_svg",
    "troops": ASSET_ROOT / "troops" / "official_svg",
}


def render_svg(page, svg_path: Path, output_path: Path):
    encoded = b64encode(svg_path.read_bytes()).decode("ascii")

    html = f"""
    <!DOCTYPE html>
    <html>
      <head>
        <style>
          html, body {{
            margin: 0;
            padding: 0;
            width: {ICON_SIZE}px;
            height: {ICON_SIZE}px;
            overflow: hidden;
            background: transparent;
          }}
          img {{
            display: block;
            width: {ICON_SIZE}px;
            height: {ICON_SIZE}px;
            object-fit: contain;
          }}
        </style>
      </head>
      <body>
        <img id="icon"
             src="data:image/svg+xml;base64,{encoded}">
      </body>
    </html>
    """

    page.set_content(html)
    page.locator("#icon").evaluate("(img) => img.decode()")

    page.screenshot(
        path=str(output_path),
        omit_background=True,
    )


def create_contact_sheet(folder: Path, output_path: Path, category: str):
    images = sorted(folder.glob("*.png"))

    if not images:
        print(f"No images found for {category}")
        return

    rows = (len(images) + COLUMNS - 1) // COLUMNS

    sheet = Image.new(
        "RGBA",
        (COLUMNS * CELL_WIDTH, rows * CELL_HEIGHT),
        (35, 38, 45, 255),
    )

    draw = ImageDraw.Draw(sheet)

    for index, image_path in enumerate(images):
        image = Image.open(image_path).convert("RGBA")

        column = index % COLUMNS
        row = index // COLUMNS

        x = column * CELL_WIDTH
        y = row * CELL_HEIGHT

        icon_x = x + (CELL_WIDTH - image.width) // 2
        icon_y = y + 5

        sheet.alpha_composite(image, (icon_x, icon_y))

        label = image_path.stem.replace("_", " ").title()
        draw.text(
            (x + 8, y + ICON_SIZE + 18),
            label,
            fill=(255, 255, 255, 255),
        )

    sheet.convert("RGB").save(output_path)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)

        context = browser.new_context(
            viewport={"width": ICON_SIZE, "height": ICON_SIZE},
            device_scale_factor=1,
        )

        page = context.new_page()

        for category, source_folder in SOURCE_FOLDERS.items():
            if not source_folder.exists():
                print(f"Missing source folder: {source_folder}")
                continue

            destination = OUTPUT / category
            destination.mkdir(parents=True, exist_ok=True)

            svg_files = sorted(source_folder.glob("*.svg"))

            print(f"\nRendering {category}: {len(svg_files)} SVGs")

            for svg_path in svg_files:
                output_path = destination / f"{svg_path.stem}.png"
                render_svg(page, svg_path, output_path)
                print(f"  Rendered: {svg_path.name}")

            contact_sheet = OUTPUT / f"{category[:-1]}_contact_sheet.png"
            create_contact_sheet(
                destination,
                contact_sheet,
                category,
            )

            print(f"Contact sheet: {contact_sheet}")

        browser.close()

    print("\nRendering complete.")


if __name__ == "__main__":
    main()
