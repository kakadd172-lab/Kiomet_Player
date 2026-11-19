
from datetime import datetime
from pathlib import Path
import time

from playwright.sync_api import sync_playwright


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCREENSHOT_DIR = PROJECT_ROOT / "screenshots"
GAME_URL = "https://kiomet.com/"


def capture_screen(page):
    """Capture the current visible browser screen."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    screenshot_path = SCREENSHOT_DIR / f"screen_{timestamp}.png"

    page.screenshot(
        path=str(screenshot_path),
        full_page=False
    )

    print(f"[Observer] Screenshot saved: {screenshot_path.name}")

    return screenshot_path


def main():
    print("[Observer] Starting KiometAI visual observer...")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)

        page = browser.new_page(
            viewport={"width": 1440, "height": 900}
        )

        print("[Observer] Opening Kiomet...")
        page.goto(GAME_URL, wait_until="domcontentloaded")

        page.wait_for_timeout(5000)

        print("[Observer] Beginning observation.")
        print("[Observer] Press Ctrl+C in this terminal to stop.")

        try:
            while True:
                capture_screen(page)
                time.sleep(5)

        except KeyboardInterrupt:
            print("\n[Observer] Observation stopped.")

        finally:
            browser.close()


if __name__ == "__main__":
    main()
