
from pathlib import Path
from playwright.sync_api import sync_playwright


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)

        page = browser.new_page(
            viewport={"width": 1440, "height": 900}
        )

        print("Opening Kiomet...")
        page.goto(
            "https://kiomet.com/",
            wait_until="domcontentloaded"
        )

        page.wait_for_timeout(5000)

        Path("screenshots").mkdir(exist_ok=True)

        page.screenshot(
            path="screenshots/initial_screen.png",
            full_page=False
        )

        print("Screenshot captured successfully!")
        print("Saved: screenshots/initial_screen.png")

        input("Press Enter to close the browser...")

        browser.close()


if __name__ == "__main__":
    main()
