
import base64
import json
import requests

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCREENSHOT_DIR = PROJECT_ROOT / "screenshots"

MODEL = "qwen2.5vl:3b"
API_URL = "http://localhost:11434/api/chat"


def get_latest_screenshot():
    screenshots = list(
        SCREENSHOT_DIR.glob("screen_*.png")
    )

    if not screenshots:
        raise FileNotFoundError(
            "No screenshots found in the screenshot directory."
        )

    return max(
        screenshots,
        key=lambda screenshot: screenshot.stat().st_mtime
    )


def classify_screen(image_path):
    image_data = base64.b64encode(
        image_path.read_bytes()
    ).decode("utf-8")

    prompt = """
You are analyzing a screenshot of the browser game Kiomet.

Your primary task is to identify the CURRENT GAME SCREEN,
not describe the website as a whole.

Possible screen types:

- main_menu: The game start interface is visible,
  including a Play button or other start controls.
- loading: The game is visibly loading.
- gameplay: The actual playable strategy map is active.
- pause_menu: An active game is covered by a pause menu.
- unknown: None of these can be identified reliably.

IMPORTANT INSTRUCTIONS:

1. Examine the central visible interface first.
2. Distinguish actual visible buttons from ordinary webpage text.
3. Do not classify the page as unknown merely because
   it contains help text or promotional information.
4. Do not mistake the website footer for the game interface.
5. Never invent interface elements.
6. Use confidence between 0 and 1.
7. If uncertain, explain exactly why.

Return JSON only:

{
  "screen_type": "string",
  "confidence": 0.0,
  "visible_buttons": [],
  "visible_text": [],
  "gameplay_visible": false,
  "uncertain_details": []
}
"""

    response = requests.post(
        API_URL,
        json={
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_data]
                }
            ],
            "format": "json",
            "stream": False
        },
        timeout=300
    )

    response.raise_for_status()

    content = response.json()["message"]["content"]

    return json.loads(content)


def main():
    print("[Classifier] Finding latest screenshot...")

    screenshot = get_latest_screenshot()

    print(f"[Classifier] Using: {screenshot.name}")
    print("[Classifier] Analyzing screen...")

    result = classify_screen(screenshot)

    print("\n========== SCREEN CLASSIFICATION ==========")
    print(json.dumps(result, indent=4))
    print("===========================================")


if __name__ == "__main__":
    main()
