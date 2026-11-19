
import base64
import json
from pathlib import Path

import requests


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCREENSHOTS_DIR = PROJECT_ROOT / "screenshots"

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5vl:3b"


# --------------------------------------------------
# Screenshot selection
# --------------------------------------------------

def get_latest_screenshot():
    screenshots = list(SCREENSHOTS_DIR.glob("screen_*.png"))

    if not screenshots:
        raise FileNotFoundError(
            f"No screenshots found in {SCREENSHOTS_DIR}"
        )

    return max(screenshots, key=lambda file: file.stat().st_mtime)


# --------------------------------------------------
# Image encoding
# --------------------------------------------------

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(
            image_file.read()
        ).decode("utf-8")


# --------------------------------------------------
# Game-state extraction
# --------------------------------------------------

def build_prompt():
    return """
You are the visual perception component of KiometAI.

Analyze this screenshot of the Kiomet strategy game.

Your task is to extract only information that is actually
visible in the screenshot.

Do not invent missing information.
Do not assume that an object exists if you cannot identify it.
Use null or an empty array when information is unavailable.

Return valid JSON using this structure:

{
    "screen_type": "gameplay | main_menu | loading | unknown",
    "player": {
        "name": null,
        "team_or_color": null
    },
    "game_status": {
        "mode": null,
        "timer": null
    },
    "map": {
        "visible": false,
        "terrain": [],
        "regions": []
    },
    "entities": {
        "cities": [],
        "castles": [],
        "units": [],
        "other_structures": []
    },
    "interface": {
        "resources": [],
        "visible_buttons": [],
        "visible_text": []
    },
    "spatial_observations": [],
    "uncertainties": []
}

Important rules:

1. Only report objects you can actually identify.
2. For entities, describe their approximate screen position
   using normalized coordinates from 0.0 to 1.0.
3. Each entity should include:
   - type
   - approximate_position
   - visible_owner_or_color
   - confidence
4. Do not confuse decorative elements with game entities.
5. Do not assume ownership based only on proximity.
6. If text is too small or unclear, mention that in uncertainties.
7. Distinguish visible evidence from interpretation.
8. Do not invent exact unit counts or resource values.

Return JSON only. No Markdown or explanations.
"""


def extract_game_state(image_path):
    encoded_image = encode_image(image_path)

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": build_prompt(),
                "images": [encoded_image]
            }
        ],
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0.1
        }
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=180
    )

    response.raise_for_status()

    result = response.json()
    content = result["message"]["content"]

    return json.loads(content)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    print("[GameState] Finding latest screenshot...")

    screenshot = get_latest_screenshot()

    print(f"[GameState] Using: {screenshot.name}")
    print("[GameState] Analyzing battlefield...")
    print("[GameState] Sending image to local model...")

    try:
        game_state = extract_game_state(screenshot)

        print("\n========== EXTRACTED GAME STATE ==========")
        print(json.dumps(game_state, indent=4))
        print("==========================================")

    except requests.exceptions.ConnectionError:
        print("[Error] Cannot connect to Ollama.")
        print("[Error] Make sure Ollama is running.")

    except requests.exceptions.Timeout:
        print("[Error] Model took too long to respond.")

    except json.JSONDecodeError:
        print("[Error] Model returned invalid JSON.")

    except Exception as error:
        print(f"[Error] {error}")


if __name__ == "__main__":
    main()
