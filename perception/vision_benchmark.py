
import base64
import json
import time
from pathlib import Path

import requests


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SCREENSHOT = (
    PROJECT_ROOT
    / "screenshots"
    / "screen_20261118_225535.png"
)

RESULTS_DIR = PROJECT_ROOT / "benchmark_results"

OLLAMA_URL = "http://localhost:11434/api/chat"

MODELS = [
    "qwen2.5vl:3b",
    "lfm3-vl:2b",
    "qwen3-vl:4b-instruct",
]

TIMEOUT = 300


# --------------------------------------------------
# Benchmark prompt
# --------------------------------------------------

def build_prompt():
    return """
You are the visual perception component of KiometAI.

Analyze this screenshot of the Kiomet strategy game.

Extract only information that is actually visible.
Do not invent objects, counts, ownership, or coordinates.

Return a JSON object using this structure:

{
    "screen_type": "gameplay | main_menu | loading | unknown",
    "player": {
        "name": null,
        "visible_status": null,
        "visible_color": null
    },
    "map": {
        "visible": false,
        "visible_territories": []
    },
    "entities": [
        {
            "type": "unknown",
            "approximate_position": {
                "x": 0.0,
                "y": 0.0
            },
            "visible_color": null,
            "ownership": "player | neutral | enemy | unknown",
            "confidence": 0.0
        }
    ],
    "interface": {
        "visible_text": [],
        "visible_buttons": [],
        "notifications": []
    },
    "spatial_observations": [],
    "uncertainties": []
}

Rules:

1. Coordinates must be normalized between 0.0 and 1.0.
   (0,0) is the top-left corner.
   (1,1) is the bottom-right corner.

2. Only include entities that you can visually identify.
3. Do not assume a structure belongs to the player
   simply because it is inside nearby territory.
4. Distinguish structures from territory boundaries
   and decorative map elements.
5. Do not invent exact unit or structure counts.
6. If ownership is unclear, use "unknown".
7. If the map is not visible, set map.visible to false.
8. Use empty arrays when information is unavailable.
9. Include confidence values between 0.0 and 1.0.
10. Return JSON only. No Markdown or explanations.
"""


# --------------------------------------------------
# Image handling
# --------------------------------------------------

def encode_image(path):
    with open(path, "rb") as image_file:
        return base64.b64encode(
            image_file.read()
        ).decode("utf-8")


# --------------------------------------------------
# Model execution
# --------------------------------------------------

def run_model(model_name, encoded_image):

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": build_prompt(),
                "images": [encoded_image]
            }
        ],
        "stream": False,
        "keep_alive": 0,
        "options": {
            "temperature": 0.1
        }
    }

    start_time = time.perf_counter()

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=TIMEOUT
    )

    elapsed = time.perf_counter() - start_time

    response.raise_for_status()

    result = response.json()
    content = result["message"]["content"]

    try:
        parsed = json.loads(content)
        valid_json = True

    except json.JSONDecodeError:
        parsed = None
        valid_json = False

    return {
        "model": model_name,
        "elapsed_seconds": round(elapsed, 2),
        "valid_json": valid_json,
        "response": content,
        "parsed_response": parsed
    }


# --------------------------------------------------
# Main benchmark
# --------------------------------------------------

def main():

    print("========================================")
    print("         KIOMETAI VISION TEST")
    print("========================================")

    if not SCREENSHOT.exists():
        print(f"[Error] Screenshot not found: {SCREENSHOT}")
        return

    RESULTS_DIR.mkdir(exist_ok=True)

    print(f"\nScreenshot: {SCREENSHOT.name}")

    encoded_image = encode_image(SCREENSHOT)

    print(f"Image size: {len(encoded_image) / 1024:.1f} KB")

    results = []

    for index, model in enumerate(MODELS, start=1):

        print("\n----------------------------------------")
        print(f"Model {index}/{len(MODELS)}: {model}")
        print("----------------------------------------")

        try:
            result = run_model(model, encoded_image)
            results.append(result)

            print(f"Time: {result['elapsed_seconds']} seconds")
            print(f"Valid JSON: {result['valid_json']}")

            output_path = RESULTS_DIR / (
                model.replace(":", "_").replace("/", "_")
                + ".json"
            )

            with open(output_path, "w", encoding="utf-8") as file:
                json.dump(
                    result,
                    file,
                    indent=4,
                    ensure_ascii=False
                )

            print(f"Saved: {output_path.name}")

        except requests.exceptions.ConnectionError:
            print("[Error] Cannot connect to Ollama.")
            print("Check whether Ollama is running.")

        except requests.exceptions.Timeout:
            print("[Error] Model timed out.")

        except Exception as error:
            print(f"[Error] {error}")

    print("\n========================================")
    print("           BENCHMARK SUMMARY")
    print("========================================")

    for result in results:
        print(
            f"{result['model']:<28}"
            f"{result['elapsed_seconds']:>8.2f}s   "
            f"JSON: {result['valid_json']}"
        )

    print("\nResults directory:", RESULTS_DIR)
    print("Benchmark completed.")


if __name__ == "__main__":
    main()
