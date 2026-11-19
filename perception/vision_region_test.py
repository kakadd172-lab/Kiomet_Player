
import base64
import json
import time
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image


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
MODEL_NAME = "qwen3-vl:4b-instruct"

TIMEOUT = 300
OVERLAP = 0.12


# --------------------------------------------------
# Image regions
# --------------------------------------------------

def create_regions(image):
    width, height = image.size

    mid_x = width // 2
    mid_y = height // 2

    overlap_x = int(width * OVERLAP / 2)
    overlap_y = int(height * OVERLAP / 2)

    return [
        {
            "name": "top_left",
            "bounds": (
                0, 0,
                mid_x + overlap_x,
                mid_y + overlap_y
            )
        },
        {
            "name": "top_right",
            "bounds": (
                mid_x - overlap_x, 0,
                width,
                mid_y + overlap_y
            )
        },
        {
            "name": "bottom_left",
            "bounds": (
                0,
                mid_y - overlap_y,
                mid_x + overlap_x,
                height
            )
        },
        {
            "name": "bottom_right",
            "bounds": (
                mid_x - overlap_x,
                mid_y - overlap_y,
                width,
                height
            )
        }
    ]


def encode_crop(image, bounds):
    cropped = image.crop(bounds)

    buffer = BytesIO()
    cropped.save(buffer, format="PNG")

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return encoded


# --------------------------------------------------
# Vision prompt
# --------------------------------------------------

def build_prompt():
    return """
You are the visual perception component of KiometAI.

Analyze this cropped region of a Kiomet strategy game screenshot.

Focus primarily on the actual game world.

Identify visible game entities such as:
- Towers
- Cities
- Castles
- Other identifiable structures
- Units, if clearly visible

Also identify visible territory colors and boundaries.

Return JSON using this structure:

{
    "entities": [
        {
            "type": "tower | city | castle | unit | other_structure | unknown",
            "center": {
                "x": 0.0,
                "y": 0.0
            },
            "visible_color": null,
            "ownership": "player | neutral | enemy | unknown",
            "confidence": 0.0
        }
    ],
    "territories": [
        {
            "visible_color": null,
            "label": null,
            "confidence": 0.0
        }
    ],
    "visible_text": [],
    "observations": [],
    "uncertainties": []
}

Rules:

1. Coordinates must be normalized between 0.0 and 1.0.
2. Coordinates refer to the CENTER of an entity.
3. (0,0) is the top-left of THIS CROPPED IMAGE.
4. (1,1) is the bottom-right of THIS CROPPED IMAGE.
5. Only identify entities that are actually visible.
6. Do not invent structures or exact counts.
7. Do not confuse territory boundaries with structures.
8. Do not infer ownership from proximity alone.
9. If an object is too small or unclear, use unknown.
10. Do not report interface icons as game structures.
11. Use empty arrays when information is unavailable.
12. Confidence must be between 0.0 and 1.0.

Return valid JSON only.
"""


# --------------------------------------------------
# Model request
# --------------------------------------------------

def analyze_region(encoded_image):

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
        "keep_alive": "5m",
        "options": {
            "temperature": 0.1
        }
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=TIMEOUT
    )

    response.raise_for_status()

    result = response.json()

    return json.loads(result["message"]["content"])


# --------------------------------------------------
# Coordinate conversion
# --------------------------------------------------

def convert_coordinates(entities, bounds, full_size):

    left, top, right, bottom = bounds
    full_width, full_height = full_size

    converted = []

    for entity in entities:

        try:
            local_x = float(entity["center"]["x"])
            local_y = float(entity["center"]["y"])

            if not (0 <= local_x <= 1 and 0 <= local_y <= 1):
                continue

            full_x = (
                left + local_x * (right - left)
            ) / full_width

            full_y = (
                top + local_y * (bottom - top)
            ) / full_height

            item = dict(entity)

            item["center_full_screen"] = {
                "x": round(full_x, 4),
                "y": round(full_y, 4)
            }

            converted.append(item)

        except (KeyError, TypeError, ValueError):
            continue

    return converted


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("========================================")
    print("      KIOMETAI REGIONAL VISION TEST")
    print("========================================")

    if not SCREENSHOT.exists():
        print(f"[Error] Screenshot not found: {SCREENSHOT}")
        return

    RESULTS_DIR.mkdir(exist_ok=True)

    image = Image.open(SCREENSHOT).convert("RGB")

    print(f"Screenshot: {SCREENSHOT.name}")
    print(f"Dimensions: {image.width} x {image.height}")
    print(f"Model: {MODEL_NAME}")

    regions = create_regions(image)

    results = {
        "model": MODEL_NAME,
        "screenshot": SCREENSHOT.name,
        "image_size": {
            "width": image.width,
            "height": image.height
        },
        "regions": [],
        "combined_entities": []
    }

    start_total = time.perf_counter()

    for index, region in enumerate(regions, start=1):

        name = region["name"]
        bounds = region["bounds"]

        print("\n----------------------------------------")
        print(f"Region {index}/4: {name}")
        print("----------------------------------------")

        encoded = encode_crop(image, bounds)

        start = time.perf_counter()

        try:
            data = analyze_region(encoded)

            elapsed = round(
                time.perf_counter() - start, 2
            )

            entities = data.get("entities", [])

            converted = convert_coordinates(
                entities,
                bounds,
                image.size
            )

            results["regions"].append({
                "name": name,
                "bounds": bounds,
                "elapsed_seconds": elapsed,
                "result": data,
                "converted_entities": converted
            })

            results["combined_entities"].extend(converted)

            print(f"Time: {elapsed}s")
            print(f"Entities detected: {len(converted)}")

            for entity in converted:
                print(
                    f"  {entity.get('type')} "
                    f"at {entity['center_full_screen']}"
                )

        except requests.exceptions.Timeout:
            print("[Error] Model timed out.")

        except requests.exceptions.ConnectionError:
            print("[Error] Cannot connect to Ollama.")
            break

        except (requests.exceptions.RequestException,
                json.JSONDecodeError,
                ValueError) as error:
            print(f"[Error] {error}")

        except Exception as error:
            print(f"[Unexpected error] {error}")

    results["total_elapsed_seconds"] = round(
        time.perf_counter() - start_total, 2
    )

    output_path = RESULTS_DIR / "regional_vision_test.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n========================================")
    print("          TEST COMPLETED")
    print("========================================")

    print(
        "Total time:",
        results["total_elapsed_seconds"],
        "seconds"
    )

    print(
        "Total raw detections:",
        len(results["combined_entities"])
    )

    print("Saved:", output_path)


if __name__ == "__main__":
    main()
