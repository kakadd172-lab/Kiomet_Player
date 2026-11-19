
import argparse
import base64
import json
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "benchmark_results"
PREVIEW_DIR = OUTPUT_DIR / "region_previews"

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3-vl:4b-instruct"

PROMPT = """
You are examining a screenshot from a strategy game.

Describe only what is visibly present in the image. Do not assume
that you know what a symbol means if its meaning is unclear.

Focus on these things:

1. MAP:
   Describe the visible map, its background, and its general layout.

2. STRUCTURES:
   Describe every distinct type of small structure-like or symbol-like
   object you can visually distinguish.
   Describe their approximate shapes and colours.
   Distinguish blue/highlighted objects from gray or neutral objects.

3. TERRITORY:
   Describe any visible coloured territory, its approximate shape,
   and which structures appear to be inside or outside it.

4. CONNECTIONS:
   Describe any visible lines or links between structures.

5. INTERFACE:
   Mention text, notifications, player information, or controls that
   are visible around the edges of the image.

6. UNCERTAINTIES:
   Explain which details are too small, obscured, or ambiguous to identify.

Important:
- Do not invent structures that you cannot see.
- Do not guess exact game meanings from appearance alone.
- Avoid exact counts unless the objects are clearly distinguishable.
- Separate direct visual observations from uncertain interpretations.
- Give a useful, detailed description rather than a one-line summary.

This is a visual inspection task, not a request to play the game.
"""


def find_screenshot(argument):
    if argument:
        path = Path(argument).expanduser()
        if not path.is_absolute():
            path = ROOT / path

        if not path.is_file():
            raise FileNotFoundError(f"Screenshot not found: {path}")

        return path

    screenshots_dir = ROOT / "screenshots"
    candidates = list(screenshots_dir.glob("screen_*.png"))

    if not candidates:
        raise FileNotFoundError(
            "No screen_*.png files found in the screenshots folder. "
            "Pass a screenshot path as an argument."
        )

    return max(candidates, key=lambda p: p.stat().st_mtime)


def make_views(image):
    width, height = image.size

    # Keep the full screenshot unchanged.
    full_view = image.copy()

    # Broad battlefield view, excluding most of the outer page margins.
    broad_box = (
        int(width * 0.17),
        0,
        int(width * 0.98),
        int(height * 0.95),
    )
    broad_view = image.crop(broad_box)

    # Closer view around the central territory and nearby structures.
    center_box = (
        int(width * 0.30),
        int(height * 0.11),
        int(width * 0.77),
        int(height * 0.79),
    )
    center_view = image.crop(center_box)

    # Upscale the closer crop so small symbols occupy more pixels.
    center_view = center_view.resize(
        (center_view.width * 2, center_view.height * 2),
        Image.Resampling.LANCZOS,
    )

    return {
        "full_screenshot": full_view,
        "broad_battlefield": broad_view,
        "central_closeup": center_view,
    }


def encode_image(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def inspect_view(name, image):
    payload = {
        "model": MODEL,
        "stream": False,
        "keep_alive": "5m",
        "format": "",
        "messages": [
            {
                "role": "user",
                "content": PROMPT,
                "images": [encode_image(image)],
            }
        ],
        "options": {
            "temperature": 0.1,
        },
    }

    started = time.perf_counter()

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=240,
    )
    response.raise_for_status()

    data = response.json()
    elapsed = time.perf_counter() - started

    answer = data.get("message", {}).get("content", "").strip()

    if not answer:
        answer = "[The model returned an empty response.]"

    return {
        "view": name,
        "seconds": round(elapsed, 2),
        "response": answer,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Probe Qwen3-VL's visual understanding of Kiomet."
    )
    parser.add_argument(
        "--image",
        help="Optional screenshot path. Defaults to the newest screen_*.png.",
    )
    args = parser.parse_args()

    screenshot = find_screenshot(args.image)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    with Image.open(screenshot) as source:
        original = source.convert("RGB")

    print(f"Screenshot: {screenshot}")
    print(f"Image size: {original.width} x {original.height}")
    print(f"Model: {MODEL}")
    print()

    views = make_views(original)
    results = []

    for index, (name, view) in enumerate(views.items(), start=1):
        preview_path = PREVIEW_DIR / f"{name}.png"
        view.save(preview_path)

        print(f"[{index}/{len(views)}] Inspecting: {name}")
        print(f"View size: {view.width} x {view.height}")
        print("Sending image to Ollama...")

        try:
            result = inspect_view(name, view)
            results.append(result)

            print(f"Completed in {result['seconds']} seconds.")
            print("-" * 70)
            print(result["response"])
            print("-" * 70)

        except requests.RequestException as error:
            print(f"Request failed: {error}")
            results.append({
                "view": name,
                "seconds": None,
                "error": str(error),
            })

    output = {
        "screenshot": str(screenshot),
        "image_size": [original.width, original.height],
        "model": MODEL,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "results": results,
    }

    output_path = OUTPUT_DIR / "vision_region_probe.json"
    output_path.write_text(
        json.dumps(output, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("Probe finished.")
    print(f"Results: {output_path}")
    print(f"Crop previews: {PREVIEW_DIR}")


if __name__ == "__main__":
    main()
