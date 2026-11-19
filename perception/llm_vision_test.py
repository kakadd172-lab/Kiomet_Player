"""Run a concise, structured visual inspection of a Kiomet screenshot."""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCREENSHOTS_DIR = PROJECT_ROOT / "screenshots"
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5vl:3b"
TIMEOUT_SECONDS = 180

VISION_REPORT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "screen_type": {"type": "string", "enum": ["gameplay", "main_menu", "loading", "unknown"]},
        "overview": {"type": "string"},
        "observations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "area": {"type": "string", "enum": ["map", "entity", "territory", "interface", "text", "other"]},
                    "description": {"type": "string"},
                    "visual_evidence": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["area", "description", "visual_evidence", "confidence"],
            },
        },
        "readable_text": {"type": "array", "items": {"type": "string"}},
        "uncertainties": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["screen_type", "overview", "observations", "readable_text", "uncertainties"],
}


def select_screenshot(argument: str | None) -> Path:
    if argument:
        path = Path(argument).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.is_file():
            raise FileNotFoundError(f"Screenshot not found: {path}")
        return path

    candidates = list(SCREENSHOTS_DIR.glob("screen_*.png"))
    if not candidates:
        raise FileNotFoundError(
            f"No screen_*.png files found in {SCREENSHOTS_DIR}; pass --image."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def build_prompt() -> str:
    return (
        "Inspect this Kiomet screenshot. Return only the requested JSON report. "
        "Write short, complete, factual sentences in ordinary English. Describe "
        "only visible evidence; do not guess game mechanics, ownership, counts, "
        "or text that cannot be read. Each observation must say what is visible "
        "and the visual evidence supporting it. Use low confidence or an uncertainty "
        "when details are ambiguous. If there is no useful observation, return an "
        "empty observations array. Never output random letters, alphabet sequences, "
        "repeated filler, or text outside the JSON object."
    )


def inspect_screenshot(image_path: str | Path) -> dict[str, Any]:
    encoded = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": [{
                "role": "user",
                "content": build_prompt(),
                "images": [encoded],
            }],
            "format": VISION_REPORT_SCHEMA,
            "stream": False,
            "keep_alive": "5m",
            "options": {"temperature": 0, "num_predict": 1200},
        },
        timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("done_reason") == "length":
        raise ValueError("Vision report was truncated; incomplete text was rejected.")
    content = payload.get("message", {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Vision model returned an empty report.")
    try:
        report = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError(f"Vision model returned invalid JSON: {error.msg}.") from error
    if not isinstance(report, dict) or set(report) != {
        "screen_type", "overview", "observations", "readable_text", "uncertainties"
    }:
        raise ValueError("Vision report has missing or unexpected fields.")
    if report["screen_type"] not in {"gameplay", "main_menu", "loading", "unknown"}:
        raise ValueError("Vision report contains an invalid screen_type.")
    for key in ("overview",):
        if not isinstance(report[key], str):
            raise ValueError(f"Vision report {key} must be a string.")
    for key in ("observations", "readable_text", "uncertainties"):
        if not isinstance(report[key], list):
            raise ValueError(f"Vision report {key} must be an array.")
    for index, item in enumerate(report["observations"]):
        if not isinstance(item, dict) or set(item) != {
            "area", "description", "visual_evidence", "confidence"
        }:
            raise ValueError(f"Observation {index} has an invalid structure.")
        if item["area"] not in {"map", "entity", "territory", "interface", "text", "other"}:
            raise ValueError(f"Observation {index} has an invalid area.")
        if not all(isinstance(item[key], str) for key in ("description", "visual_evidence")):
            raise ValueError(f"Observation {index} must contain text descriptions.")
        confidence = item["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise ValueError(f"Observation {index} confidence must be between 0 and 1.")
    for key in ("readable_text", "uncertainties"):
        if not all(isinstance(value, str) for value in report[key]):
            raise ValueError(f"Vision report {key} must contain strings.")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", help="Screenshot path; defaults to newest screen_*.png.")
    args = parser.parse_args()

    try:
        screenshot = select_screenshot(args.image)
        print(f"Screenshot: {screenshot}")
        print(f"Model: {MODEL}")
        report = inspect_screenshot(screenshot)
    except requests.exceptions.ConnectionError:
        print("[Error] Cannot connect to Ollama. Check that Ollama is running.")
        return 1
    except requests.exceptions.Timeout:
        print("[Error] Ollama did not respond before the timeout.")
        return 1
    except requests.exceptions.RequestException as error:
        print(f"[Error] Ollama request failed: {error}")
        return 1
    except (FileNotFoundError, ValueError) as error:
        print(f"[Error] {error}")
        return 1

    print("\n========== STRUCTURED VISUAL REPORT ==========")
    print(f"Screen: {report['screen_type']}")
    print(f"Overview: {report['overview']}")
    print("\nObservations:")
    for item in report["observations"]:
        print(
            f"- [{item['area']}] {item['description']} "
            f"(confidence {item['confidence']:.2f})"
        )
        print(f"  Evidence: {item['visual_evidence']}")
    print("\nReadable text:")
    for item in report["readable_text"]:
        print(f"- {item}")
    print("\nUncertainties:")
    for item in report["uncertainties"]:
        print(f"- {item}")
    print("==============================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
