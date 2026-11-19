"""Extract a conservative, schema-validated Kiomet state from one screenshot.

The vision model supplies observations only. This module rejects malformed or
structurally inconsistent output instead of printing arbitrary model text as
game state.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import requests


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCREENSHOTS_DIR = PROJECT_ROOT / "screenshots"
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5vl:3b"
REQUEST_TIMEOUT_SECONDS = 180

GAME_STATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "screen_type": {"type": "string", "enum": ["gameplay", "main_menu", "loading", "unknown"]},
        "player": {
            "type": "object",
            "properties": {
                "name": {"type": ["string", "null"]},
                "team_or_color": {"type": ["string", "null"]},
            },
            "required": ["name", "team_or_color"],
        },
        "game_status": {
            "type": "object",
            "properties": {
                "mode": {"type": ["string", "null"]},
                "timer": {"type": ["string", "null"]},
            },
            "required": ["mode", "timer"],
        },
        "map": {
            "type": "object",
            "properties": {
                "visible": {"type": "boolean"},
                "terrain": {"type": "array", "items": {"type": "string"}},
                "regions": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["visible", "terrain", "regions"],
        },
        "entities": {
            "type": "object",
            "properties": {
                "cities": {"type": "array", "items": {"$ref": "#/$defs/entity"}},
                "castles": {"type": "array", "items": {"$ref": "#/$defs/entity"}},
                "units": {"type": "array", "items": {"$ref": "#/$defs/entity"}},
                "other_structures": {"type": "array", "items": {"$ref": "#/$defs/entity"}},
            },
            "required": ["cities", "castles", "units", "other_structures"],
        },
        "interface": {
            "type": "object",
            "properties": {
                "resources": {"type": "array", "items": {"type": "string"}},
                "visible_buttons": {"type": "array", "items": {"type": "string"}},
                "visible_text": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["resources", "visible_buttons", "visible_text"],
        },
        "spatial_observations": {"type": "array", "items": {"type": "string"}},
        "uncertainties": {"type": "array", "items": {"type": "string"}},
    },
    "$defs": {
        "entity": {
            "type": "object",
            "properties": {
                "type": {"type": "string"},
                "approximate_position": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number", "minimum": 0, "maximum": 1},
                        "y": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": ["x", "y"],
                },
                "visible_owner_or_color": {"type": ["string", "null"]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["type", "approximate_position", "visible_owner_or_color", "confidence"],
        }
    },
    "required": [
        "screen_type", "player", "game_status", "map", "entities",
        "interface", "spatial_observations", "uncertainties",
    ],
}


def get_latest_screenshot() -> Path:
    screenshots = list(SCREENSHOTS_DIR.glob("screen_*.png"))
    if not screenshots:
        raise FileNotFoundError(
            f"No screenshots found in {SCREENSHOTS_DIR}; capture a screenshot first."
        )
    return max(screenshots, key=lambda path: path.stat().st_mtime)


def encode_image(image_path: str | Path) -> str:
    return base64.b64encode(Path(image_path).read_bytes()).decode("ascii")


def build_prompt() -> str:
    return (
        "Inspect this Kiomet screenshot and return only the requested JSON object. "
        "Report direct visual evidence, not guesses. Use null for unreadable or "
        "unavailable text and empty arrays when nothing can be identified. "
        "Do not invent names, counts, ownership, timers, resource values, or entities. "
        "For each entity, give its center position as normalized x/y coordinates "
        "from 0 to 1 relative to the full screenshot. Confidence describes visual "
        "certainty, not game importance. Keep descriptions short and factual. "
        "Do not output alphabet sequences, filler text, Markdown, or explanations "
        "outside the JSON object. If the screen is not gameplay, set screen_type "
        "appropriately and leave unavailable gameplay collections empty."
    )


def validate_game_state(data: Any) -> dict[str, Any]:
    """Validate key structure and coordinate/confidence ranges before use."""
    if not isinstance(data, dict):
        raise ValueError("Model output must be a JSON object.")

    required_top = {
        "screen_type", "player", "game_status", "map", "entities",
        "interface", "spatial_observations", "uncertainties",
    }
    if set(data) != required_top:
        missing = sorted(required_top - set(data))
        extra = sorted(set(data) - required_top)
        raise ValueError(f"Invalid game-state fields; missing={missing}, extra={extra}.")
    if data["screen_type"] not in {"gameplay", "main_menu", "loading", "unknown"}:
        raise ValueError("screen_type must be gameplay, main_menu, loading, or unknown.")

    for section, keys in {
        "player": {"name", "team_or_color"},
        "game_status": {"mode", "timer"},
        "map": {"visible", "terrain", "regions"},
        "entities": {"cities", "castles", "units", "other_structures"},
        "interface": {"resources", "visible_buttons", "visible_text"},
    }.items():
        value = data[section]
        if not isinstance(value, dict) or set(value) != keys:
            raise ValueError(f"{section} must contain exactly: {', '.join(sorted(keys))}.")

    for section in ("player", "game_status"):
        if any(value is not None and not isinstance(value, str) for value in data[section].values()):
            raise ValueError(f"{section} values must be strings or null.")

    if not isinstance(data["map"]["visible"], bool):
        raise ValueError("map.visible must be a boolean.")
    for key in ("terrain", "regions"):
        if not isinstance(data["map"][key], list) or not all(isinstance(x, str) for x in data["map"][key]):
            raise ValueError(f"map.{key} must be an array of strings.")

    for key, entities in data["entities"].items():
        if not isinstance(entities, list):
            raise ValueError(f"entities.{key} must be an array.")
        for index, entity in enumerate(entities):
            if not isinstance(entity, dict) or set(entity) != {
                "type", "approximate_position", "visible_owner_or_color", "confidence"
            }:
                raise ValueError(f"entities.{key}[{index}] has an invalid structure.")
            if not isinstance(entity["type"], str):
                raise ValueError(f"entities.{key}[{index}].type must be a string.")
            if entity["visible_owner_or_color"] is not None and not isinstance(entity["visible_owner_or_color"], str):
                raise ValueError(f"entities.{key}[{index}].visible_owner_or_color must be a string or null.")
            position = entity["approximate_position"]
            if not isinstance(position, dict) or set(position) != {"x", "y"}:
                raise ValueError(f"entities.{key}[{index}].approximate_position must contain x and y.")
            for axis in ("x", "y"):
                coordinate = position[axis]
                if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)) or not 0 <= coordinate <= 1:
                    raise ValueError(f"entities.{key}[{index}].{axis} must be between 0 and 1.")
            confidence = entity["confidence"]
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
                raise ValueError(f"entities.{key}[{index}].confidence must be between 0 and 1.")

    for section, keys in {
        "interface": ("resources", "visible_buttons", "visible_text"),
    }.items():
        for key in keys:
            if not isinstance(data[section][key], list) or not all(isinstance(x, str) for x in data[section][key]):
                raise ValueError(f"{section}.{key} must be an array of strings.")

    for key in ("spatial_observations", "uncertainties"):
        if not isinstance(data[key], list) or not all(isinstance(x, str) for x in data[key]):
            raise ValueError(f"{key} must be an array of strings.")
    return data


def extract_game_state(image_path: str | Path) -> dict[str, Any]:
    payload = {
        "model": MODEL_NAME,
        "messages": [{
            "role": "user",
            "content": build_prompt(),
            "images": [encode_image(image_path)],
        }],
        "format": GAME_STATE_SCHEMA,
        "stream": False,
        "keep_alive": "5m",
        "options": {"temperature": 0, "num_predict": 1800},
    }
    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    result = response.json()
    if result.get("done_reason") == "length":
        raise ValueError("Vision model output was truncated; no partial game state was accepted.")
    content = result.get("message", {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Vision model returned an empty game-state response.")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError(f"Vision model returned invalid JSON: {error.msg}.") from error
    return validate_game_state(parsed)


def main() -> int:
    print("[KiometAI] Selecting the newest gameplay screenshot...")
    try:
        screenshot = get_latest_screenshot()
        print(f"[KiometAI] Screenshot: {screenshot.name}")
        print(f"[KiometAI] Model: {MODEL_NAME}")
        print("[KiometAI] Extracting structured visual observations...")
        game_state = extract_game_state(screenshot)
    except requests.exceptions.ConnectionError:
        print("[Error] Cannot connect to Ollama. Check that Ollama is running.")
        return 1
    except requests.exceptions.Timeout:
        print("[Error] Ollama did not respond before the 180-second timeout.")
        return 1
    except requests.exceptions.HTTPError as error:
        print(f"[Error] Ollama rejected the request: {error}")
        return 1
    except (FileNotFoundError, ValueError, KeyError, TypeError) as error:
        print(f"[Error] Game-state extraction failed: {error}")
        return 1

    print("\n========== VALIDATED KIOMET GAME STATE ==========")
    print(json.dumps(game_state, indent=2, ensure_ascii=False))
    print("=================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
