"""Run tiled VLM proposals and deterministic icon verification on a screenshot."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw

from perception.vision_entity_detector import VisionEntityDetector


ROOT = Path(__file__).resolve().parents[1]
SCREENSHOTS = ROOT / "screenshots"
OUTPUT_DIR = ROOT / "benchmark_results" / "entity_detection"


def latest_screenshot() -> Path:
    candidates = list(SCREENSHOTS.glob("screen_*.png"))
    if not candidates:
        raise FileNotFoundError(
            f"No screen_*.png files found in {SCREENSHOTS}; pass --image."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def serialize_result(item) -> dict:
    recognition = item.recognition
    return {
        "region_id": item.proposal.region_id,
        "tile": item.tile_name,
        "category": item.proposal.category,
        "bbox_normalized": {
            "x": item.proposal.x,
            "y": item.proposal.y,
            "width": item.proposal.width,
            "height": item.proposal.height,
        },
        "detector_confidence": item.detector_confidence,
        "recognition": {
            "accepted": recognition.accepted,
            "status": recognition.status,
            "icon_id": recognition.icon_id,
            "display_name": recognition.display_name,
            "top_score": recognition.top_score,
            "margin": recognition.margin,
            "candidates": [asdict(candidate) for candidate in recognition.candidates],
            "colour": asdict(recognition.colour) if recognition.colour else None,
        },
    }


def save_annotated(image_path: Path, results, output_path: Path) -> None:
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size

    for item in results:
        box = item.proposal
        left = round(box.x * width)
        top = round(box.y * height)
        right = round((box.x + box.width) * width)
        bottom = round((box.y + box.height) * height)
        accepted = item.recognition.accepted
        colour = (65, 220, 120) if accepted else (255, 175, 55)
        label = item.recognition.display_name or (
            f"{box.category}: {item.recognition.status}"
        )
        label += f" [{item.detector_confidence:.2f}/{item.recognition.top_score:.2f}]"
        draw.rectangle((left, top, right, bottom), outline=colour, width=3)
        text_box = draw.textbbox((left, max(0, top - 16)), label)
        draw.rectangle(text_box, fill=(15, 22, 30))
        draw.text((left, max(0, top - 16)), label, fill=colour)

    image.save(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Propose and verify Kiomet tower/unit regions with a local Ollama VLM."
    )
    parser.add_argument("--image", type=Path, help="Screenshot path; defaults to newest screenshot.")
    parser.add_argument("--model", default="qwen3-vl:4b-instruct")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-image-side", type=int, default=960, help="Resize the model input so its longest side is at most this many pixels.")
    parser.add_argument("--threads", type=int, default=4, help="Limit Ollama CPU inference threads.")
    parser.add_argument("--tiled", action="store_true", help="Use four overlapping tiles instead of one full-screen pass.")
    parser.add_argument("--min-detector-confidence", type=float, default=0.35)
    parser.add_argument("--min-score", type=float, default=0.65)
    parser.add_argument("--min-margin", type=float, default=0.10)
    parser.add_argument("--method", choices=("shape", "grayscale"), default="shape")
    args = parser.parse_args()

    screenshot = args.image or latest_screenshot()
    if not screenshot.is_absolute():
        screenshot = (ROOT / screenshot).resolve()
    if not screenshot.is_file():
        raise FileNotFoundError(screenshot)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_json = OUTPUT_DIR / f"entities_{stamp}.json"
    output_image = OUTPUT_DIR / f"entities_{stamp}.png"

    detector = VisionEntityDetector(
        model=args.model,
        timeout=args.timeout,
        max_image_side=args.max_image_side,
        num_threads=args.threads,
        tiled=args.tiled,
        min_detector_confidence=args.min_detector_confidence,
    )

    print(f"Screenshot: {screenshot}")
    print(f"Model: {args.model}")
    print("Proposing regions in " + ("four overlapping tiles..." if args.tiled else "one resized full-screen image..."))
    results = detector.detect(
        screenshot,
        method=args.method,
        min_score=args.min_score,
        min_margin=args.min_margin,
    )

    payload = {
        "screenshot": str(screenshot),
        "model": args.model,
        "method": args.method,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "proposal_count": len(results),
        "accepted_count": sum(item.recognition.accepted for item in results),
        "entities": [serialize_result(item) for item in results],
        "limitations": [
            "The vision model proposes boxes and may miss or invent regions.",
            "Icon matching verifies identities only within proposed boxes.",
            "Scores are similarity heuristics, not calibrated probabilities.",
            "This is an experimental detector and requires labelled screenshot evaluation.",
        ],
    }
    output_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    save_annotated(screenshot, results, output_image)

    print()
    print(f"Proposed regions: {len(results)}")
    print(f"Accepted identities: {payload['accepted_count']}")
    for item in results:
        name = item.recognition.display_name or "Unidentified"
        print(
            f"- {item.proposal.category}: {name} "
            f"({item.recognition.status}, "
            f"detector={item.detector_confidence:.2f}, "
            f"match={item.recognition.top_score:.2f})"
        )
    print(f"JSON: {output_json}")
    print(f"Annotated image: {output_image}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
