"""Evaluate the deterministic icon matcher against labelled screenshot crops.

Run from the repository root:
    python perception/icon_crop_benchmark.py

The King crop is currently known to be clipped at its top edge. It is still
reported for diagnosis, but excluded from the primary accuracy figure until
that crop is replaced.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kiomet_knowledge.recognition.knowledge_loader import load_knowledge
from kiomet_knowledge.recognition.shared.confidence import assess_matches
from kiomet_knowledge.recognition.shared.image_matcher import ImageMatcher


DEFAULT_CROPS_DIR = PROJECT_ROOT / "benchmark_crops_v2"
DEFAULT_OUTPUT = PROJECT_ROOT / "benchmark_results" / "icon_crop_benchmark.json"

# Labels were checked against the source gameplay screenshot by the user.
# A crop marked exclude_from_primary_accuracy is diagnostic only.
CASES = (
    {
        "file": "radar.png",
        "expected": "Radar",
        "category": "tower",
        "exclude_from_primary_accuracy": False,
    },
    {
        "file": "barracks_upper.png",
        "expected": "Barracks",
        "category": "tower",
        "exclude_from_primary_accuracy": False,
    },
    {
        "file": "barracks_lower.png",
        "expected": "Barracks",
        "category": "tower",
        "exclude_from_primary_accuracy": False,
    },
    {
        "file": "mine.png",
        "expected": "Mine",
        "category": "tower",
        "exclude_from_primary_accuracy": False,
    },
    {
        "file": "king_marker.png",
        "expected": "King",
        "category": "unit",
        "exclude_from_primary_accuracy": True,
        "crop_issue": "Known top-edge clipping; recrop before scoring accuracy.",
    },
)


def run_benchmark(crops_dir: Path, top_k: int = 5) -> dict[str, Any]:
    """Run labelled crops through the registered tower/unit templates."""
    knowledge = load_knowledge()
    matcher = ImageMatcher(knowledge.all_icons)
    rows: list[dict[str, Any]] = []

    for case in CASES:
        image_path = crops_dir / case["file"]
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing benchmark crop: {image_path}")

        matches = matcher.match(
            image_path,
            category=case["category"],
            top_k=top_k,
        )
        assessment = assess_matches(matches)
        expected = case["expected"]
        expected_rank = next(
            (
                rank
                for rank, match in enumerate(matches, start=1)
                if match.record.display_name.casefold() == expected.casefold()
            ),
            None,
        )

        rows.append(
            {
                "file": case["file"],
                "category": case["category"],
                "expected": expected,
                "expected_rank": expected_rank,
                "top_match": (
                    matches[0].record.display_name if matches else None
                ),
                "top_score": round(assessment.top_score, 4),
                "margin": (
                    round(assessment.margin, 4)
                    if assessment.margin is not None
                    else None
                ),
                "confidence_status": assessment.status,
                "accepted": assessment.accepted,
                "exclude_from_primary_accuracy": case[
                    "exclude_from_primary_accuracy"
                ],
                "crop_issue": case.get("crop_issue"),
                "top_candidates": [
                    {
                        "name": match.record.display_name,
                        "id": match.record.id,
                        "score": round(match.score, 4),
                    }
                    for match in matches
                ],
            }
        )

    scored = [
        row
        for row in rows
        if not row["exclude_from_primary_accuracy"]
    ]
    correct = sum(row["expected_rank"] == 1 for row in scored)

    return {
        "benchmark": "labelled_screenshot_icon_crops",
        "crops_dir": str(crops_dir),
        "template_count": len(knowledge.all_icons),
        "tower_template_count": len(knowledge.towers),
        "unit_template_count": len(knowledge.troops),
        "top_k": top_k,
        "primary_accuracy": {
            "correct": correct,
            "total": len(scored),
            "fraction": round(correct / len(scored), 4) if scored else None,
            "excluded_cases": [
                row["file"]
                for row in rows
                if row["exclude_from_primary_accuracy"]
            ],
        },
        "cases": rows,
        "limitations": [
            "The matcher currently resizes each complete crop to the template dimensions.",
            "Screenshot glow and background pixels are not segmented before comparison.",
            "Similarity scores are not calibrated probabilities.",
            "The King crop is clipped and excluded from primary accuracy.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark Kiomet icon matching on labelled screenshot crops."
    )
    parser.add_argument(
        "--crops-dir",
        type=Path,
        default=DEFAULT_CROPS_DIR,
        help="Directory containing the labelled PNG crops.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of ranked candidates to retain (default: 5).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSON report path.",
    )
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    report = run_benchmark(args.crops_dir, args.top_k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    accuracy = report["primary_accuracy"]
    print("KIOMET ICON CROP BENCHMARK")
    print(f"Templates: {report['template_count']} "
          f"({report['tower_template_count']} towers, "
          f"{report['unit_template_count']} units)")
    print(
        f"Primary top-1 accuracy: {accuracy['correct']}/{accuracy['total']}"
    )
    print()
    for row in report["cases"]:
        rank = (
            f"#{row['expected_rank']}"
            if row["expected_rank"] is not None
            else "not in top-k"
        )
        note = " [diagnostic only]" if row["exclude_from_primary_accuracy"] else ""
        print(
            f"{row['file']:<22} expected={row['expected']:<10} "
            f"rank={rank:<14} top={row['top_match']!s:<12} "
            f"score={row['top_score']:.3f} "
            f"status={row['confidence_status']}{note}"
        )
        for candidate in row["top_candidates"][:3]:
            print(f"    {candidate['name']:<14} {candidate['score']:.3f}")
    print(f"\nSaved report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
