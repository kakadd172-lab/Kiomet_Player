"""Benchmark the deterministic icon matcher against labelled screenshot crops.

Run from the repository root:
    python perception/icon_matcher_benchmark.py
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kiomet_knowledge.recognition.knowledge_loader import load_knowledge
from kiomet_knowledge.recognition.shared.confidence import assess_matches
from kiomet_knowledge.recognition.shared.image_matcher import ImageMatcher


DEFAULT_LABELS = ROOT / "benchmark_crops_v2" / "labels.json"
DEFAULT_OUTPUT = ROOT / "benchmark_results" / "icon_matcher_benchmark.json"


def run_benchmark(labels_path: Path, output_path: Path, top_k: int = 5) -> dict:
    document = json.loads(labels_path.read_text(encoding="utf-8"))
    cases = document.get("cases")
    if not isinstance(cases, list):
        raise ValueError("Benchmark labels must contain a 'cases' list")

    knowledge = load_knowledge()
    matcher = ImageMatcher(knowledge.all_icons)
    results = []
    skipped = []

    for case in cases:
        name = case["name"]
        if not case.get("enabled", True):
            skipped.append({
                "name": name,
                "reason": case.get("skip_reason", "Disabled in benchmark labels"),
            })
            continue

        category = case["category"]
        expected_name = case["expected_name"]
        image_path = labels_path.parent / case["image"]
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing benchmark crop for {name}: {image_path}")

        expected = knowledge.get_icon(category, expected_name)
        if expected is None:
            raise ValueError(
                f"Unknown expected icon {expected_name!r} in category {category!r}"
            )

        ranked = matcher.match(image_path, category=category, top_k=top_k)
        assessment = assess_matches(ranked)
        rank = next(
            (index for index, item in enumerate(ranked, start=1)
             if item.record.id == expected.id),
            None,
        )

        results.append({
            "sample": name,
            "image": str(image_path.relative_to(ROOT)).replace("\\", "/"),
            "category": category,
            "expected_id": expected.id,
            "expected_name": expected.display_name,
            "expected_rank": rank,
            "top1_correct": bool(ranked and ranked[0].record.id == expected.id),
            "top_k_contains_expected": rank is not None,
            "confidence": {
                "status": assessment.status,
                "accepted": assessment.accepted,
                "top_score": round(assessment.top_score, 4),
                "margin": (
                    round(assessment.margin, 4)
                    if assessment.margin is not None else None
                ),
            },
            "candidates": [
                {
                    "id": item.record.id,
                    "name": item.record.display_name,
                    "score": round(item.score, 4),
                }
                for item in ranked
            ],
        })

    total = len(results)
    summary = {
        "samples_evaluated": total,
        "samples_skipped": len(skipped),
        "top1_correct": sum(item["top1_correct"] for item in results),
        "top1_accuracy": (
            round(sum(item["top1_correct"] for item in results) / total, 4)
            if total else None
        ),
        "expected_in_top_k": sum(item["top_k_contains_expected"] for item in results),
        "confidence_accepted": sum(item["confidence"]["accepted"] for item in results),
    }

    report = {
        "benchmark": "kiomet_icon_matcher",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "matcher": "ImageMatcher grayscale masked absolute-difference baseline",
        "top_k": top_k,
        "summary": summary,
        "results": results,
        "skipped": skipped,
        "limitations": [
            "This is a small, manually labelled crop set, not a representative accuracy estimate.",
            "The matcher resizes each entire crop to the template dimensions; it does not isolate icon pixels from glow, territory fill, or nearby UI.",
            "Similarity scores are not calibrated probabilities.",
            "King is excluded because its current crop is visibly clipped.",
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("KIOMET ICON MATCHER BENCHMARK")
    print(f"Samples evaluated: {summary['samples_evaluated']}")
    print(f"Samples skipped:   {summary['samples_skipped']}")
    print(f"Top-1 correct:     {summary['top1_correct']}/{total}")
    print(f"Top-1 accuracy:    {summary['top1_accuracy']}")
    print(f"Expected in top-{top_k}: {summary['expected_in_top_k']}/{total}")
    print(f"Confidence accepted: {summary['confidence_accepted']}/{total}")
    print()
    for item in results:
        top = item["candidates"][0] if item["candidates"] else None
        top_text = (
            f"{top['name']} ({top['score']:.3f})" if top else "no candidates"
        )
        print(
            f"{item['sample']:<18} expected={item['expected_name']:<10} "
            f"rank={item['expected_rank']} top={top_text} "
            f"status={item['confidence']['status']}"
        )
    for item in skipped:
        print(f"SKIPPED {item['name']}: {item['reason']}")
    print(f"\nSaved report: {output_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    run_benchmark(args.labels, args.output, args.top_k)


if __name__ == "__main__":
    main()
