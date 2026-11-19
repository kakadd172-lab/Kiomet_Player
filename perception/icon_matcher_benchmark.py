"""Benchmark deterministic icon matching against manually labelled crops.

Run from the repository root:
    python perception/icon_matcher_benchmark.py
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kiomet_knowledge.recognition.knowledge_loader import load_knowledge
from kiomet_knowledge.recognition.shared.confidence import assess_matches
from kiomet_knowledge.recognition.shared.image_matcher import ImageMatcher


DEFAULT_LABELS = ROOT / "benchmark_crops_v2" / "labels.json"
DEFAULT_OUTPUT = ROOT / "benchmark_results" / "icon_matcher_benchmark.json"
VALID_CATEGORIES = {"tower": "towers", "unit": "troops"}


def load_labels(labels_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load and validate the benchmark manifest before reading any image."""
    try:
        document = json.loads(labels_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in labels file {labels_path}: {error}") from error

    if not isinstance(document, dict):
        raise ValueError("Benchmark labels must be a JSON object")
    if document.get("schema_version") != 1:
        raise ValueError("Unsupported benchmark labels schema_version; expected 1")

    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Benchmark labels must contain a non-empty 'cases' list")

    seen_names: set[str] = set()
    validated: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        prefix = f"cases[{index}]"
        if not isinstance(case, dict):
            raise ValueError(f"{prefix} must be an object")

        name = case.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{prefix}.name must be a non-empty string")
        if name in seen_names:
            raise ValueError(f"Duplicate benchmark case name: {name!r}")
        seen_names.add(name)

        enabled = case.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError(f"{prefix}.enabled must be a boolean")
        if not enabled:
            validated.append(case)
            continue

        category = case.get("category")
        if category not in VALID_CATEGORIES:
            raise ValueError(
                f"{prefix}.category must be one of: {', '.join(VALID_CATEGORIES)}"
            )
        expected_name = case.get("expected_name")
        if not isinstance(expected_name, str) or not expected_name.strip():
            raise ValueError(f"{prefix}.expected_name must be a non-empty string")

        image = case.get("image")
        if not isinstance(image, str) or not image.strip():
            raise ValueError(f"{prefix}.image must be a relative file path")
        image_path = (labels_path.parent / image).resolve()
        labels_dir = labels_path.parent.resolve()
        if not image_path.is_relative_to(labels_dir):
            raise ValueError(f"{prefix}.image must stay inside {labels_dir}")
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing benchmark crop for {name}: {image_path}")

        validated.append(case)

    return document, validated


def run_benchmark(
    labels_path: Path,
    output_path: Path,
    top_k: int = 5,
    min_score: float = 0.65,
    min_margin: float = 0.10,
    method: str = "grayscale",
) -> dict[str, Any]:
    """Run the labelled-crop benchmark and save a machine-readable report."""
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if method not in {"grayscale", "shape"}:
        raise ValueError("method must be 'grayscale' or 'shape'")

    document, cases = load_labels(labels_path)
    knowledge = load_knowledge()
    matcher = ImageMatcher(knowledge.all_icons)
    results: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for case in cases:
        name = case["name"]
        if not case.get("enabled", True):
            skipped.append({
                "name": name,
                "reason": case.get("skip_reason", "Disabled in benchmark labels"),
            })
            continue

        category = case["category"]
        expected = knowledge.get_icon(category, case["expected_name"])
        if expected is None:
            raise ValueError(
                f"Unknown expected icon {case['expected_name']!r} "
                f"in category {category!r} for case {name!r}"
            )

        image_path = (labels_path.parent / case["image"]).resolve()
        category_records = (
            knowledge.towers if category == "tower" else knowledge.troops
        )
        # Rank against the complete category catalogue. top_k controls only
        # how many candidates are included in the report, not confidence.
        all_ranked = matcher.match(
            image_path,
            category=category,
            top_k=len(category_records),
            method=method,
        )
        displayed = all_ranked[:top_k]
        assessment = assess_matches(
            all_ranked, min_score=min_score, min_margin=min_margin
        )
        expected_rank = next(
            (
                rank for rank, item in enumerate(all_ranked, start=1)
                if item.record.id == expected.id
            ),
            None,
        )
        top = all_ranked[0] if all_ranked else None

        results.append({
            "sample": name,
            "image": image_path.relative_to(ROOT).as_posix(),
            "category": category,
            "expected_id": expected.id,
            "expected_name": expected.display_name,
            "expected_rank": expected_rank,
            "top1_id": top.record.id if top else None,
            "top1_name": top.record.display_name if top else None,
            "top1_correct": bool(top and top.record.id == expected.id),
            "expected_in_top_k": expected_rank is not None and expected_rank <= top_k,
            "confidence": {
                "status": assessment.status,
                "accepted": assessment.accepted,
                "top_score": round(assessment.top_score, 4),
                "margin": round(assessment.margin, 4)
                if assessment.margin is not None else None,
                "thresholds": {
                    "min_score": min_score,
                    "min_margin": min_margin,
                },
            },
            "candidates": [
                {
                    "id": item.record.id,
                    "name": item.record.display_name,
                    "score": round(item.score, 4),
                }
                for item in displayed
            ],
        })

    total = len(results)
    correct = sum(item["top1_correct"] for item in results)
    accepted = sum(item["confidence"]["accepted"] for item in results)
    summary = {
        "samples_evaluated": total,
        "samples_skipped": len(skipped),
        "top1_correct": correct,
        "top1_accuracy": round(correct / total, 4) if total else None,
        "expected_in_top_k": sum(item["expected_in_top_k"] for item in results),
        "confidence_accepted": accepted,
        "accepted_and_correct": sum(
            item["confidence"]["accepted"] and item["top1_correct"]
            for item in results
        ),
    }

    report = {
        "benchmark": "kiomet_icon_matcher",
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_screenshot": document.get("source_screenshot"),
        "label_source": document.get("label_source"),
        "matcher": (
            "ImageMatcher aspect-normalized binary shape comparison"
            if method == "shape"
            else "ImageMatcher grayscale comparison with template-alpha masking"
        ),
        "matcher_method": method,
        "top_k": top_k,
        "summary": summary,
        "results": results,
        "skipped": skipped,
        "limitations": [
            "This is a small, manually labelled crop set, not a representative accuracy estimate.",
            "The matcher resizes each entire crop to template dimensions; it does not isolate the icon from territory fill, glow, or nearby UI.",
            "Similarity scores are not calibrated probabilities; confidence thresholds are provisional heuristics.",
            "King is excluded because its current crop is visibly clipped.",
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("KIOMET ICON MATCHER BENCHMARK")
    print(f"Matcher method:          {method}")
    print(f"Samples evaluated:       {total}")
    print(f"Samples skipped:         {len(skipped)}")
    print(f"Top-1 correct:           {correct}/{total}")
    print(f"Top-1 accuracy:          {summary['top1_accuracy']}")
    print(f"Expected in top-{top_k}:        {summary['expected_in_top_k']}/{total}")
    print(f"Confidence accepted:     {accepted}/{total}")
    print(f"Accepted and correct:    {summary['accepted_and_correct']}/{total}")
    print()
    for item in results:
        print(
            f"{item['sample']:<18} expected={item['expected_name']:<10} "
            f"rank={item['expected_rank']} top={item['top1_name']} "
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
    parser.add_argument(
        "--method",
        choices=("grayscale", "shape"),
        default="grayscale",
        help="Matching method to benchmark (shape is experimental).",
    )
    parser.add_argument("--min-score", type=float, default=0.65)
    parser.add_argument("--min-margin", type=float, default=0.10)
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    for option, value in (
        ("--min-score", args.min_score),
        ("--min-margin", args.min_margin),
    ):
        if not 0.0 <= value <= 1.0:
            parser.error(f"{option} must be between 0 and 1")

    run_benchmark(
        args.labels,
        args.output,
        args.top_k,
        args.min_score,
        args.min_margin,
        args.method,
    )


if __name__ == "__main__":
    main()
