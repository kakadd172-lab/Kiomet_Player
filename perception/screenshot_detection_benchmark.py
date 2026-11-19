"""Evaluate full-screen Kiomet entity proposals against exhaustive manual labels.

Labels use normalized [0, 1] bounding boxes relative to each source screenshot.
Only screenshots marked fully_annotated=true are accepted: otherwise unmatched
predictions cannot safely be counted as false positives.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perception.vision_entity_detector import VisionEntityDetector

DEFAULT_LABELS = ROOT / "benchmark_screenshots" / "labels.json"
DEFAULT_OUTPUT = ROOT / "benchmark_results" / "screenshot_detection_benchmark.json"
VALID_CATEGORIES = {"tower", "unit"}


def _valid_box(box: Any, context: str) -> tuple[float, float, float, float]:
    if not isinstance(box, dict):
        raise ValueError(f"{context} must be an object with x, y, width, and height")
    values = []
    for key in ("x", "y", "width", "height"):
        value = box.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{context}.{key} must be a finite number")
        number = float(value)
        if not (number == number and abs(number) != float("inf")):
            raise ValueError(f"{context}.{key} must be a finite number")
        values.append(number)
    x, y, width, height = values
    if width <= 0 or height <= 0 or x < 0 or y < 0 or x + width > 1 or y + height > 1:
        raise ValueError(f"{context} must have positive size and fit within normalized screenshot bounds")
    return x, y, width, height


def load_labels(labels_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read labels and reject ambiguous/incomplete benchmark cases."""
    try:
        document = json.loads(labels_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in labels file {labels_path}: {error}") from error
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError("Labels must be an object with schema_version=1")
    screenshots = document.get("screenshots")
    if not isinstance(screenshots, list) or not screenshots:
        raise ValueError("Labels must contain a non-empty screenshots list")

    seen: set[str] = set()
    validated: list[dict[str, Any]] = []
    for index, item in enumerate(screenshots):
        prefix = f"screenshots[{index}]"
        if not isinstance(item, dict):
            raise ValueError(f"{prefix} must be an object")
        name = item.get("name")
        if not isinstance(name, str) or not name.strip() or name in seen:
            raise ValueError(f"{prefix}.name must be non-empty and unique")
        seen.add(name)
        if item.get("fully_annotated") is not True:
            raise ValueError(f"{prefix}.fully_annotated must be true before scoring precision/recall")
        image = item.get("image")
        if not isinstance(image, str) or not image.strip():
            raise ValueError(f"{prefix}.image must be a relative path")
        image_path = (labels_path.parent / image).resolve()
        if not image_path.is_relative_to(labels_path.parent.resolve()):
            raise ValueError(f"{prefix}.image must stay inside the labels directory")
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing screenshot for {name}: {image_path}")
        entities = item.get("entities")
        if not isinstance(entities, list):
            raise ValueError(f"{prefix}.entities must be a list (use [] for a verified empty screen)")
        entity_ids: set[str] = set()
        for entity_index, entity in enumerate(entities):
            entity_prefix = f"{prefix}.entities[{entity_index}]"
            if not isinstance(entity, dict):
                raise ValueError(f"{entity_prefix} must be an object")
            entity_id = entity.get("id")
            if not isinstance(entity_id, str) or not entity_id.strip() or entity_id in entity_ids:
                raise ValueError(f"{entity_prefix}.id must be non-empty and unique within the screenshot")
            entity_ids.add(entity_id)
            if entity.get("category") not in VALID_CATEGORIES:
                raise ValueError(f"{entity_prefix}.category must be 'tower' or 'unit'")
            _valid_box(entity.get("bbox"), f"{entity_prefix}.bbox")
            expected_name = entity.get("expected_name")
            if expected_name is not None and (not isinstance(expected_name, str) or not expected_name.strip()):
                raise ValueError(f"{entity_prefix}.expected_name must be a non-empty string when supplied")
        validated.append({**item, "_image_path": image_path})
    return document, validated


def box_iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return min(1.0, max(0.0, intersection / union)) if union > 0 else 0.0


def match_entities(
    expected: list[dict[str, Any]],
    predicted: list[dict[str, Any]],
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Greedy one-to-one matching, highest IoU first, requiring category match."""
    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be greater than 0 and at most 1")
    # Use augmenting paths rather than a single greedy pass. A locally best
    # IoU edge can otherwise block a second valid match and inflate false
    # negatives even though a complete one-to-one assignment exists.
    adjacency: list[list[tuple[float, int]]] = []
    for expected_index, truth in enumerate(expected):
        truth_box = _valid_box(truth["bbox"], f"expected[{expected_index}].bbox")
        edges = []
        for predicted_index, candidate in enumerate(predicted):
            if truth["category"] != candidate["category"]:
                continue
            candidate_box = _valid_box(candidate["bbox"], f"predicted[{predicted_index}].bbox")
            overlap = box_iou(truth_box, candidate_box)
            if overlap >= iou_threshold:
                edges.append((overlap, predicted_index))
        edges.sort(key=lambda pair: (-pair[0], pair[1]))
        adjacency.append(edges)

    # Process the most constrained labels first. Reassignment guarantees
    # maximum match count while preferring higher-IoU edges.
    expected_order = sorted(range(len(expected)), key=lambda index: (len(adjacency[index]), index))
    predicted_owner: dict[int, int] = {}
    chosen_iou: dict[tuple[int, int], float] = {}

    def assign(expected_index: int, visited: set[int]) -> bool:
        for overlap, predicted_index in adjacency[expected_index]:
            if predicted_index in visited:
                continue
            visited.add(predicted_index)
            previous_owner = predicted_owner.get(predicted_index)
            if previous_owner is None or assign(previous_owner, visited):
                predicted_owner[predicted_index] = expected_index
                chosen_iou[(expected_index, predicted_index)] = overlap
                if previous_owner is not None:
                    chosen_iou.pop((previous_owner, predicted_index), None)
                return True
        return False

    for expected_index in expected_order:
        assign(expected_index, set())

    matches = []
    used_expected = set(predicted_owner.values())
    used_predicted = set(predicted_owner)
    for predicted_index, expected_index in sorted(predicted_owner.items(), key=lambda pair: pair[1]):
        overlap = chosen_iou[(expected_index, predicted_index)]
        truth = expected[expected_index]
        guess = predicted[predicted_index]
        expected_name = truth.get("expected_name")
        observed_name = guess.get("recognized_name")
        matches.append({
            "expected_id": truth["id"],
            "predicted_id": guess.get("id"),
            "category": truth["category"],
            "iou": round(overlap, 4),
            "expected_name": expected_name,
            "recognized_name": observed_name,
            "identity_scored": expected_name is not None,
            "identity_correct": (
                _normalize_name(expected_name) == _normalize_name(observed_name)
                if expected_name is not None and observed_name is not None else False
            ) if expected_name is not None else None,
        })
    false_positives = [i for i in range(len(predicted)) if i not in used_predicted]
    missed = [i for i in range(len(expected)) if i not in used_expected]
    return {
        "matches": matches,
        "false_positive_indices": false_positives,
        "missed_expected_indices": missed,
        "true_positives": len(matches),
        "false_positives": len(false_positives),
        "false_negatives": len(missed),
    }


def _normalize_name(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold()) if isinstance(value, str) else ""


def _category_metrics_from_totals(
    totals_by_category: dict[str, dict[str, int]],
) -> dict[str, dict[str, int | float | None]]:
    """Add precision, recall and F1 to accumulated per-category counts."""
    return {
        category: {
            **totals,
            **_rates(
                totals["true_positives"],
                totals["false_positives"],
                totals["false_negatives"],
            ),
        }
        for category, totals in totals_by_category.items()
    }


def _category_metrics(
    expected: list[dict[str, Any]],
    predicted: list[dict[str, Any]],
    iou_threshold: float,
) -> dict[str, dict[str, int | float | None]]:
    """Calculate independent detection metrics for towers and units."""
    result = {}
    for category in sorted(VALID_CATEGORIES):
        category_expected = [item for item in expected if item["category"] == category]
        category_predicted = [item for item in predicted if item["category"] == category]
        matching = match_entities(category_expected, category_predicted, iou_threshold)
        tp = matching["true_positives"]
        fp = matching["false_positives"]
        fn = matching["false_negatives"]
        result[category] = {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            **_rates(tp, fp, fn),
        }
    return result


def _rates(tp: int, fp: int, fn: int) -> dict[str, float | None]:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    return {
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
        "f1": round(f1, 4) if f1 is not None else None,
    }


def run_benchmark(
    labels_path: Path,
    output_path: Path,
    *,
    model: str = "qwen2.5vl:3b",
    method: str = "shape",
    max_image_side: int | None = 960,
    tiled: bool = False,
    iou_threshold: float = 0.5,
    min_score: float = 0.65,
    min_margin: float = 0.10,
    timeout: float = 180.0,
    num_threads: int = 8,
    upscale: float = 1.0,
    overlap: float = 0.12,
    min_detector_confidence: float = 0.35,
) -> dict[str, Any]:
    document, cases = load_labels(labels_path)
    detector = VisionEntityDetector(
        model=model,
        max_image_side=max_image_side,
        tiled=tiled,
        timeout=timeout,
        num_threads=num_threads,
        upscale=upscale,
        overlap=overlap,
        min_detector_confidence=min_detector_confidence,
    )
    per_image = []
    total_tp = total_fp = total_fn = 0
    category_totals = {
        category: {"true_positives": 0, "false_positives": 0, "false_negatives": 0}
        for category in sorted(VALID_CATEGORIES)
    }
    identity_scored = identity_correct = 0

    for case in cases:
        image_path = case["_image_path"]
        detected = detector.detect(
            image_path,
            method=method,
            min_score=min_score,
            min_margin=min_margin,
        )
        predictions = [{
            "id": item.proposal.region_id,
            "category": item.proposal.category,
            "bbox": {
                "x": item.proposal.x,
                "y": item.proposal.y,
                "width": item.proposal.width,
                "height": item.proposal.height,
            },
            "recognized_name": item.recognition.display_name if item.recognition.accepted else None,
            "recognition_status": item.recognition.status,
            "recognition_score": item.recognition.top_score,
            "recognition_margin": item.recognition.margin,
            "detector_confidence": item.detector_confidence,
        } for item in detected]
        expected = case["entities"]
        matching = match_entities(expected, predictions, iou_threshold)
        tp, fp, fn = matching["true_positives"], matching["false_positives"], matching["false_negatives"]
        total_tp += tp
        total_fp += fp
        total_fn += fn
        for category, category_result in _category_metrics(expected, predictions, iou_threshold).items():
            totals = category_totals[category]
            totals["true_positives"] += category_result["true_positives"]
            totals["false_positives"] += category_result["false_positives"]
            totals["false_negatives"] += category_result["false_negatives"]
        for match in matching["matches"]:
            if match["identity_scored"]:
                identity_scored += 1
                identity_correct += int(bool(match["identity_correct"]))
        image_bytes = image_path.read_bytes()
        per_image.append({
            "name": case["name"],
            "image": case["image"],
            "source_file_bytes": len(image_bytes),
            "source_file_sha256": hashlib.sha256(image_bytes).hexdigest(),
            "expected_count": len(expected),
            "predicted_count": len(predictions),
            "metrics": _rates(tp, fp, fn),
            "by_category": _category_metrics(expected, predictions, iou_threshold),
            **matching,
            "predictions": predictions,
            "pipeline_metrics": detector.pipeline_metrics,
            "inference_metrics": detector.inference_metrics,
        })

    report = {
        "benchmark": "kiomet_full_screenshot_detection",
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "label_source": document.get("label_source"),
        "model": model,
        "configuration": {
            "method": method,
            "max_image_side": max_image_side,
            "tiled": tiled,
            "iou_threshold": iou_threshold,
            "min_score": min_score,
            "min_margin": min_margin,
            "timeout_seconds": timeout,
            "num_threads": num_threads,
            "upscale": upscale,
            "tile_overlap": overlap,
            "min_detector_confidence": min_detector_confidence,
        },
        "summary": {
            "screenshots_evaluated": len(per_image),
            "expected_entities": total_tp + total_fn,
            "predicted_entities": total_tp + total_fp,
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
            **_rates(total_tp, total_fp, total_fn),
            "by_category": _category_metrics_from_totals(category_totals),
            "identity_scored_matches": identity_scored,
            "identity_correct": identity_correct,
            "identity_accuracy": round(identity_correct / identity_scored, 4) if identity_scored else None,
        },
        "screenshots": per_image,
        "limitations": [
            "Metrics are meaningful only when every visible tower/unit in each screenshot is labelled.",
            "Bounding-box matching uses category agreement and the configured IoU threshold.",
            "Identity accuracy is computed only for spatially matched entities with an expected_name label.",
            "A small or unbalanced screenshot set is not a general estimate of gameplay accuracy.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = report["summary"]
    print("KIOMET FULL-SCREEN DETECTION BENCHMARK")
    print(f"Screenshots:       {summary['screenshots_evaluated']}")
    print(f"Expected entities: {summary['expected_entities']}")
    print(f"Predicted:         {summary['predicted_entities']}")
    print(f"TP / FP / FN:      {total_tp} / {total_fp} / {total_fn}")
    print(f"Precision:         {summary['precision']}")
    print(f"Recall:            {summary['recall']}")
    print(f"F1:                {summary['f1']}")
    print(f"Identity accuracy: {summary['identity_accuracy']} ({identity_correct}/{identity_scored})")
    print(f"Report: {output_path}")
    return report


def _parse_max_image_side(value: str) -> int | None:
    if value.casefold() in {"native", "none", "off"}:
        return None
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer >=256 or 'native'") from error
    if parsed < 256:
        raise argparse.ArgumentTypeError("must be at least 256 or 'native'")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default="qwen2.5vl:3b")
    parser.add_argument("--method", choices=("shape", "grayscale"), default="shape")
    parser.add_argument(
        "--max-image-side",
        type=_parse_max_image_side,
        default=960,
        metavar="PIXELS|native",
        help="maximum input side in pixels (any integer >=256), or 'native' to preserve source resolution",
    )
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--min-score", type=float, default=0.65)
    parser.add_argument("--min-margin", type=float, default=0.10)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--threads", type=int, default=8, help="Ollama CPU inference threads")
    parser.add_argument("--upscale", type=float, default=1.0, help="Optional image scale factor, capped by max-image-side")
    parser.add_argument("--overlap", type=float, default=0.12, help="Tile overlap fraction when --tiled is enabled")
    parser.add_argument("--min-detector-confidence", type=float, default=0.35)
    args = parser.parse_args()
    if args.max_image_side is not None and args.max_image_side < 256:
        parser.error("--max-image-side must be 'native' or at least 256")
    for option, value in (("--iou-threshold", args.iou_threshold), ("--min-score", args.min_score), ("--min-margin", args.min_margin)):
        if not 0 <= value <= 1 or (option == "--iou-threshold" and value == 0):
            parser.error(f"{option} is outside its valid range")
    if args.timeout <= 0 or args.threads < 1 or args.upscale < 0.25:
        parser.error("--timeout and --threads must be positive; --upscale must be at least 0.25")
    if not 0 <= args.overlap < 0.5:
        parser.error("--overlap must be between 0 and 0.5")
    if not 0 <= args.min_detector_confidence <= 1:
        parser.error("--min-detector-confidence must be between 0 and 1")
    run_benchmark(
        args.labels, args.output, model=args.model, method=args.method,
        max_image_side=args.max_image_side, tiled=args.tiled,
        iou_threshold=args.iou_threshold, min_score=args.min_score,
        min_margin=args.min_margin, timeout=args.timeout,
        num_threads=args.threads, upscale=args.upscale, overlap=args.overlap,
        min_detector_confidence=args.min_detector_confidence,
    )


if __name__ == "__main__":
    main()
