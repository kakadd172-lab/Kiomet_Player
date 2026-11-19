"""Evaluate temporal entity tracks against manually linked multi-frame labels.

The evaluator is model-agnostic: replay records contain the detector/tracker
outputs for each frame, allowing repeatable evaluation without calling Ollama.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _box(value: dict[str, Any]) -> tuple[float, float, float, float]:
    if not isinstance(value, dict):
        raise ValueError("bbox must be an object")
    try:
        values = tuple(float(value[key]) for key in ("x", "y", "width", "height"))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("bbox must contain numeric x, y, width, and height") from error
    x, y, width, height = values
    if not all(value == value and abs(value) != float("inf") for value in values):
        raise ValueError("bbox values must be finite")
    if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > 1 or y + height > 1:
        raise ValueError("bbox must have positive size and fit inside normalized frame bounds")
    return values


def _iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def _match_frame(expected: list[dict[str, Any]], predicted: list[dict[str, Any]], threshold: float):
    edges = []
    for ei, truth in enumerate(expected):
        for pi, guess in enumerate(predicted):
            if truth["category"] != guess["category"]:
                continue
            overlap = _iou(_box(truth["bbox"]), _box(guess["bbox"]))
            if overlap >= threshold:
                edges.append((overlap, ei, pi))
    # Augmenting paths maximize the number of valid spatial matches; among
    # alternatives, each expected entity tries its highest-IoU prediction first.
    adjacency: dict[int, list[tuple[float, int]]] = {index: [] for index in range(len(expected))}
    for overlap, ei, pi in edges:
        adjacency[ei].append((overlap, pi))
    for candidates in adjacency.values():
        candidates.sort(key=lambda pair: (-pair[0], pair[1]))
    owner: dict[int, int] = {}
    chosen: dict[tuple[int, int], float] = {}

    def assign(ei: int, visited: set[int]) -> bool:
        for overlap, pi in adjacency[ei]:
            if pi in visited:
                continue
            visited.add(pi)
            previous = owner.get(pi)
            if previous is None or assign(previous, visited):
                owner[pi] = ei
                chosen[(ei, pi)] = overlap
                if previous is not None:
                    chosen.pop((previous, pi), None)
                return True
        return False

    for ei in sorted(adjacency, key=lambda index: (len(adjacency[index]), index)):
        assign(ei, set())
    return [(ei, pi, chosen[(ei, pi)]) for pi, ei in sorted(owner.items(), key=lambda item: item[1])]


def evaluate_tracking_frames(
    frames: list[dict[str, Any]],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Score linked ground-truth entities against tracked frame snapshots.

    Each frame has frame_index, expected[], and predicted[]. Expected entities
    require stable id/category/bbox; predictions require track_id/category/bbox.
    """
    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be greater than 0 and at most 1")
    if not frames:
        raise ValueError("at least one frame is required")

    seen_expected: set[str] = set()
    last_track: dict[str, str] = {}
    was_matched: dict[str, bool] = {}
    gap_since_match: dict[str, int] = {}
    id_switches = 0
    fragmentations = 0
    reacquisitions = 0
    total_expected = total_predicted = true_positives = 0
    matched_ious: list[float] = []
    per_frame = []

    previous_frame_index = None
    for frame_number, frame in enumerate(frames):
        frame_index = frame.get("frame_index", frame_number)
        if not isinstance(frame_index, int) or (previous_frame_index is not None and frame_index <= previous_frame_index):
            raise ValueError("frame_index values must be strictly increasing integers")
        previous_frame_index = frame_index
        expected = frame.get("expected")
        predicted = frame.get("predicted")
        if not isinstance(expected, list) or not isinstance(predicted, list):
            raise ValueError("each frame must contain expected and predicted lists")
        expected_ids: set[str] = set()
        for item in expected:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"]:
                raise ValueError("each expected entity requires a stable non-empty id")
            if item["id"] in expected_ids:
                raise ValueError("ground-truth IDs must be unique within each frame")
            expected_ids.add(item["id"])
            if item.get("category") not in {"tower", "unit"}:
                raise ValueError("expected category must be tower or unit")
            _box(item["bbox"])
        predicted_ids: set[str] = set()
        for item in predicted:
            if not isinstance(item, dict) or not isinstance(item.get("track_id"), str) or not item["track_id"]:
                raise ValueError("each prediction requires a non-empty track_id")
            if item["track_id"] in predicted_ids:
                raise ValueError("predicted track IDs must be unique within each frame")
            predicted_ids.add(item["track_id"])
            if item.get("category") not in {"tower", "unit"}:
                raise ValueError("predicted category must be tower or unit")
            _box(item["bbox"])

        matches = _match_frame(expected, predicted, iou_threshold)
        matched_expected = set()
        matched_predicted = set()
        frame_switches = frame_fragments = 0
        for ei, pi, overlap in matches:
            truth, guess = expected[ei], predicted[pi]
            entity_id, track_id = truth["id"], guess["track_id"]
            matched_expected.add(ei)
            matched_predicted.add(pi)
            matched_ious.append(overlap)
            if entity_id in last_track and last_track[entity_id] != track_id:
                id_switches += 1
                frame_switches += 1
            if entity_id in seen_expected and not was_matched.get(entity_id, False):
                fragmentations += 1
                frame_fragments += 1
                if gap_since_match.get(entity_id, 0) > 0 and last_track.get(entity_id) == track_id:
                    reacquisitions += 1
            last_track[entity_id] = track_id
            seen_expected.add(entity_id)
            was_matched[entity_id] = True
            gap_since_match[entity_id] = 0

        for ei, truth in enumerate(expected):
            entity_id = truth["id"]
            if ei not in matched_expected:
                seen_expected.add(entity_id)
                was_matched[entity_id] = False
                gap_since_match[entity_id] = gap_since_match.get(entity_id, 0) + 1

        tp = len(matches)
        fp = len(predicted) - len(matched_predicted)
        fn = len(expected) - len(matched_expected)
        total_expected += len(expected)
        total_predicted += len(predicted)
        true_positives += tp
        per_frame.append({
            "frame_index": frame_index,
            "expected": len(expected),
            "predicted": len(predicted),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "id_switches": frame_switches,
            "fragmentations": frame_fragments,
        })

    fp = total_predicted - true_positives
    fn = total_expected - true_positives
    precision = true_positives / (true_positives + fp) if true_positives + fp else None
    recall = true_positives / (true_positives + fn) if true_positives + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    return {
        "frames_evaluated": len(frames),
        "expected_detections": total_expected,
        "predicted_detections": total_predicted,
        "true_positives": true_positives,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
        "f1": round(f1, 4) if f1 is not None else None,
        "mean_matched_iou": round(sum(matched_ious) / len(matched_ious), 4) if matched_ious else None,
        "id_switches": id_switches,
        "fragmentations": fragmentations,
        "reacquisitions": reacquisitions,
        "unique_ground_truth_entities": len(seen_expected),
        "per_frame": per_frame,
        "limitations": [
            "Identity metrics require stable ground-truth IDs across every frame in a sequence.",
            "This evaluator measures supplied replay records; it does not itself run the vision model.",
            "A track ID is not proof of real-world identity when detections are ambiguous or occluded.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSON replay records with expected and predicted entities per frame")
    parser.add_argument("--output", type=Path, help="Optional path for the JSON metrics report")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()
    document = json.loads(args.input.read_text(encoding="utf-8"))
    frames = document.get("frames") if isinstance(document, dict) else None
    if not isinstance(frames, list):
        raise ValueError("input JSON must contain a frames array")
    report = {
        "benchmark": "kiomet_temporal_tracking",
        "schema_version": 1,
        "iou_threshold": args.iou_threshold,
        "summary": evaluate_tracking_frames(frames, iou_threshold=args.iou_threshold),
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
