"""Compare two tracking replay reports over the exact same screenshot sequence.

This is an unlabeled engineering comparison. Track persistence is a diagnostic
proxy, not ground-truth identity accuracy.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _load_replay(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read replay JSON {path}: {error}") from error
    if not isinstance(data, dict) or not isinstance(data.get("frames"), list):
        raise ValueError(f"{path} is not a tracking replay report")
    if not data["frames"]:
        raise ValueError(f"{path} contains no frames")
    for index, frame in enumerate(data["frames"], start=1):
        if not isinstance(frame, dict) or not isinstance(frame.get("source_sha256"), str):
            raise ValueError(f"{path}: frame {index} has no source SHA256")
        if not isinstance(frame.get("predicted"), list):
            raise ValueError(f"{path}: frame {index} has no predicted list")
        for prediction in frame["predicted"]:
            if not isinstance(prediction, dict) or not isinstance(prediction.get("track_id"), str):
                raise ValueError(f"{path}: frame {index} has a prediction without track_id")
    return data


def _sequence_signature(report: dict[str, Any]) -> list[tuple[Any, str]]:
    return [
        (frame.get("frame_index"), frame["source_sha256"])
        for frame in report["frames"]
    ]


def _summarize(report: dict[str, Any]) -> dict[str, Any]:
    frames = report["frames"]
    seen_tracks: set[str] = set()
    track_frame_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    identity_counts: Counter[str] = Counter()
    frame_summaries = []
    total_detections = 0
    repeated_observations = 0
    new_track_observations = 0

    for frame in frames:
        predictions = frame["predicted"]
        frame_ids = [item["track_id"] for item in predictions]
        if len(frame_ids) != len(set(frame_ids)):
            raise ValueError("A replay frame contains duplicate track IDs")
        new_ids = [track_id for track_id in frame_ids if track_id not in seen_tracks]
        repeated_ids = [track_id for track_id in frame_ids if track_id in seen_tracks]
        seen_tracks.update(frame_ids)
        track_frame_counts.update(frame_ids)
        category_counts.update(item.get("category", "unknown") for item in predictions)
        identity_counts.update(item.get("identity") or "unidentified" for item in predictions)
        total_detections += len(predictions)
        repeated_observations += len(repeated_ids)
        new_track_observations += len(new_ids)
        frame_summaries.append({
            "frame_index": frame.get("frame_index"),
            "source": frame.get("source"),
            "detections": len(predictions),
            "new_track_ids": len(new_ids),
            "previously_seen_track_ids": len(repeated_ids),
            "temporarily_missing_tracks": len(frame.get("temporarily_missing_tracks", [])),
            "processing_seconds": frame.get("processing_seconds"),
        })

    frame_durations = [
        frame["processing_seconds"]
        for frame in frames
        if isinstance(frame.get("processing_seconds"), (int, float))
    ]
    track_lengths = list(track_frame_counts.values())
    return {
        "configuration": report.get("detector", {}),
        "frames": len(frames),
        "total_detections": total_detections,
        "mean_detections_per_frame": round(total_detections / len(frames), 4),
        "unique_track_ids": len(track_frame_counts),
        "mean_observed_frames_per_track": round(sum(track_lengths) / len(track_lengths), 4) if track_lengths else 0,
        "longest_observed_track_frames": max(track_lengths, default=0),
        "track_persistence_ratio": round(repeated_observations / total_detections, 4) if total_detections else None,
        "new_track_observations": new_track_observations,
        "category_observations": dict(sorted(category_counts.items())),
        "recognized_identity_observations": dict(sorted(identity_counts.items())),
        "runtime": report.get("runtime", {}),
        "measured_frame_runtime": {
            "count": len(frame_durations),
            "mean_seconds": round(sum(frame_durations) / len(frame_durations), 6) if frame_durations else None,
            "max_seconds": round(max(frame_durations), 6) if frame_durations else None,
        },
        "per_frame": frame_summaries,
    }


def compare_replays(first_path: Path, second_path: Path) -> dict[str, Any]:
    """Compare reports only when every input frame is byte-identical and ordered alike."""
    first = _load_replay(first_path)
    second = _load_replay(second_path)
    first_signature = _sequence_signature(first)
    second_signature = _sequence_signature(second)
    if first_signature != second_signature:
        raise ValueError(
            "Replay inputs differ: frame order, frame indices, or source SHA256 hashes do not match"
        )
    first_summary = _summarize(first)
    second_summary = _summarize(second)
    first_config = first_summary["configuration"]
    second_config = second_summary["configuration"]
    config_differences = {
        key: {"first": first_config.get(key), "second": second_config.get(key)}
        for key in sorted(set(first_config) | set(second_config))
        if first_config.get(key) != second_config.get(key)
    }
    first_runtime = first_summary["runtime"].get("total_seconds")
    second_runtime = second_summary["runtime"].get("total_seconds")
    runtime_change = None
    if isinstance(first_runtime, (int, float)) and isinstance(second_runtime, (int, float)) and first_runtime:
        runtime_change = round((second_runtime - first_runtime) / first_runtime * 100, 2)
    per_frame_comparison = []
    for left, right in zip(first_summary["per_frame"], second_summary["per_frame"]):
        per_frame_comparison.append({
            "frame_index": left["frame_index"],
            "first_detections": left["detections"],
            "second_detections": right["detections"],
            "detection_delta": right["detections"] - left["detections"],
            "first_processing_seconds": left["processing_seconds"],
            "second_processing_seconds": right["processing_seconds"],
        })
    return {
        "benchmark": "kiomet_tracking_replay_comparison",
        "schema_version": 1,
        "same_source_sequence": True,
        "source_frames": len(first_signature),
        "configuration_differences": config_differences,
        "first": first_summary,
        "second": second_summary,
        "delta_second_minus_first": {
            "total_detections": second_summary["total_detections"] - first_summary["total_detections"],
            "unique_track_ids": second_summary["unique_track_ids"] - first_summary["unique_track_ids"],
            "mean_detections_per_frame": round(
                second_summary["mean_detections_per_frame"] - first_summary["mean_detections_per_frame"], 4
            ),
            "track_persistence_ratio": (
                round(second_summary["track_persistence_ratio"] - first_summary["track_persistence_ratio"], 4)
                if first_summary["track_persistence_ratio"] is not None
                and second_summary["track_persistence_ratio"] is not None else None
            ),
            "runtime_percent_change": runtime_change,
        },
        "per_frame_comparison": per_frame_comparison,
        "limitations": [
            "Track persistence is not ground-truth identity accuracy; an incorrect association can persist across frames.",
            "Detection counts do not establish precision or recall without exhaustive manually linked labels.",
            "Runtime comparisons are meaningful only on the same machine under comparable system load and warm-up conditions.",
            "Compare one changed configuration at a time where possible.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True, help="Baseline replay JSON")
    parser.add_argument("--second", type=Path, required=True, help="Candidate replay JSON")
    parser.add_argument("--output", type=Path, required=True, help="Comparison report JSON")
    args = parser.parse_args()
    report = compare_replays(args.first, args.second)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "source_frames": report["source_frames"],
        "configuration_differences": report["configuration_differences"],
        "delta": report["delta_second_minus_first"],
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
