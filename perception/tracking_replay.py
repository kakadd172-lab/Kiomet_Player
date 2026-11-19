"""Replay a chronological screenshot folder through temporal entity tracking.

This runner uses local screenshots and the existing detector/tracker pipeline.
It exports per-frame track snapshots and optional annotated PNGs for inspection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys

import requests
from time import perf_counter
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont

from .tracked_entity_pipeline import TrackedEntityPipeline
from .vision_entity_detector import VisionEntityDetector

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def discover_screenshots(source: Path) -> list[Path]:
    """Return supported images in stable filename order."""
    if not source.exists():
        raise FileNotFoundError(f"Screenshot source does not exist: {source}")
    if source.is_file():
        if source.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported screenshot format: {source.suffix}")
        return [source]
    if not source.is_dir():
        raise ValueError(f"Screenshot source is not a file or directory: {source}")
    return sorted(
        (path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES),
        key=lambda path: (path.name.casefold(), str(path).casefold()),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _track_record(track: Any) -> dict[str, Any]:
    x, y, width, height = track.bbox
    return {
        "track_id": track.track_id,
        "category": track.category,
        "identity": track.identity,
        "confidence": track.confidence,
        "bbox": {"x": x, "y": y, "width": width, "height": height},
        "first_seen_frame": track.first_seen_frame,
        "last_seen_frame": track.last_seen_frame,
        "missed_frames": track.missed_frames,
        "visible": track.missed_frames == 0,
    }


def _detection_record(detection: Any) -> dict[str, Any]:
    """Preserve raw proposal and recognition evidence alongside temporal tracks."""
    proposal = detection.proposal
    recognition = detection.recognition
    return {
        "region_id": proposal.region_id,
        "tile": detection.tile_name,
        "category": proposal.category,
        "bbox": {
            "x": proposal.x,
            "y": proposal.y,
            "width": proposal.width,
            "height": proposal.height,
        },
        "detector_confidence": detection.detector_confidence,
        "recognition": {
            "accepted": recognition.accepted,
            "status": recognition.status,
            "icon_id": recognition.icon_id,
            "display_name": recognition.display_name,
            "top_score": recognition.top_score,
            "margin": recognition.margin,
            "candidates": [
                {
                    "icon_id": candidate.icon_id,
                    "display_name": candidate.display_name,
                    "score": candidate.score,
                }
                for candidate in recognition.candidates
            ],
        },
    }


def annotate_screenshot(source: Path, destination: Path, tracks: Iterable[Any], frame_index: int) -> None:
    """Write a copy of a screenshot with current track boxes and IDs."""
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for track in tracks:
        if track.missed_frames:
            continue
        x, y, box_width, box_height = track.bbox
        left, top = round(x * width), round(y * height)
        right, bottom = round((x + box_width) * width), round((y + box_height) * height)
        label = f"{track.track_id} {track.identity or 'unknown'}"
        draw.rectangle((left, top, right, bottom), outline=(30, 220, 90), width=max(2, width // 900))
        try:
            bounds = draw.textbbox((left, top), label)
            label_height = bounds[3] - bounds[1]
        except AttributeError:
            label_height = 12
        label_top = max(0, top - label_height - 4)
        draw.rectangle((left, label_top, left + max(80, len(label) * 7), top), fill=(20, 30, 20))
        draw.text((left + 2, label_top + 1), label, fill=(255, 255, 255))
    draw.text((8, 8), f"Kiomet tracking replay | frame {frame_index}", fill=(255, 255, 0))
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG")


def run_replay(
    source: Path,
    output: Path,
    *,
    pipeline: TrackedEntityPipeline | None = None,
    annotated_dir: Path | None = None,
    method: str = "shape",
    min_score: float = 0.65,
    min_margin: float = 0.10,
) -> dict[str, Any]:
    """Process all source screenshots in chronological filename order."""
    screenshots = discover_screenshots(source)
    if not screenshots:
        raise ValueError(f"No supported screenshots found in {source}")
    active_pipeline = pipeline or TrackedEntityPipeline()
    active_pipeline.reset()
    frames = []
    replay_started = perf_counter()
    for frame_index, screenshot in enumerate(screenshots, start=1):
        frame_started = perf_counter()
        current = active_pipeline.process(
            screenshot,
            method=method,
            min_score=min_score,
            min_margin=min_margin,
        )
        all_tracks = active_pipeline.snapshot(include_missing=True)
        visible_tracks = [track for track in current if track.missed_frames == 0]
        visible_records = [_track_record(track) for track in visible_tracks]
        frame_duration_seconds = perf_counter() - frame_started
        frames.append({
            "frame_index": frame_index,
            "source": screenshot.name,
            "source_path": str(screenshot),
            "source_sha256": _sha256(screenshot),
            "processing_seconds": round(frame_duration_seconds, 6),
            # "predicted" matches the tracking benchmark input schema; add a
            # manually reviewed "expected" list to evaluate this replay.
            "predicted": visible_records,
            "tracks": visible_records,
            # Keep detector and recognizer evidence separate from track confidence.
            "detections": [
                _detection_record(item)
                for item in getattr(active_pipeline, "last_detections", ())
            ],
            "temporarily_missing_tracks": [
                _track_record(track) for track in all_tracks if track.missed_frames > 0
            ],
        })
        if annotated_dir is not None:
            annotate_screenshot(
                screenshot,
                annotated_dir / f"{frame_index:05d}_{screenshot.stem}.png",
                visible_tracks,
                frame_index,
            )

    total_duration_seconds = perf_counter() - replay_started
    report = {
        "benchmark": "kiomet_temporal_tracking_replay",
        "schema_version": 1,
        "source": str(source),
        "frames_processed": len(frames),
        "runtime": {
            "total_seconds": round(total_duration_seconds, 6),
            "mean_frame_seconds": round(total_duration_seconds / len(frames), 6),
            "frames_per_second": round(len(frames) / total_duration_seconds, 4) if total_duration_seconds > 0 else None,
        },
        "detector": {
            "model": active_pipeline.detector.model,
            "method": method,
            "min_score": min_score,
            "min_margin": min_margin,
            "tiled": active_pipeline.detector.tiled,
            "max_image_side": active_pipeline.detector.max_image_side,
            "upscale": active_pipeline.detector.upscale,
            "overlap": active_pipeline.detector.overlap,
            "num_threads": active_pipeline.detector.num_threads,
            "min_detector_confidence": active_pipeline.detector.min_detector_confidence,
            "timeout_seconds": active_pipeline.detector.timeout,
        },
        "frames": frames,
        "notes": [
            "Screenshots are processed in stable filename order; ensure filenames represent chronological order.",
            "Track IDs are local to this replay and reset at its start.",
            "A track ID is a spatial association hypothesis, not proof of persistent real-world identity.",
            "Ground-truth labels are not inferred; use the tracking benchmark with manually linked labels for accuracy metrics.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Screenshot file or directory")
    parser.add_argument("--output", type=Path, required=True, help="JSON replay output")
    parser.add_argument("--annotated-dir", type=Path, help="Optional directory for annotated PNGs")
    parser.add_argument("--model", default="qwen3-vl:4b-instruct")
    parser.add_argument("--method", choices=("shape", "template", "orb"), default="shape")
    parser.add_argument("--max-image-side", type=int, default=960)
    parser.add_argument("--native-resolution", action="store_true")
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--min-score", type=float, default=0.65)
    parser.add_argument("--min-margin", type=float, default=0.10)
    parser.add_argument("--min-detector-confidence", type=float, default=0.35)
    args = parser.parse_args()
    detector = VisionEntityDetector(
        model=args.model,
        max_image_side=None if args.native_resolution else args.max_image_side,
        tiled=args.tiled,
        timeout=args.timeout,
        min_detector_confidence=args.min_detector_confidence,
    )
    pipeline = TrackedEntityPipeline(detector=detector)
    try:
        report = run_replay(
            args.source,
            args.output,
            pipeline=pipeline,
            annotated_dir=args.annotated_dir,
            method=args.method,
            min_score=args.min_score,
            min_margin=args.min_margin,
        )
    except (RuntimeError, ValueError, OSError, requests.exceptions.RequestException) as error:
        print(f"[Replay failed] {error}", file=sys.stderr)
        return 1
    print(json.dumps({
        "frames_processed": report["frames_processed"],
        "output": str(args.output),
        "annotated_dir": str(args.annotated_dir) if args.annotated_dir else None,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
