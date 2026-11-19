"""Use a local vision model to propose Kiomet entity boxes, then verify crops.

The VLM is a region proposer, not the final identity authority. Its boxes are
mapped from either a full-frame request or overlapping tiles into full-screen
coordinates and passed to deterministic tower/unit recognizers. Unknown
categories are intentionally left out rather than guessed.
"""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from time import perf_counter
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image

from .entity_recognizer import (
    EntityObservation,
    RegionProposal,
    ScreenEntityRecognizer,
)


DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen3-vl:4b-instruct"
DEFAULT_MAX_OUTPUT_TOKENS = 4096
MIN_SPLIT_DIMENSION = 64


@dataclass(frozen=True)
class VisionProposal:
    proposal: RegionProposal
    detector_confidence: float
    tile_name: str


@dataclass(frozen=True)
class VerifiedEntity:
    proposal: RegionProposal
    detector_confidence: float
    tile_name: str
    recognition: EntityObservation


def _load_rgb(image: np.ndarray | str | Path) -> Image.Image:
    if isinstance(image, (str, Path)):
        try:
            with Image.open(image) as opened:
                return opened.convert("RGB")
        except OSError as error:
            raise ValueError(f"Could not load screenshot: {image}") from error

    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Screenshot must be a non-empty image array or path")
    if image.ndim == 2:
        return Image.fromarray(image.astype(np.uint8), mode="L").convert("RGB")
    if image.ndim == 3 and image.shape[2] == 3:
        # ScreenEntityRecognizer accepts OpenCV-style BGR arrays.
        return Image.fromarray(image[:, :, ::-1].astype(np.uint8), mode="RGB")
    if image.ndim == 3 and image.shape[2] == 4:
        return Image.fromarray(image[:, :, [2, 1, 0]].astype(np.uint8), mode="RGB")
    raise ValueError(f"Unsupported screenshot shape: {image.shape!r}")


def _tile_bounds(width: int, height: int, overlap: float) -> list[tuple[str, int, int, int, int]]:
    if not isinstance(overlap, (int, float)) or isinstance(overlap, bool) or not 0 <= overlap < 0.5:
        raise ValueError("overlap must be between 0 and 0.5")
    if not isinstance(width, int) or not isinstance(height, int) or width < 1 or height < 1:
        raise ValueError("width and height must be positive integers")

    # Avoid splitting a short axis into tiny strips. This is especially
    # important for panoramic or partial-screen captures.
    tile_width = (
        min(width, int(round(width * (0.5 + overlap / 2))))
        if width >= MIN_SPLIT_DIMENSION else width
    )
    tile_height = (
        min(height, int(round(height * (0.5 + overlap / 2))))
        if height >= MIN_SPLIT_DIMENSION else height
    )
    x_positions = sorted({0, max(0, width - tile_width)})
    y_positions = sorted({0, max(0, height - tile_height)})
    result = []
    for row, top in enumerate(y_positions):
        for column, left in enumerate(x_positions):
            result.append((
                ("top" if row == 0 else "bottom") + "_" + ("left" if column == 0 else "right"),
                left,
                top,
                min(width, left + tile_width),
                min(height, top + tile_height),
            ))
    return result


def _timing_summary(metrics: list[dict[str, Any]], field: str) -> dict[str, float | int | None]:
    """Summarize a timing field without treating missing Ollama timings as zero."""
    values = [
        float(item[field])
        for item in metrics
        if isinstance(item.get(field), (int, float))
        and not isinstance(item.get(field), bool)
    ]
    if not values:
        return {"count": 0, "total_ms": None, "mean_ms": None, "min_ms": None, "max_ms": None}
    return {
        "count": len(values),
        "total_ms": round(sum(values), 3),
        "mean_ms": round(sum(values) / len(values), 3),
        "min_ms": round(min(values), 3),
        "max_ms": round(max(values), 3),
    }


def _iou(first: RegionProposal, second: RegionProposal) -> float:
    left = max(first.x, second.x)
    top = max(first.y, second.y)
    right = min(first.x + first.width, second.x + second.width)
    bottom = min(first.y + first.height, second.y + second.height)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = first.width * first.height + second.width * second.height - intersection
    return intersection / union if union > 0 else 0.0


class VisionEntityDetector:
    """Propose entity boxes with Ollama, then verify their icon identities."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        timeout: float = 180.0,
        overlap: float = 0.12,
        upscale: float = 1.0,
        max_image_side: int | None = 960,
        tiled: bool = False,
        num_threads: int | None = None,
        min_detector_confidence: float = 0.35,
        duplicate_iou: float = 0.45,
        screen_recognizer: ScreenEntityRecognizer | None = None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if not isinstance(ollama_url, str) or not ollama_url.strip():
            raise ValueError("ollama_url must be a non-empty string")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if not isinstance(upscale, (int, float)) or isinstance(upscale, bool) or upscale < 0.25:
            raise ValueError("upscale must be at least 0.25")
        if max_image_side is not None and (
            not isinstance(max_image_side, int)
            or isinstance(max_image_side, bool)
            or max_image_side < 256
        ):
            raise ValueError("max_image_side must be None (native resolution) or an integer of at least 256")
        if not isinstance(tiled, bool):
            raise ValueError("tiled must be a boolean")
        if num_threads is not None and (
            not isinstance(num_threads, int) or isinstance(num_threads, bool) or num_threads < 1
        ):
            raise ValueError("num_threads must be None (Ollama default) or a positive integer")
        if not 0 <= min_detector_confidence <= 1:
            raise ValueError("min_detector_confidence must be between 0 and 1")
        if not 0 < duplicate_iou <= 1:
            raise ValueError("duplicate_iou must be greater than 0 and at most 1")
        self.model = model
        self.ollama_url = ollama_url
        self.timeout = timeout
        self.overlap = overlap
        self.upscale = float(upscale)
        self.max_image_side = max_image_side
        self.tiled = tiled
        self.num_threads = num_threads
        self.min_detector_confidence = float(min_detector_confidence)
        self.duplicate_iou = float(duplicate_iou)
        self.screen_recognizer = screen_recognizer or ScreenEntityRecognizer()
        self.inference_metrics: list[dict[str, Any]] = []
        self.pipeline_metrics: dict[str, Any] = {}

    @staticmethod
    def _prompt(tile_width: int, tile_height: int) -> str:
        return (
            "You are inspecting a screenshot from the strategy game Kiomet. Find individual "
            "game-world entities visible in this image crop. Return one box for each distinct "
            "visible entity, including neutral/grey entities and coloured/player-controlled "
            "entities. Category 'tower' means a stationary game structure or building. "
            "Category 'unit' means any mobile or special game entity represented by a unit "
            "marker, including the King and special entities; do not restrict this category "
            "to conventional vehicles, ships, or soldiers. Use 'unknown' only when a visible "
            "game entity is clear but its tower-versus-unit category is genuinely uncertain. "
            "Do not identify exact names; identity is handled by a separate recognizer. "
            "Look for the actual compact game symbols/markers on the map, not just large "
            "objects. Include symbols on both neutral and owned territory. "
            "Do not report territory shapes, borders, roads, connection lines, text, numbers, "
            "standalone counters, buttons, panels, minimap decorations, or isolated dots as "
            "entities. A glow, shadow, circular badge, or coloured background is not an entity "
            "by itself, but do not discard an icon merely because it has a glow or badge behind it. "
            "Return a tight box around the complete visible symbol, including meaningful "
            "detached parts that clearly belong to it, while excluding most surrounding map. "
            "Do not invent entities. Do not merge nearby separate symbols. "
            f"The attached crop is exactly {tile_width} pixels wide and {tile_height} pixels "
            "high. Coordinates must be normalized fractions from 0 to 1 relative to this "
            "attached crop only, not the original screenshot. Origin (0,0) is its top-left. "
            "bbox.x and bbox.y are the top-left corner; bbox.width and bbox.height are "
            "dimensions, not bottom-right coordinates. Keep boxes fully inside this crop. "
            "Return JSON only with an 'entities' array. Every item must contain category "
            "('tower', 'unit', or 'unknown'), bbox with numeric x/y/width/height, and a "
            "numeric confidence from 0 to 1. Return all clearly visible entities rather than "
            "stopping after a small handful. If none are visible, return an empty array. "
            "Do not copy example coordinates. Do not use fixed confidence values or add explanations."
        )

    def _request_tile(self, tile: Image.Image) -> dict[str, Any]:
        source_width, source_height = tile.size
        # Native mode preserves the exact source pixels: no downscaling
        # and no optional upscaling.
        scale = 1.0 if self.max_image_side is None else min(
            self.upscale,
            self.max_image_side / max(source_width, source_height),
        )
        if scale != 1:
            tile = tile.resize(
                (max(1, round(source_width * scale)), max(1, round(source_height * scale))),
                Image.Resampling.LANCZOS,
            )
        preprocessing_started = perf_counter()
        buffer = BytesIO()
        tile.save(buffer, format="PNG")
        png_bytes = buffer.getvalue()
        encoded = base64.b64encode(png_bytes).decode("ascii")
        preprocessing_ms = (perf_counter() - preprocessing_started) * 1000.0
        request_started = perf_counter()
        response = requests.post(
            self.ollama_url,
            json={
                "model": self.model,
                "messages": [{
                    "role": "user",
                    "content": self._prompt(tile.width, tile.height),
                    "images": [encoded],
                }],
                "format": {
                    "type": "object",
                    "properties": {
                        "entities": {
                            "type": "array",
                            "maxItems": 64,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "category": {"type": "string", "enum": ["tower", "unit", "unknown"]},
                                    "bbox": {
                                        "type": "object",
                                        "properties": {
                                            "x": {"type": "number"},
                                            "y": {"type": "number"},
                                            "width": {"type": "number"},
                                            "height": {"type": "number"},
                                        },
                                        "required": ["x", "y", "width", "height"],
                                    },
                                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                                },
                                "required": ["category", "bbox", "confidence"],
                            },
                        }
                    },
                    "required": ["entities"],
                },
                "stream": False,
                "keep_alive": "5m",
                "options": {
                    "temperature": 0.1,
                    "num_predict": DEFAULT_MAX_OUTPUT_TOKENS,
                    **({"num_thread": self.num_threads} if self.num_threads is not None else {}),
                },
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        content = payload.get("message", {}).get("content")
        done_reason = payload.get("done_reason")
        if done_reason == "length":
            generated = payload.get("eval_count")
            raise ValueError(
                "Vision model response was truncated at Ollama's output-token limit "
                f"({generated if isinstance(generated, int) else 'unknown'} tokens; "
                f"configured limit={DEFAULT_MAX_OUTPUT_TOKENS}). "
                "No partial detections were accepted."
            )
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Vision model returned an empty response")
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as error:
            lines = content.splitlines()
            nearby = lines[error.lineno - 1] if 0 < error.lineno <= len(lines) else content[:180]
            nearby = nearby.strip()[:180]
            raise ValueError(
                f"Vision model returned malformed JSON ({error.msg} at line {error.lineno}, "
                f"column {error.colno}; done_reason={done_reason}; nearby={nearby!r})"
            ) from error
        if not isinstance(parsed, dict) or not isinstance(parsed.get("entities"), list):
            raise ValueError("Vision model JSON must contain an entities array")

        def duration_ms(field: str) -> float | None:
            value = payload.get(field)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return None
            return round(float(value) / 1_000_000.0, 3)

        self.inference_metrics.append({
            "tile": f"{tile.width}x{tile.height}",
            "source_width": source_width,
            "source_height": source_height,
            "input_width": tile.width,
            "input_height": tile.height,
            "resize_scale": round(scale, 6),
            "resized": tile.size != (source_width, source_height),
            "png_bytes": len(png_bytes),
            "base64_chars": len(encoded),
            "preprocessing_ms": round(preprocessing_ms, 3),
            "request_wall_ms": round((perf_counter() - request_started) * 1000.0, 3),
            "ollama_total_ms": duration_ms("total_duration"),
            "ollama_load_ms": duration_ms("load_duration"),
            "ollama_prompt_eval_ms": duration_ms("prompt_eval_duration"),
            "ollama_generation_ms": duration_ms("eval_duration"),
            "prompt_eval_tokens": payload.get("prompt_eval_count"),
            "generation_tokens": payload.get("eval_count"),
            "done_reason": payload.get("done_reason"),
        })
        return parsed

    def propose_regions(self, screenshot: np.ndarray | str | Path) -> list[VisionProposal]:
        pipeline_started = perf_counter()
        self.inference_metrics = []
        image = _load_rgb(screenshot)
        width, height = image.size
        proposals: list[VisionProposal] = []

        tiles = (
            [("full", 0, 0, width, height)]
            if not self.tiled
            else _tile_bounds(width, height, self.overlap)
        )
        for tile_name, left, top, right, bottom in tiles:
            tile = image.crop((left, top, right, bottom))
            # A failed or truncated tile must fail the frame. Silently
            # skipping it would make an incomplete detector result look valid.
            result = self._request_tile(tile)
            tile_width = right - left
            tile_height = bottom - top

            for index, item in enumerate(result["entities"]):
                if not isinstance(item, dict) or item.get("category") not in {"tower", "unit"}:
                    continue
                bbox = item.get("bbox")
                confidence = item.get("confidence")
                if not isinstance(bbox, dict):
                    continue
                try:
                    local_x = float(bbox["x"])
                    local_y = float(bbox["y"])
                    local_width = float(bbox["width"])
                    local_height = float(bbox["height"])
                    confidence = float(confidence)
                except (KeyError, TypeError, ValueError):
                    continue
                values = (local_x, local_y, local_width, local_height, confidence)
                if not all(np.isfinite(value) for value in values):
                    continue
                if not 0 <= confidence <= 1 or confidence < self.min_detector_confidence:
                    continue
                if local_width <= 0 or local_height <= 0:
                    continue
                if local_x < 0 or local_y < 0 or local_x + local_width > 1 or local_y + local_height > 1:
                    continue

                abs_x = (left + local_x * tile_width) / width
                abs_y = (top + local_y * tile_height) / height
                abs_width = local_width * tile_width / width
                abs_height = local_height * tile_height / height
                proposal = RegionProposal(
                    region_id=f"{tile_name}-{index + 1}",
                    category=item["category"],
                    x=abs_x,
                    y=abs_y,
                    width=abs_width,
                    height=abs_height,
                )
                proposals.append(VisionProposal(proposal, confidence, tile_name))

        # Overlapping tiles see some entities twice. Keep the higher-confidence
        # box when same-category proposals substantially overlap.
        proposals.sort(key=lambda item: (-item.detector_confidence, item.proposal.region_id))
        deduplicated: list[VisionProposal] = []
        for candidate in proposals:
            if any(
                candidate.proposal.category == kept.proposal.category
                and _iou(candidate.proposal, kept.proposal) >= self.duplicate_iou
                for kept in deduplicated
            ):
                continue
            deduplicated.append(candidate)
        self.pipeline_metrics = {
            "model": self.model,
            "configuration": {
                "tiled": self.tiled,
                "max_image_side": self.max_image_side,
                "upscale": self.upscale,
                "overlap": self.overlap,
                "num_threads": self.num_threads,
                "min_detector_confidence": self.min_detector_confidence,
                "duplicate_iou": self.duplicate_iou,
                "timeout_seconds": self.timeout,
            },
            "source_width": width,
            "source_height": height,
            "tile_count": len(tiles),
            "successful_inference_count": len(self.inference_metrics),
            "proposal_duration_ms": round((perf_counter() - pipeline_started) * 1000.0, 3),
            "timings": {
                field: _timing_summary(self.inference_metrics, field)
                for field in (
                    "preprocessing_ms",
                    "request_wall_ms",
                    "ollama_total_ms",
                    "ollama_load_ms",
                    "ollama_prompt_eval_ms",
                    "ollama_generation_ms",
                )
            },
            "payload": {
                "total_png_bytes": sum(item["png_bytes"] for item in self.inference_metrics),
                "total_base64_chars": sum(item["base64_chars"] for item in self.inference_metrics),
            },
        }
        return deduplicated

    def detect(
        self,
        screenshot: np.ndarray | str | Path,
        *,
        method: str = "shape",
        top_k: int = 5,
        min_score: float = 0.65,
        min_margin: float = 0.10,
        detect_colour: bool = True,
    ) -> list[VerifiedEntity]:
        """Run tiled region proposals followed by deterministic icon matching."""
        proposals = self.propose_regions(screenshot)
        recognition_started = perf_counter()
        observations = self.screen_recognizer.recognize_regions(
            screenshot,
            [item.proposal for item in proposals],
            method=method,
            top_k=top_k,
            min_score=min_score,
            min_margin=min_margin,
            detect_colour=detect_colour,
        )
        self.pipeline_metrics["recognition_duration_ms"] = round(
            (perf_counter() - recognition_started) * 1000.0, 3
        )
        self.pipeline_metrics["total_duration_ms"] = round(
            self.pipeline_metrics["proposal_duration_ms"]
            + self.pipeline_metrics["recognition_duration_ms"], 3
        )
        return [
            VerifiedEntity(
                proposal=item.proposal,
                detector_confidence=item.detector_confidence,
                tile_name=item.tile_name,
                recognition=observation,
            )
            for item, observation in zip(proposals, observations)
        ]
