"""Use a local vision model to propose Kiomet entity boxes, then verify crops.

The VLM is a region proposer, not the final identity authority. Its boxes are
mapped from overlapping tiles into full-screen coordinates and passed to the
deterministic tower/unit recognizers. Unknown categories are intentionally
left out rather than guessed.
"""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass
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
    tile_width = min(width, int(round(width * (0.5 + overlap / 2))))
    tile_height = min(height, int(round(height * (0.5 + overlap / 2))))
    step_x = max(1, width - tile_width)
    step_y = max(1, height - tile_height)
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
        upscale: float = 1.5,
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
        if not isinstance(upscale, (int, float)) or isinstance(upscale, bool) or upscale < 1:
            raise ValueError("upscale must be at least 1")
        if not 0 <= min_detector_confidence <= 1:
            raise ValueError("min_detector_confidence must be between 0 and 1")
        if not 0 < duplicate_iou <= 1:
            raise ValueError("duplicate_iou must be greater than 0 and at most 1")
        self.model = model
        self.ollama_url = ollama_url
        self.timeout = timeout
        self.overlap = overlap
        self.upscale = float(upscale)
        self.min_detector_confidence = float(min_detector_confidence)
        self.duplicate_iou = float(duplicate_iou)
        self.screen_recognizer = screen_recognizer or ScreenEntityRecognizer()

    @staticmethod
    def _prompt() -> str:
        return (
            "Find visible GAME-WORLD entity symbols in this Kiomet strategy-game image crop. "
            "Return compact bounding boxes around individual tower/building symbols or unit "
            "symbols only. Do not box territory borders, connection lines, labels, UI buttons, "
            "glows, or isolated decorative dots. Do not infer hidden objects. "
            "For each entity return category 'tower' or 'unit'; use 'unknown' if unsure. "
            "Coordinates x,y,width,height are normalized to THIS IMAGE, with origin at top-left. "
            "Boxes must tightly contain the visible symbol, not its surrounding territory. "
            "Confidence is a conservative visual estimate from 0 to 1. "
            "Return JSON only in this exact shape: "
            '{"entities":[{"category":"tower","bbox":{"x":0.1,"y":0.2,"width":0.05,"height":0.06},"confidence":0.8}]}'
        )

    def _request_tile(self, tile: Image.Image) -> dict[str, Any]:
        if self.upscale != 1:
            tile = tile.resize(
                (max(1, round(tile.width * self.upscale)), max(1, round(tile.height * self.upscale))),
                Image.Resampling.LANCZOS,
            )
        buffer = BytesIO()
        tile.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        response = requests.post(
            self.ollama_url,
            json={
                "model": self.model,
                "messages": [{
                    "role": "user",
                    "content": self._prompt(),
                    "images": [encoded],
                }],
                "format": "json",
                "stream": False,
                "keep_alive": "5m",
                "options": {"temperature": 0.1},
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        content = payload.get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Vision model returned an empty response")
        parsed = json.loads(content)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("entities"), list):
            raise ValueError("Vision model JSON must contain an entities array")
        return parsed

    def propose_regions(self, screenshot: np.ndarray | str | Path) -> list[VisionProposal]:
        image = _load_rgb(screenshot)
        width, height = image.size
        proposals: list[VisionProposal] = []

        for tile_name, left, top, right, bottom in _tile_bounds(width, height, self.overlap):
            tile = image.crop((left, top, right, bottom))
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
        observations = self.screen_recognizer.recognize_regions(
            screenshot,
            [item.proposal for item in proposals],
            method=method,
            top_k=top_k,
            min_score=min_score,
            min_margin=min_margin,
            detect_colour=detect_colour,
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
