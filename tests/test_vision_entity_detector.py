import unittest
from unittest.mock import patch

import numpy as np

from perception.vision_entity_detector import (
    VisionEntityDetector,
    _tile_bounds,
)


class FakeScreenRecognizer:
    def __init__(self):
        self.calls = []

    def recognize_regions(self, screenshot, proposals, **options):
        self.calls.append((screenshot, list(proposals), options))
        return []


class VisionEntityDetectorTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.screen_recognizer = FakeScreenRecognizer()
        self.detector = VisionEntityDetector(
            screen_recognizer=self.screen_recognizer,
            overlap=0.12,
            upscale=1,
        )

    def test_tiles_cover_full_image_with_overlap(self):
        tiles = _tile_bounds(100, 100, 0.12)
        self.assertEqual(len(tiles), 4)
        self.assertEqual(tiles[0], ("top_left", 0, 0, 56, 56))
        self.assertEqual(tiles[-1], ("bottom_right", 44, 44, 100, 100))

    def test_short_dimension_is_not_split(self):
        self.assertEqual(
            _tile_bounds(100, 40, 0.12),
            [
                ("top_left", 0, 0, 56, 40),
                ("top_right", 44, 0, 100, 40),
            ],
        )

    def test_invalid_overlap_is_rejected(self):
        with self.assertRaises(ValueError):
            _tile_bounds(100, 100, 0.5)

    def test_proposals_are_mapped_to_full_screenshot_coordinates(self):
        responses = [
            {"entities": [{
                "category": "tower",
                "bbox": {"x": 0.25, "y": 0.25, "width": 0.2, "height": 0.2},
                "confidence": 0.9,
            }]},
            {"entities": []},
            {"entities": []},
            {"entities": []},
        ]
        with patch.object(self.detector, "_request_tile", side_effect=responses):
            proposals = self.detector.propose_regions(self.image)

        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0].tile_name, "top_left")
        self.assertAlmostEqual(proposals[0].proposal.x, 0.14)
        self.assertAlmostEqual(proposals[0].proposal.y, 0.14)
        self.assertAlmostEqual(proposals[0].proposal.width, 0.112)
        self.assertAlmostEqual(proposals[0].proposal.height, 0.112)
        self.assertEqual(proposals[0].proposal.category, "tower")

    def test_unknown_category_and_low_confidence_are_ignored(self):
        responses = [
            {"entities": [
                {
                    "category": "unknown",
                    "bbox": {"x": 0.1, "y": 0.1, "width": 0.1, "height": 0.1},
                    "confidence": 0.99,
                },
                {
                    "category": "unit",
                    "bbox": {"x": 0.2, "y": 0.2, "width": 0.1, "height": 0.1},
                    "confidence": 0.1,
                },
            ]},
            {"entities": []},
            {"entities": []},
            {"entities": []},
        ]
        with patch.object(self.detector, "_request_tile", side_effect=responses):
            self.assertEqual(self.detector.propose_regions(self.image), [])

    def test_malformed_boxes_are_ignored(self):
        responses = [
            {"entities": [
                {"category": "tower", "bbox": {"x": 2, "y": 0, "width": 0.1, "height": 0.1}, "confidence": 0.9},
                {"category": "tower", "bbox": {"x": 0, "y": 0, "width": 0.1, "height": 0.1}, "confidence": "bad"},
            ]},
            {"entities": []},
            {"entities": []},
            {"entities": []},
        ]
        with patch.object(self.detector, "_request_tile", side_effect=responses):
            self.assertEqual(self.detector.propose_regions(self.image), [])

    def test_overlapping_tile_proposals_are_deduplicated(self):
        # Both boxes map to the same full-screen rectangle: [0.45, 0.50]^2.
        responses = [
            {"entities": [{
                "category": "tower",
                "bbox": {"x": 0.8035714286, "y": 0.8035714286, "width": 0.0892857143, "height": 0.0892857143},
                "confidence": 0.7,
            }]},
            {"entities": []},
            {"entities": []},
            {"entities": [{
                "category": "tower",
                "bbox": {"x": 0.0178571429, "y": 0.0178571429, "width": 0.0892857143, "height": 0.0892857143},
                "confidence": 0.95,
            }]},
        ]
        with patch.object(self.detector, "_request_tile", side_effect=responses):
            proposals = self.detector.propose_regions(self.image)

        self.assertEqual(len(proposals), 1)
        self.assertAlmostEqual(proposals[0].detector_confidence, 0.95)
        self.assertEqual(proposals[0].tile_name, "bottom_right")

    def test_detect_passes_proposals_to_deterministic_recognizer(self):
        proposal_response = [
            {"entities": [{
                "category": "unit",
                "bbox": {"x": 0.2, "y": 0.2, "width": 0.1, "height": 0.1},
                "confidence": 0.8,
            }]},
            {"entities": []},
            {"entities": []},
            {"entities": []},
        ]
        with patch.object(self.detector, "_request_tile", side_effect=proposal_response):
            self.detector.detect(self.image, method="grayscale")

        self.assertEqual(len(self.screen_recognizer.calls), 1)
        _, proposals, options = self.screen_recognizer.calls[0]
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0].category, "unit")
        self.assertEqual(options["method"], "grayscale")


if __name__ == "__main__":
    unittest.main()
