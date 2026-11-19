import unittest
from unittest.mock import patch

import numpy as np

from perception.vision_entity_detector import (
    VisionEntityDetector,
    _tile_bounds,
    _timing_summary,
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
            tiled=True,
        )

    def test_default_inference_settings_use_one_bounded_eight_thread_pass(self):
        detector = VisionEntityDetector(screen_recognizer=self.screen_recognizer)
        self.assertFalse(detector.tiled)
        self.assertEqual(detector.max_image_side, 960)
        self.assertEqual(detector.upscale, 1.0)
        self.assertEqual(detector.num_threads, 8)

    def test_native_resolution_is_supported(self):
        detector = VisionEntityDetector(
            screen_recognizer=self.screen_recognizer,
            max_image_side=None,
        )
        self.assertIsNone(detector.max_image_side)
        from PIL import Image
        with patch.object(detector, "_request_tile", return_value={"entities": []}) as request:
            detector.propose_regions(self.image)
        request.assert_called_once()
        self.assertEqual(request.call_args.args[0].size, (100, 100))

    def test_invalid_max_image_side_values_are_rejected(self):
        for value in (255, 0, -1, 960.0, True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    VisionEntityDetector(
                        screen_recognizer=self.screen_recognizer,
                        max_image_side=value,
                    )

    def test_default_detector_requests_one_full_frame(self):
        detector = VisionEntityDetector(screen_recognizer=self.screen_recognizer)
        with patch.object(detector, "_request_tile", return_value={"entities": []}) as request:
            self.assertEqual(detector.propose_regions(self.image), [])
        request.assert_called_once()
        self.assertEqual(request.call_args.args[0].size, (100, 100))

    def test_malformed_model_json_skips_tile_instead_of_crashing(self):
        detector = VisionEntityDetector(screen_recognizer=self.screen_recognizer)
        with patch.object(
            detector,
            "_request_tile",
            side_effect=ValueError("Vision model returned malformed JSON (bad token at line 1, column 2)"),
        ):
            with self.assertWarns(RuntimeWarning):
                self.assertEqual(detector.propose_regions(self.image), [])

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

    def test_prompt_defines_tile_relative_coordinates_and_categories(self):
        prompt = VisionEntityDetector._prompt(280, 315)
        self.assertIn("exactly 280 pixels wide and 315 pixels high", prompt)
        self.assertIn("relative to this attached crop only", prompt)
        self.assertIn("bbox.width and bbox.height are dimensions", prompt)
        self.assertIn("including the King and special entities", prompt)
        self.assertIn("Do not copy example coordinates", prompt)


    def test_prompt_covers_neutral_entities_and_special_units(self):
        prompt = VisionEntityDetector._prompt(640, 450)
        self.assertIn("neutral/grey entities", prompt)
        self.assertIn("coloured/player-controlled entities", prompt)
        self.assertIn("including the King and special entities", prompt)
        self.assertIn("Do not merge nearby separate symbols", prompt)

    def test_native_mode_does_not_resize_source_image(self):
        from PIL import Image
        detector = VisionEntityDetector(
            screen_recognizer=self.screen_recognizer,
            max_image_side=None,
        )
        response = unittest.mock.Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "message": {"content": '{"entities": []}'},
            "total_duration": 1_000_000,
        }
        with patch("PIL.Image.Image.resize", side_effect=AssertionError("native mode resized image")):
            with patch("perception.vision_entity_detector.requests.post", return_value=response):
                self.assertEqual(detector._request_tile(Image.new("RGB", (320, 200))), {"entities": []})

    def test_timing_summary_reports_total_mean_min_and_max(self):
        metrics = [
            {"request_wall_ms": 10.0, "ollama_total_ms": 8.0},
            {"request_wall_ms": 20.0, "ollama_total_ms": None},
            {"request_wall_ms": 30.0, "ollama_total_ms": 12.0},
        ]
        self.assertEqual(
            _timing_summary(metrics, "request_wall_ms"),
            {"count": 3, "total_ms": 60.0, "mean_ms": 20.0, "min_ms": 10.0, "max_ms": 30.0},
        )
        self.assertEqual(
            _timing_summary(metrics, "ollama_total_ms"),
            {"count": 2, "total_ms": 20.0, "mean_ms": 10.0, "min_ms": 8.0, "max_ms": 12.0},
        )
        self.assertEqual(
            _timing_summary(metrics, "missing"),
            {"count": 0, "total_ms": None, "mean_ms": None, "min_ms": None, "max_ms": None},
        )

    def test_request_records_ollama_timing_metrics(self):
        from PIL import Image
        detector = VisionEntityDetector(screen_recognizer=self.screen_recognizer)
        response = unittest.mock.Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "message": {"content": '{"entities": []}'},
            "total_duration": 2_000_000_000,
            "load_duration": 100_000_000,
            "prompt_eval_duration": 1_200_000_000,
            "eval_duration": 500_000_000,
            "prompt_eval_count": 320,
            "eval_count": 18,
            "done_reason": "stop",
        }
        with patch("perception.vision_entity_detector.requests.post", return_value=response):
            result = detector._request_tile(Image.new("RGB", (320, 200)))
        self.assertEqual(result, {"entities": []})
        metric = detector.inference_metrics[0]
        self.assertEqual(metric["source_width"], 320)
        self.assertEqual(metric["source_height"], 200)
        self.assertEqual(metric["input_width"], 320)
        self.assertEqual(metric["input_height"], 200)
        self.assertEqual(metric["resize_scale"], 1.0)
        self.assertFalse(metric["resized"])
        self.assertGreater(metric["png_bytes"], 0)
        self.assertEqual(metric["base64_chars"] % 4, 0)
        self.assertEqual(metric["ollama_total_ms"], 2000.0)
        self.assertEqual(metric["ollama_load_ms"], 100.0)
        self.assertEqual(metric["ollama_prompt_eval_ms"], 1200.0)
        self.assertEqual(metric["ollama_generation_ms"], 500.0)
        self.assertEqual(metric["prompt_eval_tokens"], 320)
        self.assertEqual(metric["generation_tokens"], 18)
        self.assertEqual(metric["done_reason"], "stop")

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
