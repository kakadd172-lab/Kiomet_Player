import json
import tempfile
import unittest
from pathlib import Path

from perception.screenshot_detection_benchmark import (
    box_iou,
    load_labels,
    match_entities,
)


class ScreenshotDetectionBenchmarkTests(unittest.TestCase):
    def entity(self, entity_id="a", category="tower", bbox=None, name="Radar"):
        return {
            "id": entity_id,
            "category": category,
            "bbox": bbox or {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2},
            "expected_name": name,
        }

    def test_iou_for_identical_and_disjoint_boxes(self):
        first = (0.1, 0.1, 0.2, 0.2)
        self.assertEqual(box_iou(first, first), 1.0)
        self.assertEqual(box_iou(first, (0.5, 0.5, 0.1, 0.1)), 0.0)

    def test_matches_one_prediction_to_one_label(self):
        expected = [self.entity()]
        predicted = [{
            "id": "p1",
            "category": "tower",
            "bbox": {"x": 0.11, "y": 0.11, "width": 0.2, "height": 0.2},
            "recognized_name": "Radar",
        }, {
            "id": "p2",
            "category": "tower",
            "bbox": {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2},
            "recognized_name": "Radar",
        }]
        result = match_entities(expected, predicted, 0.5)
        self.assertEqual(result["true_positives"], 1)
        self.assertEqual(result["false_positives"], 1)
        self.assertEqual(result["false_negatives"], 0)
        self.assertTrue(result["matches"][0]["identity_correct"])

    def test_matching_finds_maximum_number_of_valid_pairs(self):
        # Expected A overlaps both predictions, preferring p1 by IoU.
        # Expected B overlaps only p1. Greedy matching would consume p1 for
        # A and incorrectly leave B unmatched; reassignment should match both.
        expected = [
            self.entity("A", bbox={"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2}),
            self.entity("B", bbox={"x": 0.04, "y": 0.1, "width": 0.2, "height": 0.2}),
        ]
        predicted = [
            {
                "id": "p1", "category": "tower",
                "bbox": {"x": 0.09, "y": 0.1, "width": 0.2, "height": 0.2},
                "recognized_name": "Radar",
            },
            {
                "id": "p2", "category": "tower",
                "bbox": {"x": 0.16, "y": 0.1, "width": 0.2, "height": 0.2},
                "recognized_name": "Radar",
            },
        ]
        result = match_entities(expected, predicted, 0.5)
        self.assertEqual(result["true_positives"], 2)
        self.assertEqual(result["false_positives"], 0)
        self.assertEqual(result["false_negatives"], 0)
        self.assertEqual(
            {match["expected_id"]: match["predicted_id"] for match in result["matches"]},
            {"A": "p2", "B": "p1"},
        )

    def test_category_mismatch_is_false_positive_and_false_negative(self):
        result = match_entities(
            [self.entity(category="tower")],
            [{
                "id": "p1",
                "category": "unit",
                "bbox": {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2},
                "recognized_name": "King",
            }],
        )
        self.assertEqual(result["true_positives"], 0)
        self.assertEqual(result["false_positives"], 1)
        self.assertEqual(result["false_negatives"], 1)

    def test_identity_is_only_scored_when_expected_name_is_labelled(self):
        expected = [self.entity(name=None)]
        predicted = [{
            "id": "p1", "category": "tower",
            "bbox": {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2},
            "recognized_name": "Radar",
        }]
        result = match_entities(expected, predicted)
        self.assertIsNone(result["matches"][0]["identity_correct"])
        self.assertFalse(result["matches"][0]["identity_scored"])

    def test_rejects_non_exhaustive_screenshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "screen.png").write_bytes(b"test")
            manifest = {
                "schema_version": 1,
                "screenshots": [{
                    "name": "partial", "image": "screen.png",
                    "fully_annotated": False, "entities": [],
                }],
            }
            path = root / "labels.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fully_annotated"):
                load_labels(path)

    def test_accepts_exhaustive_empty_screen(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "screen.png").write_bytes(b"test")
            manifest = {
                "schema_version": 1,
                "screenshots": [{
                    "name": "empty", "image": "screen.png",
                    "fully_annotated": True, "entities": [],
                }],
            }
            path = root / "labels.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            _, cases = load_labels(path)
            self.assertEqual(len(cases), 1)
            self.assertEqual(cases[0]["entities"], [])

    def test_rejects_bbox_outside_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "screen.png").write_bytes(b"test")
            manifest = {
                "schema_version": 1,
                "screenshots": [{
                    "name": "bad", "image": "screen.png",
                    "fully_annotated": True,
                    "entities": [self.entity(bbox={
                        "x": 0.9, "y": 0.1, "width": 0.2, "height": 0.2
                    })],
                }],
            }
            path = root / "labels.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "normalized screenshot"):
                load_labels(path)


if __name__ == "__main__":
    unittest.main()
