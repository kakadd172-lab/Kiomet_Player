import unittest
from types import SimpleNamespace

import numpy as np

from perception.entity_recognizer import (
    RegionProposal,
    ScreenEntityRecognizer,
)


class FakeRecognizer:
    def __init__(self, category):
        self.category = category
        self.calls = []

    def recognize(self, image, **options):
        self.calls.append((image.copy(), options))
        match = SimpleNamespace(
            record=SimpleNamespace(
                id=f"{self.category}.example",
                display_name=f"Example {self.category}",
            ),
            score=0.91,
        )
        return SimpleNamespace(
            candidates=(match,),
            accepted=True,
            status="strong",
            icon_id=match.record.id,
            display_name=match.record.display_name,
            top_score=0.91,
            margin=0.25,
            colour=None,
        )


class ScreenEntityRecognizerTests(unittest.TestCase):
    def setUp(self):
        self.towers = FakeRecognizer("tower")
        self.troops = FakeRecognizer("unit")
        self.recognizer = ScreenEntityRecognizer(self.towers, self.troops)
        self.frame = np.zeros((100, 200, 3), dtype=np.uint8)
        self.frame[20:60, 40:100] = (10, 80, 160)

    def test_crops_normalized_box_and_preserves_screen_coordinates(self):
        proposal = RegionProposal("tower-1", "tower", 0.2, 0.2, 0.3, 0.4)
        result = self.recognizer.recognize_regions(self.frame, [proposal])[0]

        self.assertEqual(result.region_id, "tower-1")
        self.assertEqual(result.category, "tower")
        self.assertEqual(result.bbox, (0.2, 0.2, 0.3, 0.4))
        self.assertEqual(result.pixel_bbox, (40, 20, 100, 60))
        self.assertEqual(self.towers.calls[0][0].shape, (40, 60, 3))
        self.assertEqual(self.towers.calls[0][0][0, 0].tolist(), [10, 80, 160])
        self.assertEqual(result.icon_id, "tower.example")
        self.assertEqual(result.candidates[0].score, 0.91)

    def test_routes_unit_regions_to_troop_recognizer(self):
        proposal = RegionProposal("unit-1", "unit", 0.2, 0.2, 0.3, 0.4)
        result = self.recognizer.recognize_regions(self.frame, [proposal])[0]

        self.assertEqual(result.category, "unit")
        self.assertEqual(result.icon_id, "unit.example")
        self.assertEqual(len(self.troops.calls), 1)
        self.assertEqual(len(self.towers.calls), 0)

    def test_empty_proposals_return_empty_list(self):
        self.assertEqual(self.recognizer.recognize_regions(self.frame, []), [])

    def test_invalid_category_is_rejected(self):
        proposal = RegionProposal("bad", "building", 0.1, 0.1, 0.2, 0.2)
        with self.assertRaisesRegex(ValueError, "category"):
            self.recognizer.recognize_regions(self.frame, [proposal])

    def test_out_of_bounds_box_is_rejected(self):
        proposal = RegionProposal("bad", "tower", 0.9, 0.1, 0.2, 0.2)
        with self.assertRaisesRegex(ValueError, "fit"):
            self.recognizer.recognize_regions(self.frame, [proposal])

    def test_zero_sized_box_is_rejected(self):
        proposal = RegionProposal("bad", "tower", 0.1, 0.1, 0.0, 0.2)
        with self.assertRaisesRegex(ValueError, "positive"):
            self.recognizer.recognize_regions(self.frame, [proposal])

    def test_duplicate_region_ids_are_allowed_as_independent_proposals(self):
        proposal = RegionProposal("same", "tower", 0.2, 0.2, 0.3, 0.4)
        results = self.recognizer.recognize_regions(self.frame, [proposal, proposal])
        self.assertEqual([item.region_id for item in results], ["same", "same"])
        self.assertEqual(len(self.towers.calls), 2)


if __name__ == "__main__":
    unittest.main()
