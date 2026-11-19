import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from kiomet_knowledge.recognition.shared.icon_registry import IconRecord
from kiomet_knowledge.recognition.shared.image_matcher import ImageMatcher


class ImageMatcherTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        root = Path(self.temp_dir.name)

        template = np.zeros((24, 24, 4), dtype=np.uint8)
        cv2.circle(template, (12, 12), 8, (255, 255, 255, 255), -1)
        template_path = root / "circle.png"
        self.assertTrue(cv2.imwrite(str(template_path), template))

        self.record = IconRecord(
            id="tower.test",
            category="tower",
            display_name="Test Tower",
            internal_name="TestTower",
            asset_category="towers",
            filename="circle.svg",
            svg_path=root / "circle.svg",
            template_path=template_path,
            source="test",
            license="test",
        )
        self.matcher = ImageMatcher([self.record])
        self.candidate = np.zeros((24, 24, 3), dtype=np.uint8)
        cv2.circle(self.candidate, (12, 12), 8, (255, 255, 255), -1)

    def test_exact_template_is_top_match(self):
        results = self.matcher.match(self.candidate)
        self.assertEqual(len(results), 1)
        self.assertIs(results[0].record, self.record)
        self.assertAlmostEqual(results[0].score, 1.0)

    def test_category_filter(self):
        self.assertEqual(len(self.matcher.match(self.candidate, category="tower")), 1)
        self.assertEqual(self.matcher.match(self.candidate, category="unit"), [])

    def test_top_k_must_be_positive(self):
        with self.assertRaises(ValueError):
            self.matcher.match(self.candidate, top_k=0)

    def test_unknown_category_is_rejected(self):
        with self.assertRaises(ValueError):
            self.matcher.match(self.candidate, category="building")

    def test_invalid_candidate_is_rejected(self):
        with self.assertRaises(ValueError):
            self.matcher.match(np.array([]))


if __name__ == "__main__":
    unittest.main()
