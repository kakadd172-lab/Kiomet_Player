import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from kiomet_knowledge.recognition.shared.icon_registry import IconRecord
from kiomet_knowledge.recognition.towers.tower_recognizer import TowerRecognizer
from kiomet_knowledge.recognition.troops.troop_recognizer import TroopRecognizer


class CategoryRecognizerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        root = Path(self.temp_dir.name)

        circle = np.zeros((32, 32, 4), dtype=np.uint8)
        cv2.circle(circle, (16, 16), 11, (255, 255, 255, 255), -1)
        square = np.zeros((32, 32, 4), dtype=np.uint8)
        cv2.rectangle(square, (6, 6), (26, 26), (255, 255, 255, 255), -1)

        self.tower_record = self._record(root, "tower.circle", "tower", "Circle", circle)
        self.other_tower = self._record(root, "tower.square", "tower", "Square", square)
        self.troop_record = self._record(root, "unit.circle", "unit", "Circle Unit", circle)

        self.towers = TowerRecognizer([self.tower_record, self.other_tower])
        self.troops = TroopRecognizer([self.troop_record])
        self.circle_crop = cv2.cvtColor(circle, cv2.COLOR_BGRA2BGR)

    @staticmethod
    def _record(root, icon_id, category, name, image):
        path = root / f"{icon_id.replace('.', '_')}.png"
        if not cv2.imwrite(str(path), image):
            raise RuntimeError("Could not write test template")
        return IconRecord(
            id=icon_id,
            category=category,
            display_name=name,
            internal_name=name,
            asset_category="test",
            filename=path.name,
            svg_path=root / f"{path.stem}.svg",
            template_path=path,
            source="test",
            license="test",
        )

    def test_tower_recognizer_returns_named_strong_match(self):
        result = self.towers.recognize(self.circle_crop, method="grayscale")
        self.assertTrue(result.accepted)
        self.assertEqual(result.icon_id, "tower.circle")
        self.assertEqual(result.display_name, "Circle")
        self.assertEqual(result.status, "strong")

    def test_troop_recognizer_uses_unit_category(self):
        result = self.troops.recognize(self.circle_crop, method="grayscale")
        self.assertEqual(result.category, "unit")
        self.assertTrue(result.accepted)
        self.assertEqual(result.icon_id, "unit.circle")

    def test_uncertain_match_does_not_expose_a_forced_identity(self):
        result = self.towers.recognize(
            self.circle_crop,
            method="grayscale",
            min_margin=1.0,
        )
        self.assertFalse(result.accepted)
        self.assertIsNone(result.icon_id)
        self.assertIsNone(result.display_name)
        self.assertEqual(result.status, "ambiguous")

    def test_colour_measurement_is_optional(self):
        with_colour = self.towers.recognize(self.circle_crop, method="grayscale")
        without_colour = self.towers.recognize(
            self.circle_crop, method="grayscale", detect_colour=False
        )
        self.assertIsNone(with_colour.colour)
        self.assertIsNone(without_colour.colour)

    def test_wrong_category_records_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "category"):
            TowerRecognizer([self.troop_record])
        with self.assertRaisesRegex(ValueError, "category"):
            TroopRecognizer([self.tower_record])


if __name__ == "__main__":
    unittest.main()
