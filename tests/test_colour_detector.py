import unittest
import tempfile
from pathlib import Path

import cv2
import numpy as np

from kiomet_knowledge.recognition.shared.colour_detector import (
    ColourObservation,
    detect_dominant_colour,
)


class ColourDetectorTests(unittest.TestCase):
    def test_detects_dominant_red(self):
        image = np.zeros((40, 40, 3), dtype=np.uint8)
        image[:, :] = (0, 0, 220)
        result = detect_dominant_colour(image)
        self.assertIsInstance(result, ColourObservation)
        self.assertAlmostEqual(result.hue_degrees, 0.0, delta=2.0)
        self.assertEqual(result.hex_rgb, "#DC0000")
        self.assertAlmostEqual(result.pixel_ratio, 1.0)

    def test_detects_blue_in_mixed_crop(self):
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        image[:15, :] = (220, 0, 0)
        image[15:, :] = (0, 220, 0)
        result = detect_dominant_colour(image)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.hue_degrees, 240.0, delta=2.0)
        self.assertGreater(result.pixel_ratio, 0.7)

    def test_ignores_grayscale_crop(self):
        image = np.full((20, 20, 3), 150, dtype=np.uint8)
        self.assertIsNone(detect_dominant_colour(image))

    def test_ignores_transparent_pixels(self):
        image = np.zeros((20, 20, 4), dtype=np.uint8)
        image[:, :10] = (0, 0, 220, 255)
        image[:, 10:] = (255, 0, 255, 0)
        result = detect_dominant_colour(image)
        self.assertIsNotNone(result)
        self.assertEqual(result.hex_rgb, "#DC0000")
        self.assertAlmostEqual(result.pixel_ratio, 1.0)

    def test_accepts_image_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "red.png"
            image = np.zeros((12, 12, 3), dtype=np.uint8)
            image[:, :] = (0, 0, 200)
            self.assertTrue(cv2.imwrite(str(path), image))
            self.assertIsNotNone(detect_dominant_colour(path))

    def test_invalid_thresholds_are_rejected(self):
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            detect_dominant_colour(image, min_saturation=300)
        with self.assertRaises(ValueError):
            detect_dominant_colour(image, hue_tolerance_degrees=0)

    def test_invalid_image_is_rejected(self):
        with self.assertRaises(ValueError):
            detect_dominant_colour(np.array([]))


if __name__ == "__main__":
    unittest.main()
