import json
import tempfile
import unittest
from pathlib import Path

from perception.icon_matcher_benchmark import load_labels


class BenchmarkLabelTests(unittest.TestCase):
    def write_manifest(self, directory, document):
        path = Path(directory) / "labels.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def test_accepts_valid_enabled_and_disabled_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "radar.png").write_bytes(b"placeholder")
            manifest = {
                "schema_version": 1,
                "cases": [
                    {
                        "name": "radar",
                        "image": "radar.png",
                        "category": "tower",
                        "expected_name": "Radar",
                    },
                    {
                        "name": "king",
                        "enabled": False,
                        "skip_reason": "Crop needs review",
                    },
                ],
            }
            _, cases = load_labels(self.write_manifest(directory, manifest))
            self.assertEqual(len(cases), 2)

    def test_rejects_unsupported_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_manifest(directory, {"schema_version": 2, "cases": []})
            with self.assertRaisesRegex(ValueError, "schema_version"):
                load_labels(path)

    def test_rejects_duplicate_case_names(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = {
                "schema_version": 1,
                "cases": [{"name": "same"}, {"name": "same"}],
            }
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_labels(self.write_manifest(directory, manifest))

    def test_rejects_unknown_category(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "icon.png").write_bytes(b"x")
            manifest = {
                "schema_version": 1,
                "cases": [{
                    "name": "bad",
                    "image": "icon.png",
                    "category": "building",
                    "expected_name": "House",
                }],
            }
            with self.assertRaisesRegex(ValueError, "category"):
                load_labels(self.write_manifest(directory, manifest))

    def test_rejects_paths_outside_labels_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            outside = parent / "outside.png"
            outside.write_bytes(b"x")
            nested = parent / "dataset"
            nested.mkdir()
            manifest = {
                "schema_version": 1,
                "cases": [{
                    "name": "escape",
                    "image": "../outside.png",
                    "category": "tower",
                    "expected_name": "Radar",
                }],
            }
            with self.assertRaisesRegex(ValueError, "inside"):
                load_labels(self.write_manifest(nested, manifest))


if __name__ == "__main__":
    unittest.main()
