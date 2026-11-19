import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from perception.tracking_replay import discover_screenshots, run_replay


class FakePipeline:
    def __init__(self):
        self.detector = SimpleNamespace(model="fake-model")
        self.calls = []
        self.track = SimpleNamespace(
            track_id="entity-000001",
            category="tower",
            bbox=(0.1, 0.2, 0.1, 0.1),
            identity="radar",
            confidence=0.95,
            first_seen_frame=1,
            last_seen_frame=1,
            missed_frames=0,
        )

    def reset(self):
        self.calls.clear()

    def process(self, screenshot, **options):
        self.calls.append(Path(screenshot).name)
        return (self.track,)

    def snapshot(self, *, include_missing=True):
        return (self.track,)


class TrackingReplayTests(unittest.TestCase):
    def make_image(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (100, 80), (20, 30, 40)).save(path)

    def test_discovery_sorts_supported_images_and_ignores_other_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_image(root / "frame_02.png")
            self.make_image(root / "frame_01.jpg")
            (root / "notes.txt").write_text("ignore")
            paths = discover_screenshots(root)
            self.assertEqual([path.name for path in paths], ["frame_01.jpg", "frame_02.png"])

    def test_replay_processes_images_in_order_and_exports_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "screens"
            self.make_image(source / "02.png")
            self.make_image(source / "01.png")
            output = root / "result" / "replay.json"
            pipeline = FakePipeline()

            report = run_replay(source, output, pipeline=pipeline)

            self.assertEqual(pipeline.calls, ["01.png", "02.png"])
            self.assertEqual(report["frames_processed"], 2)
            self.assertEqual(report["frames"][0]["tracks"][0]["track_id"], "entity-000001")
            self.assertEqual(report["frames"][0]["predicted"][0]["track_id"], "entity-000001")
            self.assertEqual(len(report["frames"][0]["source_sha256"]), 64)
            saved = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(saved["frames"][1]["source"], "02.png")

    def test_optional_annotations_are_written(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "screens"
            self.make_image(source / "01.png")
            annotated = root / "annotated"
            run_replay(source, root / "replay.json", pipeline=FakePipeline(), annotated_dir=annotated)
            images = list(annotated.glob("*.png"))
            self.assertEqual(len(images), 1)
            with Image.open(images[0]) as image:
                self.assertEqual(image.size, (100, 80))

    def test_missing_source_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                discover_screenshots(Path(directory) / "missing")


if __name__ == "__main__":
    unittest.main()
