import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from perception.tracking_replay import discover_screenshots, run_replay


class FakePipeline:
    def __init__(self):
        self.detector = SimpleNamespace(
            model="fake-model",
            tiled=False,
            max_image_side=960,
            upscale=1.0,
            overlap=0.2,
            num_threads=None,
            min_detector_confidence=0.35,
            timeout=180.0,
        )
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
            self.assertGreaterEqual(report["runtime"]["total_seconds"], 0)
            self.assertGreaterEqual(report["runtime"]["frames_per_second"], 0)
            self.assertGreaterEqual(report["frames"][0]["processing_seconds"], 0)
            self.assertEqual(report["detector"]["max_image_side"], 960)
            self.assertEqual(report["frames"][0]["tracks"][0]["track_id"], "entity-000001")
            self.assertEqual(report["frames"][0]["predicted"][0]["track_id"], "entity-000001")
            self.assertEqual(len(report["frames"][0]["source_sha256"]), 64)
            saved = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(saved["frames"][1]["source"], "02.png")

    def test_replay_exports_raw_detection_and_recognition_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "screens"
            self.make_image(source / "01.png")
            pipeline = FakePipeline()
            candidate = SimpleNamespace(
                icon_id="tower_radar",
                display_name="Radar",
                score=0.91,
            )
            recognition = SimpleNamespace(
                accepted=False,
                status="ambiguous",
                icon_id=None,
                display_name=None,
                top_score=0.91,
                margin=0.02,
                candidates=(candidate,),
            )
            proposal = SimpleNamespace(
                region_id="full-1",
                category="tower",
                x=0.2,
                y=0.3,
                width=0.04,
                height=0.05,
            )
            pipeline.last_detections = (
                SimpleNamespace(
                    proposal=proposal,
                    tile_name="full",
                    detector_confidence=0.88,
                    recognition=recognition,
                ),
            )

            report = run_replay(
                source,
                root / "replay.json",
                pipeline=pipeline,
            )

            detection = report["frames"][0]["detections"][0]
            self.assertEqual(detection["region_id"], "full-1")
            self.assertEqual(detection["bbox"]["x"], 0.2)
            self.assertEqual(detection["detector_confidence"], 0.88)
            self.assertFalse(detection["recognition"]["accepted"])
            self.assertEqual(detection["recognition"]["status"], "ambiguous")
            self.assertEqual(
                detection["recognition"]["candidates"][0]["icon_id"],
                "tower_radar",
            )

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
