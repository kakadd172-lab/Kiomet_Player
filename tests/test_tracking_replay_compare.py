import json
import tempfile
import unittest
from pathlib import Path

from perception.tracking_replay_compare import compare_replays


def replay(path, *, detections, runtime, model, hashes=None):
    frames = []
    hashes = hashes or ["a" * 64, "b" * 64]
    for index, (count, source_hash) in enumerate(zip(detections, hashes), start=1):
        frames.append({
            "frame_index": index,
            "source": f"frame_{index}.png",
            "source_sha256": source_hash,
            "processing_seconds": 0.5 * index,
            "predicted": [
                {
                    "track_id": f"T{item}",
                    "category": "tower",
                    "identity": "radar",
                }
                for item in range(count)
            ],
            "temporarily_missing_tracks": [],
        })
    path.write_text(json.dumps({
        "frames": frames,
        "detector": {"model": model, "tiled": False},
        "runtime": {"total_seconds": runtime},
    }), encoding="utf-8")


class TrackingReplayCompareTests(unittest.TestCase):
    def test_compares_identical_source_sequence_and_reports_deltas(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "first.json", root / "second.json"
            replay(first, detections=[1, 1], runtime=10.0, model="model-a")
            replay(second, detections=[2, 1], runtime=12.0, model="model-b")

            result = compare_replays(first, second)

            self.assertTrue(result["same_source_sequence"])
            self.assertEqual(result["source_frames"], 2)
            self.assertEqual(result["delta_second_minus_first"]["total_detections"], 1)
            self.assertEqual(result["delta_second_minus_first"]["runtime_percent_change"], 20.0)
            self.assertEqual(result["configuration_differences"]["model"], {
                "first": "model-a",
                "second": "model-b",
            })
            self.assertEqual(result["per_frame_comparison"][0]["detection_delta"], 1)

    def test_rejects_different_source_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "first.json", root / "second.json"
            replay(first, detections=[1, 1], runtime=10.0, model="a")
            replay(second, detections=[1, 1], runtime=10.0, model="a", hashes=["c" * 64, "d" * 64])
            with self.assertRaisesRegex(ValueError, "source SHA256"):
                compare_replays(first, second)

    def test_rejects_duplicate_track_ids_in_a_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "first.json", root / "second.json"
            replay(first, detections=[1, 1], runtime=10.0, model="a")
            replay(second, detections=[1, 1], runtime=10.0, model="b")
            data = json.loads(second.read_text(encoding="utf-8"))
            data["frames"][0]["predicted"].append(data["frames"][0]["predicted"][0])
            second.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate track IDs"):
                compare_replays(first, second)

    def test_rejects_invalid_replay_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "first.json", root / "second.json"
            first.write_text("{}", encoding="utf-8")
            replay(second, detections=[1, 1], runtime=10.0, model="b")
            with self.assertRaisesRegex(ValueError, "not a tracking replay"):
                compare_replays(first, second)


if __name__ == "__main__":
    unittest.main()
