import unittest
from types import SimpleNamespace

from perception.entity_tracker import EntityTracker
from perception.tracked_entity_pipeline import TrackedEntityPipeline


def detection(
    x=0.1,
    *,
    category="tower",
    accepted=True,
    icon_id="tower_radar",
    display_name="Radar",
    score=0.91,
):
    proposal = SimpleNamespace(
        category=category,
        x=x,
        y=0.1,
        width=0.1,
        height=0.1,
    )
    recognition = SimpleNamespace(
        accepted=accepted,
        icon_id=icon_id if accepted else None,
        display_name=display_name if accepted else None,
        top_score=score,
    )
    return SimpleNamespace(proposal=proposal, recognition=recognition)


class FakeDetector:
    def __init__(self, frames):
        self.frames = list(frames)
        self.calls = []

    def detect(self, screenshot, **options):
        self.calls.append((screenshot, options))
        result = self.frames.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class TrackedEntityPipelineTests(unittest.TestCase):
    def test_keeps_track_id_when_verified_entity_moves(self):
        detector = FakeDetector([
            [detection(x=0.10)],
            [detection(x=0.11)],
        ])
        pipeline = TrackedEntityPipeline(detector=detector, tracker=EntityTracker())
        first = pipeline.process("frame-1")[0]
        second = pipeline.process("frame-2")[0]

        self.assertEqual(first.track_id, second.track_id)
        self.assertEqual(second.identity, "tower_radar")
        self.assertEqual(second.first_seen_frame, 1)
        self.assertEqual(second.last_seen_frame, 2)

    def test_unaccepted_detection_is_tracked_without_invented_identity(self):
        detector = FakeDetector([
            [detection(accepted=False)],
            [detection(x=0.11, accepted=True)],
        ])
        pipeline = TrackedEntityPipeline(detector=detector, tracker=EntityTracker())
        first = pipeline.process("frame-1")[0]
        second = pipeline.process("frame-2")[0]

        self.assertEqual(first.track_id, second.track_id)
        self.assertIsNone(first.identity)
        self.assertEqual(second.identity, "tower_radar")

    def test_conflicting_verified_identity_starts_a_new_track(self):
        detector = FakeDetector([
            [detection(icon_id="tower_radar")],
            [detection(icon_id="tower_factory", display_name="Factory")],
        ])
        pipeline = TrackedEntityPipeline(detector=detector, tracker=EntityTracker())
        first = pipeline.process("frame-1")[0]
        pipeline.process("frame-2")
        visible = pipeline.snapshot(include_missing=False)

        self.assertEqual(len(visible), 1)
        self.assertNotEqual(first.track_id, visible[0].track_id)
        self.assertEqual(visible[0].identity, "tower_factory")

    def test_detector_failure_does_not_advance_tracker_frame(self):
        detector = FakeDetector([
            [detection()],
            RuntimeError("vision request failed"),
        ])
        pipeline = TrackedEntityPipeline(detector=detector, tracker=EntityTracker())
        first = pipeline.process("frame-1")[0]
        with self.assertRaisesRegex(RuntimeError, "vision request failed"):
            pipeline.process("frame-2")

        self.assertEqual(pipeline.tracker.frame, 1)
        self.assertEqual(pipeline.snapshot(include_missing=False)[0].track_id, first.track_id)
        self.assertEqual(len(pipeline.last_detections), 1)

    def test_exposes_per_entity_frame_history(self):
        detector = FakeDetector([
            [detection(x=0.10)],
            [detection(x=0.12)],
            [],
        ])
        pipeline = TrackedEntityPipeline(detector=detector, tracker=EntityTracker())
        first = pipeline.process("frame-1")[0]
        pipeline.process("frame-2")
        pipeline.process("frame-3")

        history = pipeline.history(first.track_id)
        self.assertEqual([entry.frame for entry in history], [1, 2, 3])
        self.assertEqual(history[0].bbox[0], 0.10)
        self.assertEqual(history[1].bbox[0], 0.12)
        self.assertFalse(history[2].visible)

    def test_passes_detector_options_through(self):
        detector = FakeDetector([[]])
        pipeline = TrackedEntityPipeline(detector=detector, tracker=EntityTracker())
        pipeline.process("frame", method="shape", min_score=0.7)
        self.assertEqual(
            detector.calls,
            [("frame", {"method": "shape", "min_score": 0.7})],
        )

    def test_reset_clears_tracks_and_last_frame_data(self):
        detector = FakeDetector([[detection()]])
        pipeline = TrackedEntityPipeline(detector=detector, tracker=EntityTracker())
        pipeline.process("frame")
        pipeline.reset()

        self.assertEqual(pipeline.tracker.frame, 0)
        self.assertEqual(pipeline.snapshot(), ())
        self.assertEqual(pipeline.last_detections, ())
        self.assertEqual(pipeline.last_snapshot, ())


if __name__ == "__main__":
    unittest.main()
