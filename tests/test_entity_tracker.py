import unittest

from perception.entity_tracker import EntityObservation, EntityTracker


class EntityTrackerTests(unittest.TestCase):
    def observation(self, x=0.1, *, category="tower", identity="Radar", confidence=0.9):
        return EntityObservation(category, (x, 0.1, 0.1, 0.1), identity, confidence)

    def test_assigns_stable_id_when_entity_moves_slightly(self):
        tracker = EntityTracker()
        first = tracker.update([self.observation()])[0]
        second = tracker.update([self.observation(0.11)])[0]
        self.assertEqual(first.track_id, second.track_id)
        self.assertEqual(second.first_seen_frame, 1)
        self.assertEqual(second.last_seen_frame, 2)
        self.assertEqual(second.missed_frames, 0)

    def test_different_categories_never_match(self):
        tracker = EntityTracker()
        first = tracker.update([self.observation(category="tower")])[0]
        second = tracker.update([self.observation(category="unit")])[0]
        self.assertNotEqual(first.track_id, second.track_id)

    def test_conflicting_known_identity_never_matches(self):
        tracker = EntityTracker()
        first = tracker.update([self.observation(identity="Radar")])[0]
        second = tracker.update([self.observation(identity="Factory")])[0]
        self.assertNotEqual(first.track_id, second.track_id)

    def test_one_missing_frame_is_retained_then_reacquired(self):
        tracker = EntityTracker(max_missed_frames=2)
        first = tracker.update([self.observation()])[0]
        missing = tracker.update([])
        self.assertEqual(missing[0].missed_frames, 1)
        reacquired = tracker.update([self.observation(0.11)])[0]
        self.assertEqual(first.track_id, reacquired.track_id)

    def test_track_expires_after_configured_missed_frames(self):
        tracker = EntityTracker(max_missed_frames=1)
        tracker.update([self.observation()])
        tracker.update([])
        expired = tracker.update([])
        self.assertEqual(expired, ())

    def test_snapshot_can_exclude_temporarily_missing_tracks(self):
        tracker = EntityTracker()
        tracker.update([self.observation()])
        tracker.update([])
        self.assertEqual(len(tracker.snapshot()), 1)
        self.assertEqual(tracker.snapshot(include_missing=False), ())

    def test_maximum_cardinality_matching_avoids_greedy_stranding(self):
        tracker = EntityTracker(min_iou=0.0, max_center_distance=0.0)
        # A can match either old track, B can match only the first.
        tracker.update([
            EntityObservation("tower", (0.10, 0.1, 0.2, 0.2), "A"),
            EntityObservation("tower", (0.04, 0.1, 0.2, 0.2), "B"),
        ])
        result = tracker.update([
            EntityObservation("tower", (0.09, 0.1, 0.2, 0.2), "A"),
            EntityObservation("tower", (0.04, 0.1, 0.2, 0.2), "B"),
        ])
        self.assertEqual(len({item.track_id for item in result}), 2)
        self.assertEqual({item.identity: item.track_id for item in result}["A"], "entity-000001")
        self.assertEqual({item.identity: item.track_id for item in result}["B"], "entity-000002")

    def test_invalid_observation_does_not_advance_frame(self):
        tracker = EntityTracker()
        with self.assertRaises(ValueError):
            tracker.update([EntityObservation("building", (0.1, 0.1, 0.2, 0.2))])
        self.assertEqual(tracker.frame, 0)

    def test_reset_restarts_ids_and_frame(self):
        tracker = EntityTracker()
        tracker.update([self.observation()])
        tracker.reset()
        result = tracker.update([self.observation()])
        self.assertEqual(tracker.frame, 1)
        self.assertEqual(result[0].track_id, "entity-000001")


if __name__ == "__main__":
    unittest.main()
