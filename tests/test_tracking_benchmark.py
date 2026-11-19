import unittest

from perception.tracking_benchmark import evaluate_tracking_frames


def entity(entity_id, x=0.1, *, category="tower", track_id=None):
    value = {
        "category": category,
        "bbox": {"x": x, "y": 0.1, "width": 0.1, "height": 0.1},
    }
    if track_id is None:
        value["id"] = entity_id
    else:
        value["track_id"] = track_id
    return value


class TrackingBenchmarkTests(unittest.TestCase):
    def test_perfect_sequence_has_perfect_detection_metrics(self):
        frames = [
            {"frame_index": 1, "expected": [entity("A")], "predicted": [entity("p", track_id="T1")]},
            {"frame_index": 2, "expected": [entity("A", 0.11)], "predicted": [entity("p", 0.11, track_id="T1")]},
        ]
        result = evaluate_tracking_frames(frames)
        self.assertEqual(result["precision"], 1.0)
        self.assertEqual(result["recall"], 1.0)
        self.assertEqual(result["f1"], 1.0)
        self.assertEqual(result["id_switches"], 0)
        self.assertEqual(result["fragmentations"], 0)
        self.assertEqual(result["mean_matched_iou"], 1.0)

    def test_changed_track_id_counts_identity_switch(self):
        frames = [
            {"frame_index": 1, "expected": [entity("A")], "predicted": [entity("p", track_id="T1")]},
            {"frame_index": 2, "expected": [entity("A")], "predicted": [entity("p", track_id="T2")]},
        ]
        result = evaluate_tracking_frames(frames)
        self.assertEqual(result["id_switches"], 1)

    def test_gap_then_return_counts_fragmentation_and_reacquisition(self):
        frames = [
            {"frame_index": 1, "expected": [entity("A")], "predicted": [entity("p", track_id="T1")]},
            {"frame_index": 2, "expected": [entity("A")], "predicted": []},
            {"frame_index": 3, "expected": [entity("A")], "predicted": [entity("p", track_id="T1")]},
        ]
        result = evaluate_tracking_frames(frames)
        self.assertEqual(result["fragmentations"], 1)
        self.assertEqual(result["reacquisitions"], 1)
        self.assertEqual(result["id_switches"], 0)

    def test_gap_followed_by_new_track_is_fragmentation_not_reacquisition(self):
        frames = [
            {"frame_index": 1, "expected": [entity("A")], "predicted": [entity("p", track_id="T1")]},
            {"frame_index": 2, "expected": [entity("A")], "predicted": []},
            {"frame_index": 3, "expected": [entity("A")], "predicted": [entity("p", track_id="T2")]},
        ]
        result = evaluate_tracking_frames(frames)
        self.assertEqual(result["fragmentations"], 1)
        self.assertEqual(result["reacquisitions"], 0)
        self.assertEqual(result["id_switches"], 1)

    def test_counts_false_positives_and_false_negatives(self):
        frames = [{
            "frame_index": 1,
            "expected": [entity("A"), entity("B", 0.5)],
            "predicted": [entity("p", track_id="T1"), entity("extra", 0.8, track_id="T2")],
        }]
        result = evaluate_tracking_frames(frames)
        self.assertEqual(result["true_positives"], 1)
        self.assertEqual(result["false_positives"], 1)
        self.assertEqual(result["false_negatives"], 1)
        self.assertEqual(result["precision"], 0.5)
        self.assertEqual(result["recall"], 0.5)

    def test_category_mismatch_is_not_matched(self):
        frames = [{
            "frame_index": 1,
            "expected": [entity("A", category="tower")],
            "predicted": [entity("p", category="unit", track_id="T1")],
        }]
        result = evaluate_tracking_frames(frames)
        self.assertEqual(result["true_positives"], 0)
        self.assertEqual(result["false_positives"], 1)
        self.assertEqual(result["false_negatives"], 1)

    def test_matching_maximizes_valid_pairs_in_ambiguous_frame(self):
        frames = [{
            "frame_index": 1,
            "expected": [entity("A", 0.1), entity("B", 0.0)],
            "predicted": [
                entity("p1", 0.05, track_id="T1"),
                entity("p2", 0.14, track_id="T2"),
            ],
        }]
        result = evaluate_tracking_frames(frames, iou_threshold=0.3)
        self.assertEqual(result["true_positives"], 2)
        self.assertEqual(result["false_positives"], 0)
        self.assertEqual(result["false_negatives"], 0)

    def test_rejects_non_increasing_frame_indices(self):
        frames = [
            {"frame_index": 2, "expected": [], "predicted": []},
            {"frame_index": 2, "expected": [], "predicted": []},
        ]
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            evaluate_tracking_frames(frames)

    def test_rejects_duplicate_predicted_track_id_in_one_frame(self):
        frames = [{
            "expected": [],
            "predicted": [entity("p1", track_id="T1"), entity("p2", 0.5, track_id="T1")],
        }]
        with self.assertRaisesRegex(ValueError, "unique within each frame"):
            evaluate_tracking_frames(frames)

    def test_rejects_out_of_bounds_normalized_box(self):
        frames = [{
            "expected": [entity("A", 0.95)],
            "predicted": [],
        }]
        with self.assertRaisesRegex(ValueError, "normalized frame bounds"):
            evaluate_tracking_frames(frames)

    def test_rejects_invalid_iou_threshold(self):
        with self.assertRaises(ValueError):
            evaluate_tracking_frames([{"expected": [], "predicted": []}], iou_threshold=0)

    def test_rejects_missing_stable_ground_truth_id(self):
        frames = [{
            "expected": [{"category": "tower", "bbox": {"x": 0.1, "y": 0.1, "width": 0.1, "height": 0.1}}],
            "predicted": [],
        }]
        with self.assertRaisesRegex(ValueError, "stable non-empty id"):
            evaluate_tracking_frames(frames)


if __name__ == "__main__":
    unittest.main()
