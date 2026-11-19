import unittest
from pathlib import Path

from kiomet_knowledge.recognition.shared.icon_registry import IconRecord
from kiomet_knowledge.recognition.shared.image_matcher import MatchResult
from kiomet_knowledge.recognition.shared.confidence import assess_matches


def make_match(icon_id, score):
    record = IconRecord(
        id=icon_id,
        category="tower",
        display_name=icon_id,
        internal_name=icon_id,
        asset_category="towers",
        filename=f"{icon_id}.svg",
        svg_path=Path(f"{icon_id}.svg"),
        template_path=Path(f"{icon_id}.png"),
        source="test",
        license="test",
    )
    return MatchResult(record=record, score=score)


class ConfidenceTests(unittest.TestCase):
    def test_empty_matches_return_no_match(self):
        result = assess_matches([])
        self.assertEqual(result.status, "no_match")
        self.assertFalse(result.accepted)
        self.assertIsNone(result.best_match)

    def test_clear_high_score_is_strong(self):
        best = make_match("tower.city", 0.92)
        result = assess_matches([best, make_match("tower.factory", 0.55)])
        self.assertIs(result.best_match, best)
        self.assertEqual(result.status, "strong")
        self.assertTrue(result.accepted)
        self.assertAlmostEqual(result.margin, 0.37)

    def test_close_scores_are_ambiguous(self):
        result = assess_matches([
            make_match("tower.city", 0.91),
            make_match("tower.factory", 0.88),
        ])
        self.assertEqual(result.status, "ambiguous")
        self.assertFalse(result.accepted)

    def test_low_top_score_is_rejected(self):
        result = assess_matches([
            make_match("tower.city", 0.40),
            make_match("tower.factory", 0.10),
        ])
        self.assertEqual(result.status, "low_score")
        self.assertFalse(result.accepted)

    def test_single_candidate_is_not_automatically_accepted(self):
        result = assess_matches([make_match("tower.city", 0.99)])
        self.assertEqual(result.status, "ambiguous")
        self.assertIsNone(result.margin)
        self.assertFalse(result.accepted)

    def test_input_order_does_not_change_best_match(self):
        best = make_match("tower.city", 0.90)
        result = assess_matches([make_match("tower.factory", 0.30), best])
        self.assertIs(result.best_match, best)

    def test_thresholds_are_validated(self):
        with self.assertRaises(ValueError):
            assess_matches([], min_score=1.1)
        with self.assertRaises(ValueError):
            assess_matches([], min_margin=-0.1)


if __name__ == "__main__":
    unittest.main()
