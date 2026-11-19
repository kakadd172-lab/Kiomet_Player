import unittest

from kiomet_knowledge.recognition.knowledge_loader import (
    KNOWLEDGE,
    load_knowledge,
)


class KnowledgeLoaderTests(unittest.TestCase):
    def test_expected_catalogue_counts(self):
        self.assertEqual(
            KNOWLEDGE.summary(),
            {"towers": 27, "troops": 10, "total": 37},
        )

    def test_tower_lookup_accepts_display_name_and_id(self):
        by_name = KNOWLEDGE.get_tower("Headquarters")
        by_id = KNOWLEDGE.get_tower("tower.headquarters")

        self.assertIsNotNone(by_name)
        self.assertIs(by_name, by_id)
        self.assertEqual(by_name.display_name, "Headquarters")

    def test_troop_lookup_accepts_name_and_id(self):
        by_name = KNOWLEDGE.get_troop("Tank")
        by_id = KNOWLEDGE.get_troop("unit.tank")

        self.assertIsNotNone(by_name)
        self.assertIs(by_name, by_id)
        self.assertEqual(by_name.category, "unit")

    def test_category_lookup_is_explicit(self):
        self.assertIsNotNone(KNOWLEDGE.get_icon("tower", "City"))
        self.assertIsNotNone(KNOWLEDGE.get_icon("unit", "Soldier"))
        self.assertIsNone(KNOWLEDGE.get_icon("building", "City"))

    def test_unknown_names_return_none(self):
        self.assertIsNone(KNOWLEDGE.get_tower("not-a-tower"))
        self.assertIsNone(KNOWLEDGE.get_troop("not-a-unit"))

    def test_loader_returns_independent_wrapper(self):
        loaded = load_knowledge()
        self.assertEqual(loaded.summary(), KNOWLEDGE.summary())
        self.assertIsNot(loaded, KNOWLEDGE)


if __name__ == "__main__":
    unittest.main()
